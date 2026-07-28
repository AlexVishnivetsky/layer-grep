from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node, Parser

from chunker import EXTENSION_LANGS, PY_LANG

# Bump on any change to extract_python_imports()/extract_js_imports() that would resolve the
# same source differently, or when a new extension gains import-graph support (existing
# already-hashed-unchanged files of that extension would otherwise never get (re)processed -
# see db_schema._wipe_for_imports_version(), which reads this same list) - mirrors
# chunker.CHUNKER_VERSION's role for code chunking.
IMPORTS_VERSION = 3

# Every extension extract_imports() knows how to handle - the single source of truth for
# both indexer.py's per-file dispatch and db_schema.py's version-migration wipe (which needs
# to know which already-indexed files must be force-reprocessed, not just re-derive its own
# copy of this list and risk it drifting out of sync).
IMPORT_GRAPH_EXTENSIONS = frozenset({".py", ".js", ".jsx", ".ts", ".tsx"})


@dataclass(frozen=True)
class ImportEdge:
    source: Path
    target: Path  # resolved file inside project_root
    module: str   # as written in the source, e.g. "app.api.account.helpers" or ".bar"


def _text(node: Node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8")


def _resolve_dotted(parts: list[str], base: Path) -> Path | None:
    """`a.b.c` under `base`: walk directories part by part, a `.py` file ends the walk early
    (remaining parts, if any, are names inside that module) - if all parts are consumed and
    it's a directory, fall back to its `__init__.py`."""
    path = base
    for part in parts:
        path = path / part
        file_candidate = path.with_suffix(".py")
        if file_candidate.is_file():
            return file_candidate
        if not path.is_dir():
            return None
    init_file = path / "__init__.py"
    return init_file if init_file.is_file() else None


def _resolve_from_names(base_dir: Path, names: list[str]) -> Path | None:
    """`from <module> import a, b` where <module> resolved to a package dir: prefer a
    submodule file matching one of the imported names over the package's __init__.py."""
    for name in names:
        candidate = base_dir / f"{name}.py"
        if candidate.is_file():
            return candidate
    init_file = base_dir / "__init__.py"
    return init_file if init_file.is_file() else None


def _relative_base(current_file: Path, level: int) -> Path:
    package_dir = current_file.parent
    for _ in range(level - 1):
        package_dir = package_dir.parent
    return package_dir


def _imported_names(node: Node, source: bytes) -> list[str]:
    names = []
    for n in node.children_by_field_name("name"):
        if n.type == "dotted_name":
            names.append(_text(n, source))
        elif n.type == "aliased_import":
            inner = n.child_by_field_name("name")
            if inner is not None:
                names.append(_text(inner, source))
    return names


def extract_python_imports(path: Path, project_root: Path) -> list[ImportEdge]:
    source = path.read_bytes()
    parser = Parser(PY_LANG.language)
    tree = parser.parse(source)

    edges: list[ImportEdge] = []

    def add_edge(target: Path | None, module_display: str) -> None:
        if target is not None:
            edges.append(ImportEdge(path, target, module_display))

    def handle_import_statement(node: Node) -> None:
        for child in node.named_children:
            if child.type == "dotted_name":
                parts = _text(child, source).split(".")
                add_edge(_resolve_dotted(parts, project_root), ".".join(parts))
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                if name_node is not None:
                    parts = _text(name_node, source).split(".")
                    add_edge(_resolve_dotted(parts, project_root), ".".join(parts))

    def handle_import_from_statement(node: Node) -> None:
        module_node = node.child_by_field_name("module_name")
        if module_node is None:
            return
        names = _imported_names(node, source)

        if module_node.type == "dotted_name":
            parts = _text(module_node, source).split(".")
            base = project_root
            module_display = ".".join(parts)
        elif module_node.type == "relative_import":
            prefix = next((c for c in module_node.named_children if c.type == "import_prefix"), None)
            level = _text(prefix, source).count(".") if prefix is not None else 1
            remainder = next((c for c in module_node.named_children if c.type == "dotted_name"), None)
            parts = _text(remainder, source).split(".") if remainder is not None else []
            base = _relative_base(path, level)
            module_display = "." * level + ".".join(parts)
        else:
            return

        if parts:
            target = _resolve_dotted(parts, base)
        else:
            init_file = base / "__init__.py"
            target = init_file if init_file.is_file() else None

        if target is not None and target.name == "__init__.py" and names:
            refined = _resolve_from_names(target.parent, names)
            if refined is not None:
                target = refined

        add_edge(target, module_display)

    def walk(node: Node) -> None:
        if node.type == "import_statement":
            handle_import_statement(node)
        elif node.type == "import_from_statement":
            handle_import_from_statement(node)
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return edges


# `./foo` (no extension) can refer to any of these, or to `./foo/index.<ext>` if it's a
# directory - order matches a TS-aware bundler's usual preference (prefer a same-language
# source file over the compiled/plain-JS sibling that's often committed alongside it).
_JS_EXTENSIONS_TO_TRY = (".ts", ".tsx", ".js", ".jsx")


def _string_fragment_text(string_node: Node, source: bytes) -> str | None:
    """The unquoted contents of a `string` node - going through its `string_fragment` child
    instead of slicing off the first/last byte handles single/double-quoted strings the same
    way without caring which quote character was actually used."""
    frag = next((c for c in string_node.children if c.type == "string_fragment"), None)
    return _text(frag, source) if frag is not None else None


def _resolve_js_specifier(specifier: str, from_file: Path) -> Path | None:
    """Only relative specifiers (`./foo`, `../bar`) resolve to a real file - a bare specifier
    (`react`, `lodash`) is an external package the same way an unresolvable dotted Python
    import is external, and a path-alias specifier (tsconfig `baseUrl`/webpack alias) would
    need build-tool-specific config this function has no way to discover project-agnostically
    - both are deliberately left unresolved rather than guessed at."""
    if not specifier.startswith("."):
        return None
    base = (from_file.parent / specifier).resolve()
    if base.is_file():
        return base
    for ext in _JS_EXTENSIONS_TO_TRY:
        candidate = base.parent / (base.name + ext)
        if candidate.is_file():
            return candidate
    if base.is_dir():
        for ext in _JS_EXTENSIONS_TO_TRY:
            candidate = base / f"index{ext}"
            if candidate.is_file():
                return candidate
    return None


def extract_js_imports(path: Path, project_root: Path) -> list[ImportEdge]:
    """`import`/`export ... from` (default, named, namespace, `import type`/`export type` -
    all of these share the same `import_statement`/`export_statement` node shape with a
    `source` field, so no special-casing needed per form) and CommonJS `require(...)` calls.
    Doesn't cover the rarer TS `import X = require(...)` legacy interop form.

    `project_root` is unused here (unlike extract_python_imports, JS/TS specifiers only ever
    resolve relative to the importing file, never to project_root) - kept in the signature
    so extract_imports() can dispatch to either extractor uniformly."""
    spec = EXTENSION_LANGS[path.suffix.lower()]
    source = path.read_bytes()
    parser = Parser(spec.language)
    tree = parser.parse(source)

    edges: list[ImportEdge] = []

    def add_edge(string_node: Node) -> None:
        if string_node.type != "string":
            return
        specifier = _string_fragment_text(string_node, source)
        if not specifier:
            return
        target = _resolve_js_specifier(specifier, path)
        if target is not None:
            edges.append(ImportEdge(path, target, specifier))

    def walk(node: Node) -> None:
        if node.type in ("import_statement", "export_statement"):
            source_node = node.child_by_field_name("source")
            if source_node is not None:
                add_edge(source_node)
        elif node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type == "identifier" and _text(fn, source) == "require":
                args = node.child_by_field_name("arguments")
                first_arg = next((c for c in args.named_children if c.type == "string"), None) if args else None
                if first_arg is not None:
                    add_edge(first_arg)
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return edges


def extract_imports(path: Path, project_root: Path) -> list[ImportEdge]:
    """Single entry point indexer.py calls - dispatches to the right language-specific
    extractor by extension, so indexer.py doesn't need its own per-language branching (and
    can't drift out of sync with IMPORT_GRAPH_EXTENSIONS about which extensions are covered)."""
    suffix = path.suffix.lower()
    if suffix == ".py":
        return extract_python_imports(path, project_root)
    if suffix in (".js", ".jsx", ".ts", ".tsx"):
        return extract_js_imports(path, project_root)
    return []


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    project_root = Path(sys.argv[1])
    target_file = Path(sys.argv[2])

    for edge in extract_imports(target_file, project_root):
        status = "OK" if edge.target.is_file() else "MISSING"
        try:
            rel = edge.target.relative_to(project_root)
        except ValueError:
            rel = edge.target
        print(f"  {edge.module!r:60s} -> {rel}  [{status}]")


if __name__ == "__main__":
    main()
