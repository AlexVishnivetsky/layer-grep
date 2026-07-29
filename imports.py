from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node, Parser

from chunker import EXTENSION_LANGS, PY_LANG, RUST_LANG

# Bump on any change to extract_python_imports()/extract_js_imports()/extract_rust_imports()/
# extract_c_imports() that would resolve the same source differently, or when a new extension
# gains import-graph support (existing already-hashed-unchanged files of that extension would
# otherwise never get (re)processed - see db_schema._wipe_for_imports_version(), which reads
# this same list) - mirrors chunker.CHUNKER_VERSION's role for code chunking.
IMPORTS_VERSION = 7

# Every extension extract_imports() knows how to handle - the single source of truth for
# both indexer.py's per-file dispatch and db_schema.py's version-migration wipe (which needs
# to know which already-indexed files must be force-reprocessed, not just re-derive its own
# copy of this list and risk it drifting out of sync).
IMPORT_GRAPH_EXTENSIONS = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".rs", ".c", ".h",
    ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx",
})


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


# --- Rust: `use` resolution needs a real per-crate module tree ------------------------------
#
# Unlike Python (module path = directory path, always) and JS/TS (specifier always relative
# to the importing file), Rust's `crate::`/`self::`/`super::` paths are relative to each
# file's *position in its crate's module tree* - which is built from `mod` declarations, not
# directory structure. A file nobody ever declares `mod` for isn't part of any tree at all
# (dead code, from the compiler's perspective), and a file named `foo.rs` hosts its own
# submodules in a sibling directory `foo/`, not in `foo.rs`'s own (nonexistent) directory.

_RUST_SCAN_EXCLUDED_DIRS = frozenset({"target", ".git"})


@dataclass(frozen=True)
class _RustModuleTree:
    module_to_file: dict[str, Path]
    file_to_module: dict[Path, str]


@dataclass(frozen=True)
class _RustProject:
    # crate_name (hyphens normalized to underscores, matching how Rust source actually
    # spells them) -> its [lib]-rooted tree - the only thing another file can `use
    # that_name::...` into, since nothing outside a crate can reference its [[bin]] targets
    lib_crates: dict[str, _RustModuleTree]
    # every file reachable from ANY root (lib or a [[bin]]) -> the tree it belongs to, for
    # resolving crate::/self::/super:: from that specific file regardless of which kind of
    # root its compilation unit has
    file_trees: dict[Path, _RustModuleTree]


def _resolve_rust_mod_file(
    containing_file: Path, is_file_root: bool, inline_segments: list[str], mod_name: str,
) -> Path | None:
    """Where `mod <mod_name>;` (a bodyless, file-based module declaration) points. A file's
    own submodules normally live in a directory named after its stem, sitting next to it -
    *except* when the file itself is a "root" for directory-nesting purposes, in which case
    its own directory already plays that role directly. That's true for mod.rs (nested
    directory root) and for whichever file is *this compilation unit's own* crate/bin root
    (module_path == "") - which isn't decided by filename, since a `[[bin]] path = ...` can
    point at any name (e.g. rustdesk's src/naming.rs is a root exactly like main.rs would be).
    `inline_segments` accounts for a bodyless `mod` declared *inside* an inline `mod outer {
    ... }` block within the same file - rare, but its child file still nests one directory
    level per inline ancestor, same as if each were its own file."""
    host_dir = containing_file.parent if is_file_root else containing_file.parent / containing_file.stem
    for segment in inline_segments:
        host_dir = host_dir / segment
    candidate = host_dir / f"{mod_name}.rs"
    if candidate.is_file():
        return candidate
    candidate = host_dir / mod_name / "mod.rs"
    return candidate if candidate.is_file() else None


def _walk_rust_mod_tree(
    file: Path, module_path: str,
    module_to_file: dict[str, Path], file_to_module: dict[Path, str],
    visited: set[Path],
) -> None:
    """Populates module_to_file/file_to_module by following `mod` declarations from `file`
    (a crate/bin root, or a submodule reached from one) - NOT by walking the filesystem, so a
    file no `mod` anywhere ever names simply never appears in either map (accurate to how
    rustc itself decides what's part of the crate). `visited` guards against a pathological
    mod cycle (shouldn't happen in valid Rust, but a missing check would infinite-loop on one
    rather than just producing a wrong-but-harmless partial tree)."""
    if file in visited or not file.is_file():
        return
    visited.add(file)
    module_to_file[module_path] = file
    file_to_module[file] = module_path
    is_file_root = module_path == "" or file.name == "mod.rs"

    source = file.read_bytes()
    tree = Parser(RUST_LANG.language).parse(source)

    def scan(container_children: list[Node], prefix: str, inline_segments: list[str]) -> None:
        for child in container_children:
            if child.type != "mod_item":
                continue
            name_node = child.child_by_field_name("name")
            if name_node is None:
                continue
            mod_name = _text(name_node, source)
            child_path = f"{prefix}::{mod_name}" if prefix else mod_name
            body = child.child_by_field_name("body")
            if body is not None:
                # inline `mod foo { ... }` - the submodule's content lives in this SAME file,
                # just under its own path; still need to recurse for further nested mods
                module_to_file[child_path] = file
                scan(body.named_children, child_path, [*inline_segments, mod_name])
            else:
                child_file = _resolve_rust_mod_file(file, is_file_root, inline_segments, mod_name)
                if child_file is not None:
                    _walk_rust_mod_tree(child_file, child_path, module_to_file, file_to_module, visited)

    scan(tree.root_node.named_children, module_path, [])


def _normalize_crate_name(name: str) -> str:
    # Cargo.toml package names are conventionally hyphenated, but Rust source can only ever
    # reference a crate by an identifier - rustc silently treats "-" as "_" for this purpose,
    # so "rustdesk-portable-packer" is `use rustdesk_portable_packer::...` in actual code
    return name.replace("-", "_")


def _build_rust_project(project_root: Path) -> _RustProject:
    import tomllib
    lib_crates: dict[str, _RustModuleTree] = {}
    file_trees: dict[Path, _RustModuleTree] = {}

    for cargo_path in sorted(project_root.rglob("Cargo.toml")):
        if any(part in _RUST_SCAN_EXCLUDED_DIRS for part in cargo_path.relative_to(project_root).parts):
            continue
        try:
            with cargo_path.open("rb") as f:
                data = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            continue
        package = data.get("package", {})
        raw_name = package.get("name")
        if not raw_name:
            continue
        crate_dir = cargo_path.parent

        lib_path = data.get("lib", {}).get("path")
        lib_root = crate_dir / lib_path if lib_path else crate_dir / "src" / "lib.rs"
        if lib_root.is_file():
            module_to_file: dict[str, Path] = {}
            file_to_module: dict[Path, str] = {}
            _walk_rust_mod_tree(lib_root, "", module_to_file, file_to_module, set())
            tree = _RustModuleTree(module_to_file, file_to_module)
            lib_crates[_normalize_crate_name(raw_name)] = tree
            file_trees.update(dict.fromkeys(file_to_module, tree))

        bin_entries = data.get("bin", [])
        if bin_entries:
            bin_roots = [
                crate_dir / entry["path"] if entry.get("path") else crate_dir / "src" / "bin" / f"{entry['name']}.rs"
                for entry in bin_entries if entry.get("name")
            ]
        else:
            # no explicit [[bin]] list - Cargo's own default is src/main.rs, if present
            default_main = crate_dir / "src" / "main.rs"
            bin_roots = [default_main] if default_main.is_file() else []

        for bin_root in bin_roots:
            if not bin_root.is_file() or bin_root in file_trees:
                continue
            module_to_file = {}
            file_to_module = {}
            _walk_rust_mod_tree(bin_root, "", module_to_file, file_to_module, set())
            tree = _RustModuleTree(module_to_file, file_to_module)
            file_trees.update(dict.fromkeys(file_to_module, tree))

    return _RustProject(lib_crates, file_trees)


_RUST_PROJECT_CACHE: dict[Path, _RustProject] = {}


def _rust_project(project_root: Path) -> _RustProject:
    """Cached per (resolved) project_root for the process's lifetime - building the module
    tree means walking every Cargo.toml and every reachable .rs file in the project, real
    work worth doing once per indexing run rather than once per file. Same "rebuild on next
    process start" tradeoff project_config._PROJECT_CONFIG_CACHE already accepts for the
    long-lived MCP server: restructuring `mod` declarations mid-session needs a server
    restart to be picked up, not just a re-index."""
    resolved = project_root.resolve()
    if resolved not in _RUST_PROJECT_CACHE:
        _RUST_PROJECT_CACHE[resolved] = _build_rust_project(resolved)
    return _RUST_PROJECT_CACHE[resolved]


def _use_paths(node: Node, source: bytes) -> list[list[str]]:
    """All the full `::`-segment paths named by one `use` tree node - one entry per leaf
    item, handling every shape the grammar produces: a bare/scoped path, a `{...}` group
    (recursively - `use foo::{bar::{a, b}, baz}` is valid), a `*` wildcard, an `as` alias
    (resolved by its pre-alias name - the alias doesn't change what file it points to), and
    `self` inside a group (meaning "the group's own base path", not an extra path segment)."""
    if node.type in ("identifier", "crate", "self", "super"):
        return [[_text(node, source)]]
    if node.type == "scoped_identifier":
        path_node = node.child_by_field_name("path")
        name_node = node.child_by_field_name("name")
        if path_node is None or name_node is None:
            return []
        return [p + [_text(name_node, source)] for p in _use_paths(path_node, source)]
    if node.type == "use_as_clause":
        path_node = node.child_by_field_name("path")
        return _use_paths(path_node, source) if path_node is not None else []
    if node.type == "use_wildcard":
        base_node = node.named_children[0] if node.named_children else None
        return _use_paths(base_node, source) if base_node is not None else []
    if node.type == "use_list":
        results = []
        for c in node.named_children:
            results.extend(_use_paths(c, source))
        return results
    if node.type == "scoped_use_list":
        path_node = node.child_by_field_name("path")
        list_node = node.child_by_field_name("list")
        if path_node is None or list_node is None:
            return []
        bases = _use_paths(path_node, source)
        tails = _use_paths(list_node, source)
        return [base if tail == ["self"] else base + tail for base in bases for tail in tails]
    return []


def _lookup_module_prefix(segments: list[str], module_to_file: dict[str, Path]) -> Path | None:
    """The deepest known module along `segments` - `use crate::foo::Bar` where "foo" is a
    real submodule but "Bar" is just a struct name inside it (not itself a submodule) must
    still resolve to foo's file, the same "walk until it stops being a module" logic
    extract_python_imports._resolve_dotted() uses for dotted Python imports."""
    for i in range(len(segments), -1, -1):
        candidate = "::".join(segments[:i])
        if candidate in module_to_file:
            return module_to_file[candidate]
    return None


def _resolve_rust_path(segments: list[str], from_file: Path, project: _RustProject) -> Path | None:
    if not segments:
        return None
    head, rest = segments[0], segments[1:]
    tree = project.file_trees.get(from_file)

    if head == "crate":
        return _lookup_module_prefix(rest, tree.module_to_file) if tree is not None else None
    if head in ("self", "super"):
        if tree is None:
            return None
        current_path = tree.file_to_module.get(from_file)
        if current_path is None:
            return None
        base = current_path.split("::") if current_path else []
        if head == "super":
            if not base:
                return None
            base = base[:-1]
        return _lookup_module_prefix(base + rest, tree.module_to_file)
    if head in project.lib_crates:
        # covers both "referring to a sibling crate by name" and "referring to your own
        # crate by name instead of `crate::`" (2018-edition-style) uniformly - if `head` is
        # this file's own crate name, project.lib_crates[head] just happens to be `tree`
        return _lookup_module_prefix(rest, project.lib_crates[head].module_to_file)
    return None  # std/external dependency, or a crate not found anywhere in this project


def extract_rust_imports(path: Path, project_root: Path) -> list[ImportEdge]:
    """`use` declarations, resolved through a real per-crate module tree (see above) rather
    than a directory-path guess. Doesn't cover `#[path = "..."]`-relocated modules or
    `#[cfg(...)]`-gated alternate `mod` bodies - both are edge cases a first pass can afford
    to leave unresolved (same "give up, no edge" treatment as an external dependency)."""
    project = _rust_project(project_root)
    source = path.read_bytes()
    tree = Parser(RUST_LANG.language).parse(source)

    edges: list[ImportEdge] = []

    def walk(node: Node) -> None:
        if node.type == "use_declaration":
            inner = node.named_children[0] if node.named_children else None
            if inner is not None:
                for segments in _use_paths(inner, source):
                    target = _resolve_rust_path(segments, path, project)
                    if target is not None:
                        edges.append(ImportEdge(path, target, "::".join(segments)))
        for c in node.children:
            walk(c)

    walk(tree.root_node)
    return edges


_C_FAMILY_EXTENSIONS = frozenset({".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"})
_C_SCAN_EXCLUDED_DIRS = frozenset({".git", "node_modules"})


def _build_c_basename_index(project_root: Path) -> dict[str, list[Path]]:
    """Every C/C++ file's bare filename -> every path in the project bearing it - built once
    per project (see _c_basename_index's cache) to back _resolve_c_include's fallback."""
    index: dict[str, list[Path]] = {}
    for p in project_root.rglob("*"):
        if any(part in _C_SCAN_EXCLUDED_DIRS for part in p.relative_to(project_root).parts):
            continue
        if p.is_file() and p.suffix.lower() in _C_FAMILY_EXTENSIONS:
            index.setdefault(p.name, []).append(p)
    return index


_C_BASENAME_INDEX_CACHE: dict[Path, dict[str, list[Path]]] = {}


def _c_basename_index(project_root: Path) -> dict[str, list[Path]]:
    """Cached per (resolved) project_root for the process's lifetime - same "rebuild on next
    process start" tradeoff as _rust_project/project_config._PROJECT_CONFIG_CACHE for the
    long-lived MCP server: adding/removing/renaming a file mid-session needs a restart to be
    picked up here, not just a re-index."""
    resolved = project_root.resolve()
    if resolved not in _C_BASENAME_INDEX_CACHE:
        _C_BASENAME_INDEX_CACHE[resolved] = _build_c_basename_index(resolved)
    return _C_BASENAME_INDEX_CACHE[resolved]


def _resolve_c_include(specifier: str, from_file: Path, project_root: Path) -> Path | None:
    """Quoted `#include "foo.h"` - resolved relative to the including file's own directory,
    the first place a real C compiler looks for a quoted include before falling back to any
    project-specific `-I` search paths. Those extra search paths aren't discoverable from the
    source tree alone (build-system-specific compiler flags, not something layergrep has
    visibility into) directly - but a very common real-world instance of this (e.g. Unreal
    Engine's Public/Private module split, where a .cpp in Private/ includes its sibling header
    by bare name, relying on the module's own Public/ dir being one of its registered include
    paths - confirmed on a real UE project, issue #22) can be recovered without knowing the
    actual search path at all: fall back to matching by bare filename anywhere in the project.
    Only used when there's EXACTLY ONE file in the whole project with that basename - if there
    are several (a real, if less common, occurrence - e.g. abseil-cpp has three unrelated
    config.h files across different subsystems), picking one would risk a confidently wrong
    edge, so this deliberately gives up rather than guesses, same "give up, no edge" precedent
    as an external Python package or an unresolved JS bare specifier."""
    candidate = (from_file.parent / specifier).resolve()
    if candidate.is_file():
        return candidate
    basename = Path(specifier).name
    matches = _c_basename_index(project_root).get(basename, [])
    return matches[0] if len(matches) == 1 else None


def extract_c_imports(path: Path, project_root: Path) -> list[ImportEdge]:
    """Quoted `#include "foo.h"` only - resolved via `preproc_include`'s `path` field. No
    extension-guessing needed unlike JS/TS (`#include` already names the real file's exact
    extension). Angle-bracket `#include <foo.h>` (system/library headers via the compiler's
    own search path) is left unresolved, same treatment as an external Python package - no
    project-agnostic way to know a compiler's system include paths.

    Shared verbatim between C and C++ (issue #22): `#include` is identical across both
    languages (a preprocessor-level construct, not a language-level one), and tree-sitter-c/
    tree-sitter-cpp produce the exact same `preproc_include`/`string_literal`/
    `system_lib_string` shape for it - verified directly against the real grammar, not
    assumed. `EXTENSION_LANGS` (not a hardcoded `C_LANG`) picks the right grammar for
    whichever extension this file actually is, so this one function serves both dispatch
    entries below without duplicating a byte of logic. C++ has no module-per-file mapping
    the way Rust's `use` does (barring C++20 modules - a genuinely different, much newer
    mechanism real-world C++ overwhelmingly doesn't use yet) - so no per-file tree is needed
    here, unlike extract_rust_imports.

    `project_root` backs _resolve_c_include's bare-filename fallback (see its own docstring)
    for the common case where a quoted include isn't actually relative to the including
    file, but to an extra compiler search path this function has no other way to discover."""
    source = path.read_bytes()
    parser = Parser(EXTENSION_LANGS[path.suffix.lower()].language)
    tree = parser.parse(source)

    edges: list[ImportEdge] = []

    def walk(node: Node) -> None:
        if node.type == "preproc_include":
            path_node = node.child_by_field_name("path")
            if path_node is not None and path_node.type == "string_literal":
                content = next((c for c in path_node.children if c.type == "string_content"), None)
                if content is not None:
                    specifier = _text(content, source)
                    target = _resolve_c_include(specifier, path, project_root)
                    if target is not None:
                        edges.append(ImportEdge(path, target, specifier))
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
    if suffix == ".rs":
        return extract_rust_imports(path, project_root)
    if suffix in (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"):
        return extract_c_imports(path, project_root)
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
