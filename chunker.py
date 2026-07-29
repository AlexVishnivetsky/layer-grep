from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import tree_sitter_c as tsc
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
import tree_sitter_typescript as tstypescript
from tree_sitter import Language, Node, Parser

# Bump whenever chunking logic changes in a way that affects existing chunks (bug fixes,
# new node handling, etc.) - indexer.py compares this against what a .db was built with and
# forces a full reindex on mismatch, since file-content hashes alone can't detect that a def
# unchanged in the codebase should now be re-chunked differently.
CHUNKER_VERSION = 8

FUNCTION_VALUE_TYPES = frozenset({"arrow_function", "function_expression", "function"})

# Module-level statements worth chunking as data even though they're not a function/class -
# e.g. a constants file that's wall-to-wall `NAME = 'value'` with zero defs is otherwise
# invisible to search entirely. "expression_statement" is shared across languages (both
# grammars use it for a bare `x = y` / `window.x = y` used as a statement). "with_statement"
# covers route-registration files like `with router('/some_path'): router(...,
# name='...')` (Python only) - these have zero functions/classes too, but the route paths
# and handler names inside are exactly what a literal-grep query is likely to contain.
SWEEPABLE_TOP_LEVEL_TYPES = frozenset({
    "expression_statement", "with_statement", "const_item", "static_item",
    # C: `#define X 10` (real macro constants and the near-universal header-guard `#define`
    # alike - not worth distinguishing, see TRANSPARENT_CONTAINER_TYPES docstring below),
    # plain top-level `declaration` (globals, `extern` decls, function prototypes - a
    # prototype has no body to chunk as a function, it's data about the API surface, same
    # spirit as a Rust const_item), and a non-container `typedef` alias (`typedef int
    # MyInt;` - `typedef struct {...} X;` never reaches here, see wrapper_types on C_LANG).
    "preproc_def", "declaration", "type_definition",
})
# Never swept into a data block, and don't break one either - just skipped
IGNORED_TOP_LEVEL_TYPES = frozenset({
    "import_statement", "import_from_statement", "future_import_statement", "use_declaration",
    "preproc_include",
    # Recovery artifact from a preprocessor directive appearing where a declaration was
    # expected (seen with `#ifdef __cplusplus` / `extern "C" { ... }` nesting where the
    # inner guard's #endif lands inside the linkage_specification's body) - not real content.
    "preproc_call",
})


@dataclass(frozen=True)
class LangSpec:
    language: Language
    function_types: frozenset[str]        # top-level `function foo() {}` / `def foo():`
    class_types: frozenset[str]            # top-level `class Foo` / a container whose body can hold methods
    wrapper_types: frozenset[str]          # wraps a def: decorator (py) / export (js/ts) - unwrap to find it
    method_types: frozenset[str]           # method nodes directly in a class body
    field_fn_types: frozenset[str] = field(default_factory=frozenset)   # class fields that may hold a function value
    top_level_assign_types: frozenset[str] = field(default_factory=frozenset)  # `const X = () => {}` at module level
    # node types that behave like a leading comment when found directly above a def (doc
    # comments, and - for Rust - `#[attr]` nodes, which are separate sibling nodes rather
    # than a wrapper around the item they annotate)
    annotation_types: frozenset[str] = field(default_factory=lambda: frozenset({"comment"}))


PY_LANG = LangSpec(
    language=Language(tspython.language()),
    function_types=frozenset({"function_definition"}),
    class_types=frozenset({"class_definition"}),
    wrapper_types=frozenset({"decorated_definition"}),
    method_types=frozenset({"function_definition"}),
)

_JS_LANGUAGE = Language(tsjavascript.language())
JS_LANG = LangSpec(
    language=_JS_LANGUAGE,
    function_types=frozenset({"function_declaration"}),
    class_types=frozenset({"class_declaration"}),
    wrapper_types=frozenset({"export_statement"}),
    method_types=frozenset({"method_definition"}),
    field_fn_types=frozenset({"field_definition"}),
    top_level_assign_types=frozenset({"lexical_declaration", "variable_declaration"}),
)

TS_LANG = LangSpec(
    language=Language(tstypescript.language_typescript()),
    function_types=JS_LANG.function_types,
    class_types=JS_LANG.class_types,
    wrapper_types=JS_LANG.wrapper_types,
    method_types=JS_LANG.method_types,
    field_fn_types=frozenset({"public_field_definition", "field_definition"}),
    top_level_assign_types=JS_LANG.top_level_assign_types,
)

TSX_LANG = LangSpec(
    language=Language(tstypescript.language_tsx()),
    function_types=TS_LANG.function_types,
    class_types=TS_LANG.class_types,
    wrapper_types=TS_LANG.wrapper_types,
    method_types=TS_LANG.method_types,
    field_fn_types=TS_LANG.field_fn_types,
    top_level_assign_types=TS_LANG.top_level_assign_types,
)

# Rust has no OOP-style classes: methods live in a separate `impl` block, not inside the
# `struct`/`enum` they belong to, and traits/modules can also directly hold functions. All
# of these are treated as "class-like" containers here (process_class() already tolerates a
# body with no method-shaped children, which is exactly a plain struct/enum) so that `impl`
# blocks - and `mod { ... }` inline test/sub modules, which mix arbitrary item kinds - get
# their nested functions chunked instead of the whole container becoming one opaque blob.
RUST_LANG = LangSpec(
    language=Language(tsrust.language()),
    function_types=frozenset({"function_item"}),
    class_types=frozenset({"struct_item", "enum_item", "union_item", "trait_item", "impl_item", "mod_item"}),
    wrapper_types=frozenset(),
    method_types=frozenset({"function_item", "function_signature_item"}),
    annotation_types=frozenset({"line_comment", "block_comment", "attribute_item"}),
)

# C has no OOP concept at all - struct/union/enum are plain data containers with no nested
# methods (method_types/field_fn_types stay empty). `type_definition` is a wrapper_type
# purely for `typedef struct/enum/union {...} Name;`: the aggregate itself is anonymous (no
# `name` field), and the real name lives one level up on the typedef - _unwrap() already
# finds the inner struct/enum/union via class_types membership, _node_name() below is
# extended to prefer the typedef's own name over the (missing) inner one.
C_LANG = LangSpec(
    language=Language(tsc.language()),
    function_types=frozenset({"function_definition"}),
    class_types=frozenset({"struct_specifier", "union_specifier", "enum_specifier"}),
    wrapper_types=frozenset({"type_definition"}),
    method_types=frozenset(),
)

EXTENSION_LANGS: dict[str, LangSpec] = {
    ".py": PY_LANG,
    ".js": JS_LANG,
    ".jsx": JS_LANG,
    ".ts": TS_LANG,
    ".tsx": TSX_LANG,
    ".rs": RUST_LANG,
    ".c": C_LANG,
    # Ambiguous with C++ headers (issue #20) - treated as C for now since tree-sitter-c
    # parses the common subset fine; a C++-flavored .h will need issue #20 to sniff content
    # and dispatch to a C++ LangSpec instead of guessing from the extension alone.
    ".h": C_LANG,
}

# C's preprocessor conditional blocks (#ifdef/#if/#elif/#else) and `extern "C" { ... }`
# linkage blocks routinely wrap an entire header's real content - the #ifndef/#define/#endif
# include guard is near-universal - so if these aren't looked through, virtually no .h file
# would produce any function/struct/enum chunks at all; everything would be hidden inside one
# opaque top-level node. These node types are "transparent": their content children get
# spliced into the enclosing sibling sequence (top-level, or a struct/union body) as if the
# wrapper wasn't there, recursively (an #ifdef's #else branch is itself transparent, etc).
TRANSPARENT_CONTAINER_TYPES = frozenset({
    "preproc_ifdef", "preproc_if", "preproc_elif", "preproc_else", "linkage_specification",
})
# Per-container field name(s) holding the condition/tag/type part rather than real content -
# these must NOT be spliced in alongside the actual body (an `identifier` macro name or a
# `#if` condition expression can never legitimately be a top-level statement on its own).
_TRANSPARENT_CONTAINER_SKIP_FIELDS: dict[str, frozenset[str]] = {
    "preproc_ifdef": frozenset({"name"}),
    "preproc_if": frozenset({"condition"}),
    "preproc_elif": frozenset({"condition"}),
}
# linkage_specification's real content (`extern "C" { ... }`) lives one level deeper, inside
# its `body` (declaration_list) field - descending into its own direct children would just
# yield that declaration_list as a single node instead of looking inside it.
_TRANSPARENT_CONTAINER_DESCEND_FIELD: dict[str, str] = {
    "linkage_specification": "body",
}


def _flatten_transparent(nodes: Sequence[Node]) -> list[Node]:
    result: list[Node] = []
    for node in nodes:
        if node.type not in TRANSPARENT_CONTAINER_TYPES:
            result.append(node)
            continue
        descend_field = _TRANSPARENT_CONTAINER_DESCEND_FIELD.get(node.type)
        if descend_field is not None:
            body = node.child_by_field_name(descend_field)
            children = list(body.named_children) if body is not None else []
        else:
            skip_fields = _TRANSPARENT_CONTAINER_SKIP_FIELDS.get(node.type, frozenset())
            children = [
                child for i, child in enumerate(node.children)
                if child.is_named and node.field_name_for_child(i) not in skip_fields
            ]
        result.extend(_flatten_transparent(children))
    return result


@dataclass
class Chunk:
    file_path: Path
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    node_type: str  # "function" | "class" | "method" | "block"
    name: str
    text: str
    # which chunking *strategy* produced this (as opposed to node_type's structural kind).
    # "method" nodes count as "function" here since they're not a distinct strategy, just a
    # nested case of the same AST pass.
    chunk_source_kind: str = "function"  # "function" | "class" | "top_level_assignment_group" | "json_key" | "raw_file_fallback"


def _unwrap(node: Node, spec: LangSpec) -> Node:
    if node.type not in spec.wrapper_types:
        return node
    for child in node.children:
        if child.type in spec.function_types or child.type in spec.class_types \
                or child.type in spec.top_level_assign_types:
            return child
    return node


def _c_declarator_leaf_name(node: Node | None, source: bytes) -> str | None:
    """Descends a C declarator chain (`function_declarator`/`pointer_declarator`/
    `array_declarator` -> ... -> `identifier`) to the leaf name - shared by function/struct
    naming (below) and top-level declaration/typedef/macro naming (`_first_bound_name`),
    which all bottom out at the same declarator shape."""
    while node is not None:
        if node.type in ("identifier", "type_identifier"):
            return source[node.start_byte:node.end_byte].decode("utf-8")
        node = node.child_by_field_name("declarator")
    return None


def _decl_name(node: Node, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name") or node.child_by_field_name("property")
    if name_node is not None:
        return source[name_node.start_byte:name_node.end_byte].decode("utf-8")
    # C `function_definition` has no "name" field of its own - the identifier is nested
    # inside its "declarator" field (`function_declarator`, possibly wrapped in a
    # `pointer_declarator` for a pointer-returning function like `int *foo(void)`).
    declarator_name = _c_declarator_leaf_name(node.child_by_field_name("declarator"), source)
    if declarator_name is not None:
        return declarator_name
    # Rust `impl Foo { }` / `impl Trait for Foo { }` has no "name" field at all - fall back
    # to the "type" (+ "trait") field so impl blocks aren't all just "<anonymous>".
    type_node = node.child_by_field_name("type")
    if type_node is None:
        return None
    type_text = source[type_node.start_byte:type_node.end_byte].decode("utf-8")
    trait_node = node.child_by_field_name("trait")
    if trait_node is None:
        return type_text
    trait_text = source[trait_node.start_byte:trait_node.end_byte].decode("utf-8")
    return f"{trait_text} for {type_text}"


def _node_name(node: Node, spec: LangSpec, source: bytes) -> str:
    effective = _unwrap(node, spec)
    if effective is not node:
        # C `typedef struct/enum/union {...} Name;` - the aggregate _unwrap() found is
        # anonymous (no `name` field of its own), but the typedef wrapping it carries the
        # real name in its own `declarator` field. Prefer that before falling back to
        # whatever (missing) name the unwrapped node itself would report.
        declarator = node.child_by_field_name("declarator")
        if declarator is not None and declarator.type in ("type_identifier", "identifier"):
            return source[declarator.start_byte:declarator.end_byte].decode("utf-8")
    return _decl_name(effective, source) or "<anonymous>"


def _single_function_assignment(node: Node, source: bytes) -> tuple[str, Node] | None:
    """`const X = (...) => {}` / `const X = function() {}` - exactly one declarator, function-valued."""
    declarators = [c for c in node.named_children if c.type == "variable_declarator"]
    if len(declarators) != 1:
        return None
    name = _decl_name(declarators[0], source)
    value_node = declarators[0].child_by_field_name("value")
    if name is None or value_node is None or value_node.type not in FUNCTION_VALUE_TYPES:
        return None
    return name, value_node


def _first_bound_name(node: Node, source: bytes) -> str | None:
    """Best-effort name for a data block, from its first statement's assigned identifier
    (`NAME = ...` / `const NAME = ...`) - falls back to a generic label if not recognized."""
    target = node
    if node.type == "expression_statement" and node.named_child_count:
        target = node.named_children[0]

    if target.type in ("assignment", "augmented_assignment", "assignment_expression", "augmented_assignment_expression"):
        left = target.child_by_field_name("left")
        if left is not None and left.type == "identifier":
            return source[left.start_byte:left.end_byte].decode("utf-8")
    elif target.type in ("lexical_declaration", "variable_declaration"):
        declarators = [c for c in target.named_children if c.type == "variable_declarator"]
        if declarators:
            name_node = declarators[0].child_by_field_name("name")
            if name_node is not None and name_node.type == "identifier":
                return source[name_node.start_byte:name_node.end_byte].decode("utf-8")
    elif target.type in ("const_item", "static_item"):
        name_node = target.child_by_field_name("name")
        if name_node is not None:
            return source[name_node.start_byte:name_node.end_byte].decode("utf-8")
    elif target.type in ("declaration", "type_definition", "preproc_def"):
        return _c_declarator_name(target, source)
    return None


def _c_declarator_name(node: Node, source: bytes) -> str | None:
    """C `declaration`/`type_definition`/`preproc_def` - `#define NAME ...` has its own
    `name` field directly; the others need the shared declarator-chain descent."""
    declarator = node.child_by_field_name("declarator") or node.child_by_field_name("name")
    return _c_declarator_leaf_name(declarator, source)


def _same_source_line(prev_node: Node, next_start_byte: int, source: bytes) -> bool:
    """Whether prev_node's own content and a following byte offset fall on the same source
    line - a "no newline in between" check, but first trimming a trailing newline that's
    part of prev_node's OWN span: tree-sitter-rust's `line_comment` swallows it, so its
    end_byte already sits at the start of the *next* line (unlike e.g. Python's `comment`
    node) - without this trim, the gap itself would be empty/newline-free and two directly
    adjacent lines would be misdetected as "the same line"."""
    end = prev_node.end_byte
    if end > prev_node.start_byte and source[end - 1:end] == b"\n":
        end -= 1
    return b"\n" not in source[end:next_start_byte]


def _leading_comments(siblings: list[Node], index: int, annotation_types: frozenset[str], source: bytes) -> list[Node]:
    """Comment (and, for Rust, `#[attr]`) nodes are separate siblings in tree-sitter, not
    part of the following def's byte range - without this, a docblock right above a
    function/method/class would either be dropped entirely or (worse, inside a class body)
    end up attached to the class header instead of the specific method it was documenting."""
    comments = []
    i = index - 1
    while i >= 0 and siblings[i].type in annotation_types:
        # an annotation on the same line as what precedes IT is a trailing comment of that
        # earlier statement, not a leading one for the node at `index` - without this check
        # it would get attributed to both (once as someone's trailing, once here)
        if i > 0 and _same_source_line(siblings[i - 1], siblings[i].start_byte, source):
            break
        comments.append(siblings[i])
        i -= 1
    comments.reverse()
    return comments


def _same_line_trailing_comment(siblings: list[Node], index: int, annotation_types: frozenset[str], source: bytes) -> Node | None:
    """`X = 1  # comment` - a same-line trailing comment is often the only source of
    meaning for a short constant name, and would otherwise be silently dropped (it isn't
    part of the statement node's own byte range, and _leading_comments only looks backward)."""
    if index + 1 >= len(siblings):
        return None
    nxt = siblings[index + 1]
    if nxt.type in annotation_types and _same_source_line(siblings[index], nxt.start_byte, source):
        return nxt
    return None


def chunk_file(path: Path) -> list[Chunk]:
    spec = EXTENSION_LANGS.get(path.suffix.lower())
    if spec is None:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    source = path.read_bytes()
    parser = Parser(spec.language)
    tree = parser.parse(source)

    def make_chunk(node: Node, node_type: str, name: str, comments: Sequence[Node] = ()) -> Chunk:
        if comments:
            start_line = comments[0].start_point[0] + 1
            comment_text = "\n".join(source[c.start_byte:c.end_byte].decode("utf-8") for c in comments)
            text = comment_text + "\n" + source[node.start_byte:node.end_byte].decode("utf-8")
        else:
            start_line = node.start_point[0] + 1
            text = source[node.start_byte:node.end_byte].decode("utf-8")
        return Chunk(
            file_path=path,
            start_line=start_line,
            end_line=node.end_point[0] + 1,
            node_type=node_type,
            name=name,
            text=text,
            chunk_source_kind="class" if node_type == "class" else "function",
        )

    def field_function_name(node: Node) -> str | None:
        """field_definition/public_field_definition `name = (...) => {}` -> bound method-like field."""
        value_node = node.child_by_field_name("value")
        if value_node is None or value_node.type not in FUNCTION_VALUE_TYPES:
            return None
        return _decl_name(node, source)

    def process_class(node: Node, name_prefix: str, leading_comments: Sequence[Node] = ()) -> list[Chunk]:
        effective = _unwrap(node, spec)
        class_name = _node_name(node, spec, source)
        full_name = f"{name_prefix}.{class_name}" if name_prefix else class_name

        body = effective.child_by_field_name("body")
        if body is None:
            # A class with no body block is a grammar edge case / malformed input - skip just
            # this class rather than raising, so one bad class doesn't lose every other
            # function/class chunk in the same file.
            return []
        body_children = _flatten_transparent(body.named_children)

        method_entries: list[tuple[Node, str, list[Node]]] = []
        nested_class_chunks: list[Chunk] = []
        consumed_ids: set[int] = set()

        for index, child in enumerate(body_children):
            child_effective = _unwrap(child, spec)  # unwrap decorators (@staticmethod etc) before matching
            comments = _leading_comments(body_children, index, spec.annotation_types, source)

            if child_effective.type in spec.method_types:
                name = _decl_name(child_effective, source) or "<anonymous>"
                method_entries.append((child, name, comments))
            elif child.type in spec.field_fn_types:
                fn_name = field_function_name(child)
                if fn_name is None:
                    continue
                method_entries.append((child, fn_name, comments))
            elif child_effective.type in spec.class_types:
                nested_class_chunks.extend(process_class(child, name_prefix=full_name, leading_comments=comments))
            else:
                continue

            consumed_ids.add(id(child))
            consumed_ids.update(id(c) for c in comments)

        field_like = [c for c in body_children if id(c) not in consumed_ids]

        header_parts = []
        if leading_comments:
            header_parts.append("\n".join(source[c.start_byte:c.end_byte].decode("utf-8") for c in leading_comments))
        header_parts.append(source[node.start_byte:body.start_byte].decode("utf-8").rstrip())
        header_parts.extend(source[c.start_byte:c.end_byte].decode("utf-8") for c in field_like)

        start_line = leading_comments[0].start_point[0] + 1 if leading_comments else node.start_point[0] + 1
        chunks = [Chunk(
            file_path=path,
            start_line=start_line,
            end_line=node.end_point[0] + 1,
            node_type="class",
            name=full_name,
            text="\n".join(header_parts).rstrip() + "\n",
            chunk_source_kind="class",
        )]

        for child, method_name, comments in method_entries:
            chunks.append(make_chunk(child, "method", f"{full_name}.{method_name}", comments))
        chunks.extend(nested_class_chunks)

        return chunks

    chunks: list[Chunk] = []
    top_children = _flatten_transparent(tree.root_node.named_children)
    # (node, its own leading comments, its own same-line trailing comment or None)
    block_members: list[tuple[Node, list[Node], Node | None]] = []
    consumed_trailing_ids: set[int] = set()

    def member_span(entry: tuple[Node, list[Node], Node | None]) -> tuple[int, int]:
        member_node, leading, trailing = entry
        start = (leading[0] if leading else member_node).start_point[0]
        end = (trailing if trailing else member_node).end_point[0]
        return start, end

    def flush_block() -> None:
        if not block_members:
            return
        first_start, _ = member_span(block_members[0])
        _, last_end = member_span(block_members[-1])

        text_parts = []
        for member_node, leading, trailing in block_members:
            if leading:
                text_parts.append(
                    "\n".join(source[c.start_byte:c.end_byte].decode("utf-8") for c in leading)
                )
            member_text = source[member_node.start_byte:member_node.end_byte].decode("utf-8")
            if trailing is not None:
                member_text += "  " + source[trailing.start_byte:trailing.end_byte].decode("utf-8")
            text_parts.append(member_text)

        chunks.append(Chunk(
            file_path=path,
            start_line=first_start + 1,
            end_line=last_end + 1,
            node_type="block",
            name=_first_bound_name(block_members[0][0], source) or f"{path.stem}#{len(chunks) + 1}",
            text="\n".join(text_parts),
            chunk_source_kind="top_level_assignment_group",
        ))
        block_members.clear()

    def append_to_block(member_node: Node, leading: list[Node], index: int) -> None:
        trailing = _same_line_trailing_comment(top_children, index, spec.annotation_types, source)
        if trailing is not None:
            consumed_trailing_ids.add(id(trailing))
        if block_members:
            _, prev_end = member_span(block_members[-1])
            gap_start = (leading[0] if leading else member_node).start_point[0]
            if gap_start - prev_end >= 2:  # a blank line between groups starts a new chunk
                flush_block()
        block_members.append((member_node, leading, trailing))

    for index, node in enumerate(top_children):
        if id(node) in consumed_trailing_ids:
            continue  # already attached as another member's trailing comment

        effective = _unwrap(node, spec)
        comments = _leading_comments(top_children, index, spec.annotation_types, source)

        if effective.type in spec.function_types:
            flush_block()
            chunks.append(make_chunk(node, "function", _node_name(node, spec, source), comments))
        elif effective.type in spec.class_types:
            flush_block()
            chunks.extend(process_class(node, name_prefix="", leading_comments=comments))
        elif effective.type in spec.top_level_assign_types:
            assignment = _single_function_assignment(effective, source)
            if assignment is not None:
                flush_block()
                name, _value_node = assignment
                chunks.append(make_chunk(node, "function", name, comments))
            else:
                append_to_block(node, comments, index)
        elif node.type == "expression_statement" and node.named_child_count == 0:
            # A bare `;` (null statement, zero named children) - not a deliberate statement
            # in practice, but an ERROR-recovery artifact: tree-sitter-c can't parse a custom
            # macro used as a GCC attribute annotation (e.g. `PRINTF_LIKE(2, 3)` wrapping
            # `__attribute__((format(printf,...)))`), so it splits the whole declaration into
            # an ERROR node (left un-chunked below, correctly) plus this leftover semicolon.
            # Carries no information regardless of language - skip rather than sweep it.
            continue
        elif node.type in IGNORED_TOP_LEVEL_TYPES:
            continue
        elif node.type in SWEEPABLE_TOP_LEVEL_TYPES:
            append_to_block(node, comments, index)
        # else: unhandled top-level statement kind (if/try/assert/...) - left un-chunked

    flush_block()
    return chunks


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    target = Path(sys.argv[1])
    for chunk in chunk_file(target):
        print(f"[{chunk.node_type}] {chunk.name}  ({chunk.file_path}:{chunk.start_line}-{chunk.end_line})")
        print("-" * 60)
        print(chunk.text)
        print("=" * 60)


if __name__ == "__main__":
    main()
