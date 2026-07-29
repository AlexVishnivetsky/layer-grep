from __future__ import annotations

import pytest

from chunker import chunk_file


def test_python_function(tmp_path, write_file):
    p = write_file(tmp_path, "mod.py", "def foo(x):\n    return x + 1\n")
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "function"
    assert chunks[0].name == "foo"
    assert chunks[0].chunk_source_kind == "function"
    assert "return x + 1" in chunks[0].text


def test_python_leading_comment_attached(tmp_path, write_file):
    src = (
        "# does the thing\n"
        "def foo():\n"
        "    pass\n"
    )
    p = write_file(tmp_path, "mod.py", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1  # comment line, not the `def` line
    assert "# does the thing" in chunks[0].text


def test_python_class_with_methods(tmp_path, write_file):
    src = (
        "class Foo:\n"
        "    def bar(self):\n"
        "        return 1\n"
        "\n"
        "    def baz(self):\n"
        "        return 2\n"
    )
    p = write_file(tmp_path, "mod.py", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert set(by_name) == {"Foo", "Foo.bar", "Foo.baz"}
    assert by_name["Foo"].node_type == "class"
    assert by_name["Foo.bar"].node_type == "method"
    assert by_name["Foo.baz"].node_type == "method"


def test_python_nested_class(tmp_path, write_file):
    src = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def m(self):\n"
        "            pass\n"
    )
    p = write_file(tmp_path, "mod.py", src)
    chunks = chunk_file(p)
    names = {c.name for c in chunks}
    assert "Outer" in names
    assert "Outer.Inner" in names
    assert "Outer.Inner.m" in names


def test_python_decorated_function_and_method(tmp_path, write_file):
    src = (
        "import functools\n"
        "\n"
        "@functools.cache\n"
        "def foo():\n"
        "    pass\n"
        "\n"
        "class Bar:\n"
        "    @staticmethod\n"
        "    def baz():\n"
        "        pass\n"
    )
    p = write_file(tmp_path, "mod.py", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert by_name["foo"].node_type == "function"
    assert "@functools.cache" in by_name["foo"].text
    assert by_name["Bar.baz"].node_type == "method"
    assert "@staticmethod" in by_name["Bar.baz"].text


def test_python_class_without_body_does_not_abort_file(tmp_path, write_file):
    # A malformed/edge-case class - process_class's own None-body guard is what's under
    # test here, so this just needs chunk_file to not raise for a file with an otherwise
    # normal sibling function - the actual `body is None` branch is exercised directly in
    # test_chunker_internals.py-style unit tests if tree-sitter ever produces one; on
    # well-formed source tree-sitter never omits the body, so this mainly guards against a
    # regression making chunk_file intolerant of *any* other unusual class shape.
    src = "class Foo:\n    pass\n\n\ndef bar():\n    return 1\n"
    p = write_file(tmp_path, "mod.py", src)
    chunks = chunk_file(p)
    names = {c.name for c in chunks}
    assert "Foo" in names
    assert "bar" in names


def test_python_top_level_assignment_block(tmp_path, write_file):
    src = (
        "FOO = 'a'\n"
        "BAR = 'b'\n"
        "\n"
        "BAZ = 'c'\n"
    )
    p = write_file(tmp_path, "mod.py", src)
    chunks = chunk_file(p)
    assert len(chunks) == 2  # blank line splits FOO/BAR from BAZ
    assert chunks[0].node_type == "block"
    assert chunks[0].name == "FOO"
    assert "FOO = 'a'" in chunks[0].text and "BAR = 'b'" in chunks[0].text
    assert chunks[1].name == "BAZ"


def test_python_trailing_comment_on_assignment(tmp_path, write_file):
    src = "TIMEOUT = 30  # seconds\n"
    p = write_file(tmp_path, "mod.py", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert "# seconds" in chunks[0].text


def test_python_with_statement_swept_as_block(tmp_path, write_file):
    src = (
        "with router('/task_statuses'):\n"
        "    router('GET', handler=list_statuses, name='list_task_statuses')\n"
    )
    p = write_file(tmp_path, "mod.py", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "block"
    assert "list_task_statuses" in chunks[0].text


def test_python_import_statements_ignored(tmp_path, write_file):
    src = (
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )
    p = write_file(tmp_path, "mod.py", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].name == "foo"


def test_js_arrow_function_assignment(tmp_path, write_file):
    p = write_file(tmp_path, "mod.js", "const add = (a, b) => {\n  return a + b;\n};\n")
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].name == "add"
    assert chunks[0].node_type == "function"


def test_js_class_field_arrow_method(tmp_path, write_file):
    src = (
        "class Widget {\n"
        "  onClick = (event) => {\n"
        "    console.log(event);\n"
        "  }\n"
        "}\n"
    )
    p = write_file(tmp_path, "mod.js", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "Widget" in by_name
    assert "Widget.onClick" in by_name
    assert by_name["Widget.onClick"].node_type == "method"


def test_tsx_supported(tmp_path, write_file):
    src = "const App = () => {\n  return null;\n};\n"
    p = write_file(tmp_path, "app.tsx", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].name == "App"


def test_rust_function(tmp_path, write_file):
    p = write_file(tmp_path, "mod.rs", "fn foo(x: i32) -> i32 {\n    x + 1\n}\n")
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "function"
    assert chunks[0].name == "foo"


def test_rust_doc_comment_and_attribute_attached(tmp_path, write_file):
    src = (
        "/// Does the thing.\n"
        "#[allow(dead_code)]\n"
        "fn foo() {}\n"
    )
    p = write_file(tmp_path, "mod.rs", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1  # doc comment line, not the `fn` line
    assert "/// Does the thing." in chunks[0].text
    assert "#[allow(dead_code)]" in chunks[0].text


def test_rust_struct_with_fields_and_separate_impl_block(tmp_path, write_file):
    src = (
        "struct Point {\n"
        "    x: f32,\n"
        "    y: f32,\n"
        "}\n"
        "\n"
        "impl Point {\n"
        "    fn magnitude(&self) -> f32 {\n"
        "        (self.x * self.x + self.y * self.y).sqrt()\n"
        "    }\n"
        "}\n"
    )
    p = write_file(tmp_path, "mod.rs", src)
    chunks = chunk_file(p)
    by_name: dict[str, list] = {}
    for c in chunks:
        by_name.setdefault(c.name, []).append(c)
    assert "Point" in by_name  # struct header chunk and impl-block chunk share this name
    assert len(by_name["Point"]) == 2
    assert "x: f32" in by_name["Point"][0].text
    assert "Point.magnitude" in by_name
    assert by_name["Point.magnitude"][0].node_type == "method"


def test_rust_trait_impl_naming(tmp_path, write_file):
    src = (
        "struct Filter;\n"
        "\n"
        "impl Default for Filter {\n"
        "    fn default() -> Self {\n"
        "        Filter\n"
        "    }\n"
        "}\n"
    )
    p = write_file(tmp_path, "mod.rs", src)
    chunks = chunk_file(p)
    names = {c.name for c in chunks}
    assert "Default for Filter" in names  # impl blocks have no `name` field - falls back to type(+trait)
    assert "Default for Filter.default" in names


def test_rust_nested_test_module(tmp_path, write_file):
    src = (
        "fn foo() -> u32 {\n"
        "    1\n"
        "}\n"
        "\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    use super::*;\n"
        "\n"
        "    #[test]\n"
        "    fn test_foo() {\n"
        "        assert_eq!(foo(), 1);\n"
        "    }\n"
        "}\n"
    )
    p = write_file(tmp_path, "mod.rs", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "foo" in by_name
    assert "tests" in by_name
    assert "#[cfg(test)]" in by_name["tests"].text  # leading attribute attached to the mod
    assert "tests.test_foo" in by_name
    assert by_name["tests.test_foo"].node_type == "method"
    assert "#[test]" in by_name["tests.test_foo"].text


def test_rust_top_level_const_swept_as_block(tmp_path, write_file):
    src = "const MAX_ITEMS: usize = 4;\nconst MIN_ITEMS: usize = 1;\n"
    p = write_file(tmp_path, "mod.rs", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "block"
    assert chunks[0].name == "MAX_ITEMS"
    assert chunks[0].chunk_source_kind == "top_level_assignment_group"


def test_rust_use_declaration_ignored(tmp_path, write_file):
    src = "use std::sync::Arc;\n\nfn foo() {}\n"
    p = write_file(tmp_path, "mod.rs", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].name == "foo"


def test_rust_bodyless_mod_declaration_produces_no_chunk(tmp_path, write_file):
    p = write_file(tmp_path, "mod.rs", "pub mod utils;\n")
    chunks = chunk_file(p)
    assert chunks == []


def test_c_function(tmp_path, write_file):
    p = write_file(tmp_path, "mod.c", "int add(int a, int b) {\n    return a + b;\n}\n")
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "function"
    assert chunks[0].name == "add"


def test_c_pointer_returning_function_naming(tmp_path, write_file):
    """`int *foo(void)` - the identifier is nested inside a pointer_declarator wrapping the
    function_declarator, not directly on it."""
    p = write_file(tmp_path, "mod.c", "int *make_thing(void) {\n    return 0;\n}\n")
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].name == "make_thing"


def test_c_doc_comment_attached(tmp_path, write_file):
    src = (
        "/// Adds two numbers.\n"
        "// implementation note\n"
        "int add(int a, int b) {\n"
        "    return a + b;\n"
        "}\n"
    )
    p = write_file(tmp_path, "mod.c", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1  # doc comment line, not the `int` line
    assert "/// Adds two numbers." in chunks[0].text
    assert "// implementation note" in chunks[0].text


def test_c_struct_fields(tmp_path, write_file):
    src = "struct Point {\n    int x;\n    int y;\n};\n"
    p = write_file(tmp_path, "mod.c", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "class"
    assert chunks[0].name == "Point"
    assert "int x;" in chunks[0].text


def test_c_anonymous_typedef_struct_naming(tmp_path, write_file):
    """`typedef struct { ... } Name;` - the struct itself has no `name` field (anonymous);
    the real name lives on the typedef's own `declarator` field."""
    src = "typedef struct {\n    int width;\n    int height;\n} Rectangle;\n"
    p = write_file(tmp_path, "mod.c", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "class"
    assert chunks[0].name == "Rectangle"
    assert "int width;" in chunks[0].text


def test_c_enum_and_union(tmp_path, write_file):
    src = (
        "enum Color {\n    RED,\n    GREEN,\n    BLUE\n};\n"
        "\n"
        "union Value {\n    int i;\n    float f;\n};\n"
    )
    p = write_file(tmp_path, "mod.c", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "Color" in by_name
    assert "RED" in by_name["Color"].text
    assert "Value" in by_name
    assert "int i;" in by_name["Value"].text


def test_c_header_guard_does_not_hide_content(tmp_path, write_file):
    """The near-universal `#ifndef X / #define X / ... / #endif` include guard wraps the
    ENTIRE file in one preproc_ifdef node - without flattening it, every declaration inside
    a guarded .h file would be invisible to chunking."""
    src = (
        "#ifndef SAMPLE_H\n"
        "#define SAMPLE_H\n"
        "\n"
        "int add(int a, int b) {\n"
        "    return a + b;\n"
        "}\n"
        "\n"
        "#endif\n"
    )
    p = write_file(tmp_path, "mod.h", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "add" in by_name
    assert by_name["add"].node_type == "function"


def test_c_ifdef_else_both_branches_chunked(tmp_path, write_file):
    src = (
        "#ifdef DEBUG\n"
        "int mode(void) { return 1; }\n"
        "#else\n"
        "int mode_release(void) { return 2; }\n"
        "#endif\n"
    )
    p = write_file(tmp_path, "mod.c", src)
    chunks = chunk_file(p)
    names = {c.name for c in chunks}
    assert "mode" in names
    assert "mode_release" in names


def test_c_extern_c_linkage_block_flattened(tmp_path, write_file):
    src = (
        "#ifdef __cplusplus\n"
        'extern "C" {\n'
        "#endif\n"
        "\n"
        "int foo(void) {\n"
        "    return 1;\n"
        "}\n"
        "\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif\n"
    )
    p = write_file(tmp_path, "mod.h", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "foo" in by_name
    assert by_name["foo"].node_type == "function"


def test_c_include_ignored(tmp_path, write_file):
    src = '#include <stdio.h>\n#include "other.h"\n\nint foo(void) { return 1; }\n'
    p = write_file(tmp_path, "mod.c", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].name == "foo"


def test_c_define_swept_as_block(tmp_path, write_file):
    src = "#define MAX_ITEMS 10\n#define MIN_ITEMS 1\n"
    p = write_file(tmp_path, "mod.c", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "block"
    assert chunks[0].name == "MAX_ITEMS"


def test_c_function_prototype_swept_as_block(tmp_path, write_file):
    """A prototype has no body to chunk as a function - it's data about the API surface,
    same spirit as a top-level const."""
    src = "int add(int a, int b);\nvoid free_thing(struct Thing *t);\n"
    p = write_file(tmp_path, "mod.h", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "block"
    assert chunks[0].name == "add"
    assert "free_thing" in chunks[0].text


def test_c_gcc_attribute_macro_leftover_semicolon_not_swept(tmp_path, write_file):
    """tree-sitter-c can't parse a custom macro used as a GCC attribute annotation (only the
    literal __attribute__/__declspec spelling) - `PRINTF_LIKE(2, 3)` after a declaration
    splits into an ERROR node (correctly left un-chunked) plus a leftover bare `;`
    (expression_statement with zero named children) that must not be swept as a data block."""
    src = (
        "void report(const char *fmt, ...)\n"
        "    PRINTF_LIKE(1, 2);\n"
        "\n"
        "int real_func(void) {\n"
        "    return 1;\n"
        "}\n"
    )
    # .c specifically (not .h) - since issue #20, .h dispatches to the C++ grammar, which
    # recovers from this same pattern differently (a real declaration + a real macro-call
    # expression_statement, not a bare-semicolon artifact) - see the C++-specific version of
    # this scenario below.
    p = write_file(tmp_path, "mod.c", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].name == "real_func"
    assert all(";" != c.text.strip() for c in chunks)


def test_c_global_declaration_swept_as_block(tmp_path, write_file):
    src = "const int GLOBAL_CONST = 42;\nstatic int counter = 0;\n"
    p = write_file(tmp_path, "mod.c", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "block"
    assert chunks[0].name == "GLOBAL_CONST"


def test_cpp_free_function(tmp_path, write_file):
    p = write_file(tmp_path, "mod.cpp", "int add(int a, int b) {\n    return a + b;\n}\n")
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "function"
    assert chunks[0].name == "add"


def test_cpp_class_with_constructor_destructor_and_method(tmp_path, write_file):
    src = (
        "class Point {\n"
        "public:\n"
        "    Point(int x, int y) : x_(x), y_(y) {}\n"
        "    ~Point() {}\n"
        "    int getX() const { return x_; }\n"
        "private:\n"
        "    int x_;\n"
        "    int y_;\n"
        "};\n"
    )
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "Point" in by_name
    assert by_name["Point"].node_type == "class"
    assert "int x_;" in by_name["Point"].text
    assert by_name["Point.Point"].node_type == "method"
    assert by_name["Point.~Point"].node_type == "method"
    assert by_name["Point.getX"].node_type == "method"


def test_cpp_operator_overload_naming(tmp_path, write_file):
    src = (
        "class Point {\n"
        "public:\n"
        "    Point operator+(const Point& other) const { return other; }\n"
        "};\n"
    )
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    names = {c.name for c in chunks}
    assert "Point.operator+" in names


def test_cpp_struct_with_method(tmp_path, write_file):
    src = "struct Rect {\n    int width;\n    int height;\n\n    int area() const { return width * height; }\n};\n"
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "Rect" in by_name
    assert "int width;" in by_name["Rect"].text
    assert by_name["Rect.area"].node_type == "method"


def test_cpp_namespace_nests_free_functions_and_classes(tmp_path, write_file):
    src = (
        "namespace app {\n"
        "\n"
        "int free_function(int a) { return a; }\n"
        "\n"
        "class Widget {\n"
        "public:\n"
        "    void run() {}\n"
        "};\n"
        "\n"
        "}\n"
    )
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    names = {c.name for c in chunks}
    assert "app" in names
    assert "app.free_function" in names
    assert "app.Widget" in names
    assert "app.Widget.run" in names


def test_cpp_nested_namespace(tmp_path, write_file):
    src = "namespace outer {\nnamespace inner {\nvoid nested_func() {}\n}\n}\n"
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    names = {c.name for c in chunks}
    assert "outer" in names
    assert "outer.inner" in names
    assert "outer.inner.nested_func" in names


def test_cpp_anonymous_namespace(tmp_path, write_file):
    src = "namespace {\nint hidden_helper() { return 1; }\n}\n"
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    names = {c.name for c in chunks}
    assert "<anonymous>" in names
    assert "<anonymous>.hidden_helper" in names


def test_cpp_template_function_unwrapped(tmp_path, write_file):
    src = "template <typename T>\nT max_value(T a, T b) {\n    return a > b ? a : b;\n}\n"
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].name == "max_value"
    assert chunks[0].node_type == "function"
    assert "template <typename T>" in chunks[0].text


def test_cpp_template_class_unwrapped(tmp_path, write_file):
    src = "template <typename T>\nclass Box {\npublic:\n    T value;\n    T get() const { return value; }\n};\n"
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "Box" in by_name
    assert "template <typename T>" in by_name["Box"].text
    assert by_name["Box.get"].node_type == "method"


def test_cpp_using_declaration_ignored(tmp_path, write_file):
    src = "using namespace std;\n\nint foo() { return 1; }\n"
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].name == "foo"


def test_cpp_alias_declaration_swept_as_block(tmp_path, write_file):
    src = "using MyInt = int;\nusing MyFloat = float;\n"
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].node_type == "block"
    assert chunks[0].name == "MyInt"


def test_cpp_header_extension_dispatches_to_cpp_grammar(tmp_path, write_file):
    # .h is ambiguous between C/C++ - resolved by routing through the C++ grammar (a superset
    # for the constructs that matter here), so a C++-flavored .h file chunks correctly too
    src = "class Widget {\npublic:\n    void run() {}\n};\n"
    p = write_file(tmp_path, "mod.h", src)
    chunks = chunk_file(p)
    names = {c.name for c in chunks}
    assert "Widget" in names
    assert "Widget.run" in names


def test_cpp_export_macro_before_class_name_still_recognized_as_class(tmp_path, write_file):
    # DLL/module export macros conventionally spelled <NAME>_API (MSVC __declspec wrapper,
    # e.g. Unreal Engine's <MODULE>_API) between `class`/`struct` and the class name aren't
    # valid C++ grammar - without stripping them first, tree-sitter-cpp misparses the whole
    # class as a bogus function_definition (found on a real UE project, issue #20)
    src = (
        "class MYLIB_API Widget : public Base\n"
        "{\n"
        "public:\n"
        "    void run() {}\n"
        "};\n"
    )
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "Widget" in by_name
    assert by_name["Widget"].node_type == "class"
    assert by_name["Widget.run"].node_type == "method"


def test_cpp_export_macro_stripping_preserves_byte_offsets(tmp_path, write_file):
    # the macro token is blanked with equal-length spaces, not removed, so start_line/end_line
    # for everything else in the file stay exactly what a human reading the real source
    # would expect
    src = (
        "// leading comment\n"
        "\n"
        "class MYLIB_API Widget\n"
        "{\n"
        "public:\n"
        "    void run() {}\n"
        "};\n"
    )
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert by_name["Widget.run"].start_line == 6


def test_cpp_export_macro_struct_before_name_still_recognized(tmp_path, write_file):
    src = "struct MYLIB_API Data\n{\n    int value() { return 1; }\n};\n"
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "Data" in by_name
    assert by_name["Data"].node_type == "class"
    assert by_name["Data.value"].node_type == "method"


def test_cpp_out_of_class_method_definition_naming(tmp_path, write_file):
    # `ReturnType ClassName::Method(...) { ... }` - the standard "declare in header, define
    # in .cpp" pattern. The declarator's own child is a qualified_identifier, not a plain
    # identifier - without handling it, _decl_name() falls through to the Rust impl-block
    # fallback and wrongly names the chunk after the return type instead of the method.
    src = "void Widget::run()\n{\n    x = 1;\n}\n"
    p = write_file(tmp_path, "mod.cpp", src)
    chunks = chunk_file(p)
    assert len(chunks) == 1
    assert chunks[0].name == "Widget::run"
    assert chunks[0].node_type == "function"


def test_cpp_extern_c_linkage_block_flattened(tmp_path, write_file):
    src = (
        "#ifdef __cplusplus\n"
        'extern "C" {\n'
        "#endif\n"
        "\n"
        "int foo(void) {\n"
        "    return 1;\n"
        "}\n"
        "\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif\n"
    )
    p = write_file(tmp_path, "mod.hpp", src)
    chunks = chunk_file(p)
    by_name = {c.name: c for c in chunks}
    assert "foo" in by_name
    assert by_name["foo"].node_type == "function"


def test_unsupported_extension_raises(tmp_path, write_file):
    p = write_file(tmp_path, "mod.rb", "def foo; end\n")
    with pytest.raises(ValueError):
        chunk_file(p)


def test_chunker_version_is_int():
    import chunker
    assert isinstance(chunker.CHUNKER_VERSION, int)
