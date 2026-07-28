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


def test_unsupported_extension_raises(tmp_path, write_file):
    p = write_file(tmp_path, "mod.rb", "def foo; end\n")
    with pytest.raises(ValueError):
        chunk_file(p)


def test_chunker_version_is_int():
    import chunker
    assert isinstance(chunker.CHUNKER_VERSION, int)
