from __future__ import annotations

from imports import extract_python_imports


def test_absolute_dotted_import(tmp_path, write_file):
    write_file(tmp_path, "pkg/__init__.py", "")
    write_file(tmp_path, "pkg/helpers.py", "def foo():\n    pass\n")
    src = write_file(tmp_path, "main.py", "import pkg.helpers\n")

    edges = extract_python_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "pkg" / "helpers.py"
    assert edges[0].module == "pkg.helpers"


def test_from_import_names_prefers_submodule_over_init(tmp_path, write_file):
    write_file(tmp_path, "pkg/__init__.py", "")
    write_file(tmp_path, "pkg/helpers.py", "def foo():\n    pass\n")
    src = write_file(tmp_path, "main.py", "from pkg import helpers\n")

    edges = extract_python_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "pkg" / "helpers.py"


def test_from_import_falls_back_to_init(tmp_path, write_file):
    write_file(tmp_path, "pkg/__init__.py", "VALUE = 1\n")
    src = write_file(tmp_path, "main.py", "from pkg import VALUE\n")

    edges = extract_python_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "pkg" / "__init__.py"


def test_relative_import_single_dot(tmp_path, write_file):
    write_file(tmp_path, "pkg/__init__.py", "")
    write_file(tmp_path, "pkg/sibling.py", "def bar():\n    pass\n")
    src = write_file(tmp_path, "pkg/main.py", "from . import sibling\n")

    edges = extract_python_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "pkg" / "sibling.py"
    # no dotted-name remainder after the dot (the imported name "sibling" is a `name`,
    # not part of the module path itself) - module_display is just the dot(s)
    assert edges[0].module == "."


def test_relative_import_double_dot_goes_up_a_package(tmp_path, write_file):
    write_file(tmp_path, "pkg/__init__.py", "")
    write_file(tmp_path, "pkg/sub/__init__.py", "")
    write_file(tmp_path, "pkg/target.py", "def baz():\n    pass\n")
    src = write_file(tmp_path, "pkg/sub/main.py", "from .. import target\n")

    edges = extract_python_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "pkg" / "target.py"


def test_cross_package_import_resolves_across_project_root(tmp_path, write_file):
    # one module importing from a sibling package - both live directly under project_root,
    # neither is an ancestor of the other
    write_file(tmp_path, "engine/__init__.py", "")
    write_file(tmp_path, "engine/core.py", "def run():\n    pass\n")
    src = write_file(tmp_path, "webapp/app.py", "import engine.core\n")

    edges = extract_python_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "engine" / "core.py"


def test_unresolvable_import_yields_no_edge(tmp_path, write_file):
    src = write_file(tmp_path, "main.py", "import totally_missing_package\n")
    edges = extract_python_imports(src, tmp_path)
    assert edges == []


def test_no_imports_yields_no_edges(tmp_path, write_file):
    src = write_file(tmp_path, "main.py", "def foo():\n    pass\n")
    edges = extract_python_imports(src, tmp_path)
    assert edges == []


def test_imports_version_is_int():
    import imports
    assert isinstance(imports.IMPORTS_VERSION, int)
