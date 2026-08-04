from __future__ import annotations

from imports import (
    extract_c_imports,
    extract_imports,
    extract_js_imports,
    extract_python_imports,
    extract_rust_imports,
)


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


def test_type_checking_guarded_import_is_excluded(tmp_path, write_file):
    write_file(tmp_path, "pkg/__init__.py", "")
    write_file(tmp_path, "pkg/other.py", "class Other:\n    pass\n")
    src = write_file(
        tmp_path, "pkg/main.py",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from pkg.other import Other\n",
    )
    edges = extract_python_imports(src, tmp_path)
    assert edges == []


def test_type_checking_attribute_form_guarded_import_is_excluded(tmp_path, write_file):
    write_file(tmp_path, "pkg/__init__.py", "")
    write_file(tmp_path, "pkg/other.py", "class Other:\n    pass\n")
    src = write_file(
        tmp_path, "pkg/main.py",
        "import typing\n"
        "if typing.TYPE_CHECKING:\n"
        "    from pkg.other import Other\n",
    )
    edges = extract_python_imports(src, tmp_path)
    assert edges == []


def test_type_checking_else_branch_is_still_walked(tmp_path, write_file):
    write_file(tmp_path, "pkg/__init__.py", "")
    write_file(tmp_path, "pkg/guarded.py", "class Guarded:\n    pass\n")
    write_file(tmp_path, "pkg/real.py", "class Real:\n    pass\n")
    src = write_file(
        tmp_path, "pkg/main.py",
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from pkg.guarded import Guarded\n"
        "else:\n"
        "    from pkg.real import Real\n",
    )
    edges = extract_python_imports(src, tmp_path)
    assert [e.target for e in edges] == [tmp_path / "pkg" / "real.py"]


def test_imports_versions_are_ints():
    import imports
    assert set(imports.IMPORTS_VERSIONS) == set(imports.IMPORT_VERSION_GROUPS)
    assert all(isinstance(v, int) for v in imports.IMPORTS_VERSIONS.values())


def test_import_graph_extensions_matches_group_union():
    import imports
    expected = frozenset().union(*imports.IMPORT_VERSION_GROUPS.values())
    assert imports.IMPORT_GRAPH_EXTENSIONS == expected


def test_js_default_import_resolves_with_inferred_extension(tmp_path, write_file):
    write_file(tmp_path, "helpers.js", "export function foo() {}\n")
    src = write_file(tmp_path, "main.js", 'import foo from "./helpers";\n')
    edges = extract_js_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.js"
    assert edges[0].module == "./helpers"


def test_js_named_import(tmp_path, write_file):
    write_file(tmp_path, "helpers.js", "export const a = 1;\n")
    src = write_file(tmp_path, "main.js", 'import { a } from "./helpers";\n')
    edges = extract_js_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.js"


def test_js_namespace_import(tmp_path, write_file):
    write_file(tmp_path, "helpers.js", "export const a = 1;\n")
    src = write_file(tmp_path, "main.js", 'import * as ns from "./helpers";\n')
    edges = extract_js_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.js"


def test_js_export_from_and_export_star(tmp_path, write_file):
    write_file(tmp_path, "helpers.js", "export const a = 1;\n")
    src = write_file(tmp_path, "main.js", (
        'export { a } from "./helpers";\n'
        'export * from "./helpers";\n'
    ))
    edges = extract_js_imports(src, tmp_path)
    assert len(edges) == 2
    assert all(e.target == tmp_path / "helpers.js" for e in edges)


def test_js_require_call(tmp_path, write_file):
    write_file(tmp_path, "helpers.js", "module.exports = {};\n")
    src = write_file(tmp_path, "main.js", 'const helpers = require("./helpers");\n')
    edges = extract_js_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.js"


def test_ts_import_prefers_ts_extension_over_js(tmp_path, write_file):
    # both a .ts and a .js file with the same stem exist - .ts wins (source over a
    # possibly-compiled/vendored sibling), per _JS_EXTENSIONS_TO_TRY's order
    write_file(tmp_path, "helpers.js", "export const a = 1;\n")
    write_file(tmp_path, "helpers.ts", "export const a: number = 1;\n")
    src = write_file(tmp_path, "main.ts", 'import { a } from "./helpers";\n')
    edges = extract_js_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.ts"


def test_js_import_resolves_to_directory_index(tmp_path, write_file):
    write_file(tmp_path, "helpers/index.js", "export const a = 1;\n")
    src = write_file(tmp_path, "main.js", 'import { a } from "./helpers";\n')
    edges = extract_js_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers" / "index.js"


def test_js_parent_relative_import(tmp_path, write_file):
    write_file(tmp_path, "helpers.js", "export const a = 1;\n")
    src = write_file(tmp_path, "sub/main.js", 'import { a } from "../helpers";\n')
    edges = extract_js_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.js"


def test_js_bare_specifier_yields_no_edge(tmp_path, write_file):
    # an external package (node_modules) - same treatment as an unresolvable Python import
    src = write_file(tmp_path, "main.js", 'import react from "react";\n')
    edges = extract_js_imports(src, tmp_path)
    assert edges == []


def test_js_no_imports_yields_no_edges(tmp_path, write_file):
    src = write_file(tmp_path, "main.js", "function foo() {}\n")
    edges = extract_js_imports(src, tmp_path)
    assert edges == []


def test_extract_imports_dispatches_by_extension(tmp_path, write_file):
    write_file(tmp_path, "helpers.js", "export const a = 1;\n")
    py_src = write_file(tmp_path, "main.py", "import totally_missing_package\n")
    js_src = write_file(tmp_path, "main.js", 'import { a } from "./helpers";\n')
    rs_src = write_file(tmp_path, "main.rs", "fn main() {}\n")

    assert extract_imports(py_src, tmp_path) == []  # dispatched to the Python extractor
    js_edges = extract_imports(js_src, tmp_path)
    assert len(js_edges) == 1 and js_edges[0].target == tmp_path / "helpers.js"
    assert extract_imports(rs_src, tmp_path) == []  # dispatched to the Rust extractor, no `use` statements to find

    write_file(tmp_path, "helpers.h", "int foo(void);\n")
    c_src = write_file(tmp_path, "main.c", '#include "helpers.h"\n')
    c_edges = extract_imports(c_src, tmp_path)
    assert len(c_edges) == 1 and c_edges[0].target == tmp_path / "helpers.h"


def test_rust_crate_path_resolves_through_mod_tree(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "src/audio/mod.rs", "pub mod engine;\n")
    write_file(tmp_path, "src/audio/engine.rs", "pub fn run() {}\n")
    src = write_file(tmp_path, "src/lib.rs", "pub mod audio;\n\nuse crate::audio::engine::run;\n")

    edges = extract_rust_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "src" / "audio" / "engine.rs"
    assert edges[0].module == "crate::audio::engine::run"


def test_rust_self_resolves_within_current_module(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    src = write_file(tmp_path, "src/lib.rs", (
        "use self::helpers::bar;\n"
        "\n"
        "mod helpers {\n"
        "    pub fn bar() {}\n"
        "}\n"
    ))
    edges = extract_rust_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == src  # inline mod - same file


def test_rust_super_goes_up_one_module(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "src/audio/engine.rs", "pub fn run() {}\n")
    write_file(tmp_path, "src/lib.rs", "pub mod audio;\n")
    write_file(tmp_path, "src/audio/mod.rs", "pub mod engine;\n\nuse super::audio::engine::run;\n")

    edges = extract_rust_imports(tmp_path / "src" / "audio" / "mod.rs", tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "src" / "audio" / "engine.rs"


def test_rust_cross_crate_resolution_via_sibling_package(tmp_path, write_file):
    # rustdesk-style layout: a path-dependency sub-crate under libs/, referenced by its own
    # package name rather than through the current crate's own module tree
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "libs/scrap/Cargo.toml", '[package]\nname = "scrap"\n')
    write_file(tmp_path, "libs/scrap/src/lib.rs", "pub struct Capturer;\n")
    src = write_file(tmp_path, "src/lib.rs", "use scrap::Capturer;\n")

    edges = extract_rust_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "libs" / "scrap" / "src" / "lib.rs"


def test_rust_hyphenated_crate_name_normalized_to_underscore(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "libs/my-helper/Cargo.toml", '[package]\nname = "my-helper"\n')
    write_file(tmp_path, "libs/my-helper/src/lib.rs", "pub fn helper() {}\n")
    src = write_file(tmp_path, "src/lib.rs", "use my_helper::helper;\n")

    edges = extract_rust_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "libs" / "my-helper" / "src" / "lib.rs"


def test_rust_use_list_multiple_items(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "src/audio/mod.rs", "pub mod engine;\npub mod dsp;\n")
    write_file(tmp_path, "src/audio/engine.rs", "pub fn run() {}\n")
    write_file(tmp_path, "src/audio/dsp.rs", "pub fn process() {}\n")
    src = write_file(tmp_path, "src/lib.rs", "pub mod audio;\n\nuse crate::audio::{engine, dsp};\n")

    edges = extract_rust_imports(src, tmp_path)
    targets = {e.target for e in edges}
    assert targets == {tmp_path / "src" / "audio" / "engine.rs", tmp_path / "src" / "audio" / "dsp.rs"}


def test_rust_use_wildcard(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "src/audio/mod.rs", "pub fn run() {}\n")
    src = write_file(tmp_path, "src/lib.rs", "pub mod audio;\n\nuse crate::audio::*;\n")

    edges = extract_rust_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "src" / "audio" / "mod.rs"


def test_rust_use_as_alias_resolves_by_real_name(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "src/audio/mod.rs", "pub fn run() {}\n")
    src = write_file(tmp_path, "src/lib.rs", "pub mod audio;\n\nuse crate::audio as aud;\n")

    edges = extract_rust_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "src" / "audio" / "mod.rs"


def test_rust_external_crate_yields_no_edge(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    src = write_file(tmp_path, "src/lib.rs", "use serde::Serialize;\n")
    edges = extract_rust_imports(src, tmp_path)
    assert edges == []


def test_rust_unreachable_file_not_in_module_tree_yields_no_edge(tmp_path, write_file):
    # a file no `mod` declaration anywhere ever names isn't part of the crate's tree at all -
    # self::/super:: from it can't be resolved (no known position), same as rustc treating it
    # as dead code
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "src/lib.rs", "// note: does NOT declare `mod orphan;`\n")
    src = write_file(tmp_path, "src/orphan.rs", "use self::helpers::bar;\n")
    edges = extract_rust_imports(src, tmp_path)
    assert edges == []


def test_rust_bin_target_gets_own_tree(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", (
        '[package]\n'
        'name = "myapp"\n'
        '\n'
        '[[bin]]\n'
        'name = "naming"\n'
        'path = "src/naming.rs"\n'
    ))
    write_file(tmp_path, "src/helpers.rs", "pub fn helper() {}\n")
    src = write_file(tmp_path, "src/naming.rs", "mod helpers;\n\nuse crate::helpers::helper;\n")

    edges = extract_rust_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "src" / "helpers.rs"


def test_c_quoted_include_resolves_relative_to_including_file(tmp_path, write_file):
    write_file(tmp_path, "helpers.h", "#ifndef HELPERS_H\n#define HELPERS_H\nint foo(void);\n#endif\n")
    src = write_file(tmp_path, "main.c", '#include "helpers.h"\n')
    edges = extract_c_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.h"
    assert edges[0].module == "helpers.h"


def test_c_quoted_include_subdirectory_path(tmp_path, write_file):
    write_file(tmp_path, "include/helpers.h", "int foo(void);\n")
    src = write_file(tmp_path, "src/main.c", '#include "../include/helpers.h"\n')
    edges = extract_c_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "include" / "helpers.h"


def test_c_angle_bracket_include_yields_no_edge(tmp_path, write_file):
    # a system/library header via the compiler's own search path - same treatment as an
    # external Python package or a bare JS specifier
    src = write_file(tmp_path, "main.c", "#include <stdio.h>\n")
    edges = extract_c_imports(src, tmp_path)
    assert edges == []


def test_c_unresolvable_quoted_include_yields_no_edge(tmp_path, write_file):
    # a project-specific -I search path this function has no visibility into (see the open
    # question in issue #21) - left unresolved rather than guessed at
    src = write_file(tmp_path, "main.c", '#include "does_not_exist.h"\n')
    edges = extract_c_imports(src, tmp_path)
    assert edges == []


def test_c_include_resolves_by_unique_basename_across_project(tmp_path, write_file):
    # Unreal Engine's Public/Private module split: a .cpp in Private/ includes its sibling
    # header by bare name, relying on the module's own Public/ dir being a registered
    # compiler include path - not "relative to the including file" at all. Recovered without
    # knowing the actual search path: unique basename anywhere in the project.
    write_file(tmp_path, "Public/Widget.h", "class Widget {};\n")
    src = write_file(tmp_path, "Private/Widget.cpp", '#include "Widget.h"\n')
    edges = extract_c_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "Public" / "Widget.h"


def test_c_include_ambiguous_basename_yields_no_edge(tmp_path, write_file):
    # two unrelated files share a basename (a real, if less common, occurrence - e.g.
    # abseil-cpp has three unrelated config.h files) - picking one would risk a confidently
    # wrong edge, so this must give up rather than guess
    write_file(tmp_path, "moduleA/config.h", "int a;\n")
    write_file(tmp_path, "moduleB/config.h", "int b;\n")
    src = write_file(tmp_path, "main.cpp", '#include "config.h"\n')
    edges = extract_c_imports(src, tmp_path)
    assert edges == []


def test_c_include_relative_resolution_takes_priority_over_basename_fallback(tmp_path, write_file):
    # when the file actually IS relative to the including file, that's used directly - the
    # basename index (which could point elsewhere, or be ambiguous) is never even consulted
    write_file(tmp_path, "helpers.h", "// the real, relative one\n")
    write_file(tmp_path, "other/helpers.h", "// a same-named decoy elsewhere\n")
    src = write_file(tmp_path, "main.c", '#include "helpers.h"\n')
    edges = extract_c_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.h"


def test_c_no_includes_yields_no_edges(tmp_path, write_file):
    src = write_file(tmp_path, "main.c", "int main(void) { return 0; }\n")
    edges = extract_c_imports(src, tmp_path)
    assert edges == []


def test_c_include_inside_header_guard_still_found(tmp_path, write_file):
    # #include nested inside a #ifndef/#endif header guard - the extractor walks the whole
    # tree recursively (unlike chunker.py, it doesn't need the transparent-container
    # flattening mechanism to find nodes buried inside a preproc_ifdef)
    write_file(tmp_path, "helpers.h", "int foo(void);\n")
    src = write_file(tmp_path, "main.h", (
        "#ifndef MAIN_H\n"
        "#define MAIN_H\n"
        '#include "helpers.h"\n'
        "#endif\n"
    ))
    edges = extract_c_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.h"


def test_cpp_quoted_include_resolves_relative_to_including_file(tmp_path, write_file):
    write_file(tmp_path, "helpers.hpp", "int foo();\n")
    src = write_file(tmp_path, "main.cpp", '#include "helpers.hpp"\n')
    edges = extract_c_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.hpp"


def test_cpp_angle_bracket_include_yields_no_edge(tmp_path, write_file):
    # a standard-library header via the compiler's own search path - same treatment as C
    src = write_file(tmp_path, "main.cpp", "#include <string>\n")
    edges = extract_c_imports(src, tmp_path)
    assert edges == []


def test_cpp_quoted_include_subdirectory_path(tmp_path, write_file):
    write_file(tmp_path, "detail/helpers.hpp", "int foo();\n")
    src = write_file(tmp_path, "src/main.cc", '#include "../detail/helpers.hpp"\n')
    edges = extract_c_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "detail" / "helpers.hpp"


def test_extract_imports_dispatches_cpp_extensions(tmp_path, write_file):
    write_file(tmp_path, "helpers.hpp", "int foo();\n")
    src = write_file(tmp_path, "main.cxx", '#include "helpers.hpp"\n')
    edges = extract_imports(src, tmp_path)
    assert len(edges) == 1
    assert edges[0].target == tmp_path / "helpers.hpp"
