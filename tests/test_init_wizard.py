from __future__ import annotations

import json

import init_wizard


def test_suggest_config_detects_dir_convention(tmp_path, write_file):
    write_file(tmp_path, "api/handlers.py", "def foo(): pass\n")
    write_file(tmp_path, "models/user.py", "class User: pass\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    layer_names = {l["name"] for l in draft["layers"]}
    assert "backend/api" in layer_names
    assert "models" in layer_names


def test_suggest_config_detects_django_file_convention(tmp_path, write_file):
    # Django apps put one concern per *file* at the app root (models.py/views.py), not per
    # directory - the exact gap found by cloning django-locallibrary-tutorial (directory-only
    # matching returned an empty "layers": [] despite thoroughly Django-conventional code)
    write_file(tmp_path, "catalog/models.py", "class Book: pass\n")
    write_file(tmp_path, "catalog/views.py", "def index(request): pass\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    by_name = {l["name"]: l for l in draft["layers"]}
    assert "models.py" in by_name["models"]["files"]
    assert "views.py" in by_name["backend/api"]["files"]


def test_suggest_config_detects_fastapi_core_config_file(tmp_path, write_file):
    # "core" alone is deliberately NOT a dir candidate (too ambiguous), but "config.py" as a
    # filename is specific enough - found by cloning full-stack-fastapi-template
    write_file(tmp_path, "backend/app/core/config.py", "class Settings: pass\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    by_name = {l["name"]: l for l in draft["layers"]}
    assert "config.py" in by_name["config"]["files"]


def test_suggest_config_empty_when_no_conventions_match(tmp_path, write_file):
    write_file(tmp_path, "randomly_named_thing.py", "def x(): pass\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    assert draft["layers"] == []


def test_suggest_config_frontend_extension_fallback(tmp_path, write_file):
    write_file(tmp_path, "weirdly_named_dir/App.tsx", "export const App = () => null;\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    frontend = next((l for l in draft["layers"] if l["name"] == "frontend"), None)
    assert frontend is not None
    assert frontend["dirs"] == []  # no dir matched - human must fill this in by hand


def test_suggest_config_translations_dir_found(tmp_path, write_file):
    write_file(tmp_path, "langs/ru.json", '{"a": "b"}')
    draft = init_wizard.suggest_project_config(tmp_path)
    assert draft["translations"]["files"] == ["langs/ru.json"]


def test_suggest_config_translations_excludes_vendored_copies(tmp_path, write_file):
    # the real vendor-leak bug: _SCAN_EXCLUDED_DIR_NAMES filtering was applied when first
    # deciding a directory name like "locales" is a real candidate, but not re-applied in
    # the follow-up rglob over the whole tree - found via synthetic reproduction this session
    write_file(tmp_path, "locales/ru.json", '{"a": "b"}')
    write_file(tmp_path, "node_modules/some-i18n-pkg/locales/en.json", '{"x": "y"}')
    draft = init_wizard.suggest_project_config(tmp_path)
    assert draft["translations"]["files"] == ["locales/ru.json"]


def test_suggest_config_scan_excludes_venv_and_vendor_dirs(tmp_path, write_file):
    write_file(tmp_path, ".venv/Lib/site-packages/somepkg/models.py", "class Fake: pass\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    assert draft["layers"] == []


def test_suggest_config_rust_conventional_dirs(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "src/lib.rs", "pub fn foo() {}\n")
    write_file(tmp_path, "tests/integration.rs", "fn test_foo() {}\n")
    write_file(tmp_path, "examples/demo.rs", "fn main() {}\n")
    write_file(tmp_path, "benches/bench_foo.rs", "fn bench_foo() {}\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    layer_names = {l["name"] for l in draft["layers"]}
    assert {"tests", "examples", "benches"} <= layer_names


def test_suggest_config_rust_commands_convention(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "src-tauri/src/commands.rs", "fn do_thing() {}\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    by_name = {l["name"]: l for l in draft["layers"]}
    assert "commands.rs" in by_name["commands"]["files"]


def test_suggest_config_rust_bin_targets_become_layers(tmp_path, write_file):
    write_file(tmp_path, "Cargo.toml", (
        '[package]\n'
        'name = "myapp"\n'
        '\n'
        '[[bin]]\n'
        'name = "naming"\n'
        'path = "src/naming.rs"\n'
    ))
    write_file(tmp_path, "src/naming.rs", "fn main() {}\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    by_name = {l["name"]: l for l in draft["layers"]}
    assert "naming" in by_name
    assert by_name["naming"]["files"] == ["naming.rs"]


def test_suggest_config_rust_bin_target_default_path(tmp_path, write_file):
    # a [[bin]] entry with no "path" falls back to Cargo's own src/bin/<name>.rs convention
    write_file(tmp_path, "Cargo.toml", (
        '[package]\n'
        'name = "myapp"\n'
        '\n'
        '[[bin]]\n'
        'name = "helper"\n'
    ))
    write_file(tmp_path, "src/bin/helper.rs", "fn main() {}\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    by_name = {l["name"]: l for l in draft["layers"]}
    assert by_name["helper"]["files"] == ["helper.rs"]


def test_suggest_config_default_layer_reflects_language_specific_matches(tmp_path, write_file):
    # default_layer must be computed AFTER Rust/C-specific layers are added, not before - a
    # project whose only match comes from a language-specific convention (no general-dict
    # hit at all) still ends up with a non-empty `layers`, so default_layer should say
    # "backend/other", not "other"
    write_file(tmp_path, "Cargo.toml", '[package]\nname = "myapp"\n')
    write_file(tmp_path, "tests/integration.rs", "fn test_foo() {}\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    assert draft["layers"] != []
    assert draft["default_layer"] == "backend/other"


def test_suggest_config_rust_heuristics_skipped_without_rust(tmp_path, write_file):
    # a Python project with a coincidental "tests" dir shouldn't get Rust-flavored layers
    write_file(tmp_path, "tests/test_foo.py", "def test_foo(): pass\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    layer_names = {l["name"] for l in draft["layers"]}
    assert "tests" not in layer_names


def test_suggest_config_c_conventional_dirs(tmp_path, write_file):
    write_file(tmp_path, "include/mylib.h", "#ifndef MYLIB_H\n#define MYLIB_H\n#endif\n")
    write_file(tmp_path, "src/mylib.c", "int foo(void) { return 1; }\n")
    write_file(tmp_path, "tests/test_mylib.c", "int main(void) { return 0; }\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    layer_names = {l["name"] for l in draft["layers"]}
    assert {"include", "tests"} <= layer_names


def test_suggest_config_c_platform_dirs(tmp_path, write_file):
    # native cross-platform C apps split GUI/OS-integration code into dedicated directories
    # (PuTTY-style windows/unix) the same way a web project splits out "frontend"
    write_file(tmp_path, "windows/window.c", "int WinMain(void) { return 0; }\n")
    write_file(tmp_path, "unix/gtkwin.c", "int main(void) { return 0; }\n")
    write_file(tmp_path, "core.c", "int core_fn(void) { return 1; }\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    by_name = {l["name"]: l for l in draft["layers"]}
    assert "windows" in by_name["platform"]["dirs"]
    assert "unix" in by_name["platform"]["dirs"]


def test_suggest_config_c_vendor_dir(tmp_path, write_file):
    write_file(tmp_path, "third_party/zlib/zlib.c", "int z(void) { return 1; }\n")
    write_file(tmp_path, "src/main.c", "int main(void) { return 0; }\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    by_name = {l["name"]: l for l in draft["layers"]}
    assert "third_party" in by_name["vendor"]["dirs"]


def test_suggest_config_c_heuristics_skipped_without_c(tmp_path, write_file):
    # a Python project with coincidental "include"/"windows"-named dirs shouldn't get
    # C-flavored layers
    write_file(tmp_path, "include/foo.py", "def foo(): pass\n")
    write_file(tmp_path, "windows/bar.py", "def bar(): pass\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    layer_names = {l["name"] for l in draft["layers"]}
    assert "include" not in layer_names
    assert "platform" not in layer_names


def test_suggest_config_cpp_conventional_dirs(tmp_path, write_file):
    # a pure C++ project (no .c files at all) still gets the C-family layer candidates -
    # this is the same "C-family project" convention, not distinctly a C one
    write_file(tmp_path, "include/mylib.hpp", "class Widget {};\n")
    write_file(tmp_path, "src/mylib.cpp", "int foo() { return 1; }\n")
    write_file(tmp_path, "tests/test_mylib.cpp", "int main() { return 0; }\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    layer_names = {l["name"] for l in draft["layers"]}
    assert {"include", "tests"} <= layer_names
    assert "C++" in draft["_detected_languages"]


def test_suggest_config_c_and_cpp_together_do_not_duplicate_layers(tmp_path, write_file):
    # a real, common combination: plain .h headers (flags "C") alongside .cpp sources (flags
    # "C++") in the same project - the shared C-family dict must apply exactly once, not
    # once per detected language
    write_file(tmp_path, "include/mylib.h", "void foo(void);\n")
    write_file(tmp_path, "src/mylib.cpp", "void foo() {}\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    assert {"C", "C++"} <= set(draft["_detected_languages"])
    by_name: dict[str, list] = {}
    for entry in draft["layers"]:
        by_name.setdefault(entry["name"], []).append(entry)
    assert len(by_name["include"]) == 1


def test_find_cmake_targets_literal_target_and_sources(tmp_path, write_file):
    write_file(tmp_path, "CMakeLists.txt",
               "add_library(mylib STATIC src/foo.c src/bar.c)\n")
    write_file(tmp_path, "src/foo.c", "int foo(void) { return 1; }\n")
    write_file(tmp_path, "src/bar.c", "int bar(void) { return 2; }\n")
    targets = init_wizard._find_cmake_targets(tmp_path)
    assert targets == [("mylib", ["bar.c", "foo.c"])]


def test_find_cmake_targets_skips_variable_target_name(tmp_path, write_file):
    # curl's real libcurl/CLI targets are all declared this way - nothing downstream is
    # resolvable once the target's own name isn't literal text
    write_file(tmp_path, "CMakeLists.txt", "add_library(${LIB_NAME} STATIC src/foo.c)\n")
    write_file(tmp_path, "src/foo.c", "int foo(void) { return 1; }\n")
    assert init_wizard._find_cmake_targets(tmp_path) == []


def test_find_cmake_targets_drops_only_dynamic_source_tokens(tmp_path, write_file):
    # real PuTTY calls mix literal sources with one generator-expression entry in the same
    # call - the literal sources must still be found, not discarded along with the dynamic one
    write_file(tmp_path, "CMakeLists.txt",
               "add_library(network STATIC\n"
               "  errsock.c\n"
               "  $<TARGET_OBJECTS:logging>\n"
               "  proxy.c)\n")
    write_file(tmp_path, "errsock.c", "int e(void) { return 1; }\n")
    write_file(tmp_path, "proxy.c", "int p(void) { return 1; }\n")
    targets = init_wizard._find_cmake_targets(tmp_path)
    assert targets == [("network", ["errsock.c", "proxy.c"])]


def test_find_cmake_targets_aborts_whole_target_on_bare_variable_reference(tmp_path, write_file):
    # real bug found validating against libuv: `add_executable(uv_run_tests
    # ${uv_test_sources} uv_win_longpath.manifest)` builds its entire ~150-file source list
    # into one variable first (list(APPEND uv_test_sources ...)) - dropping only the
    # variable token and keeping the incidental literal one would report a
    # "complete-looking" 1-file target that's actually missing ~150 real sources, worse
    # than not detecting it at all. Unlike a $<...> generator expression (additive, safe to
    # drop alone), a bare ${...} variable reference must abort the whole target.
    write_file(tmp_path, "CMakeLists.txt",
               "add_executable(myapp ${SOURCES} main.c)\n")
    write_file(tmp_path, "main.c", "int main(void) { return 0; }\n")
    assert init_wizard._find_cmake_targets(tmp_path) == []


def test_find_cmake_targets_skips_interface_target_with_no_real_sources(tmp_path, write_file):
    write_file(tmp_path, "CMakeLists.txt", "add_library(header_only INTERFACE)\n")
    assert init_wizard._find_cmake_targets(tmp_path) == []


def test_find_cmake_targets_merges_same_name_across_manifests(tmp_path, write_file):
    # cross-platform C's common idiom - PuTTY declares "pageant" once per platform dir with
    # overlapping-but-not-identical sources; both genuinely belong to the same deliverable
    write_file(tmp_path, "unix/CMakeLists.txt", "add_executable(pageant unix_pageant.c)\n")
    write_file(tmp_path, "windows/CMakeLists.txt", "add_executable(pageant win_pageant.c)\n")
    write_file(tmp_path, "unix/unix_pageant.c", "int main(void) { return 0; }\n")
    write_file(tmp_path, "windows/win_pageant.c", "int main(void) { return 0; }\n")
    targets = init_wizard._find_cmake_targets(tmp_path)
    assert targets == [("pageant", ["unix_pageant.c", "win_pageant.c"])]


def test_find_cmake_targets_handles_trailing_comment_inside_call(tmp_path, write_file):
    # a stray ')' inside a real in-call comment (seen in PuTTY's CMakeLists.txt) must not
    # prematurely close the paren-depth scan
    write_file(tmp_path, "CMakeLists.txt",
               "add_library(mylib STATIC\n"
               "  foo.c # see foo(bar) for details\n"
               "  bar.c)\n")
    write_file(tmp_path, "foo.c", "int foo(void) { return 1; }\n")
    write_file(tmp_path, "bar.c", "int bar(void) { return 1; }\n")
    targets = init_wizard._find_cmake_targets(tmp_path)
    assert targets == [("mylib", ["bar.c", "foo.c"])]


def test_find_cmake_targets_ignores_custom_macro_with_add_executable_suffix(tmp_path, write_file):
    # fmt's real CUDA test tree calls a distinct `cuda_add_executable(...)` macro with no
    # guaranteed argument shape - a bare substring match would misfire on it
    write_file(tmp_path, "CMakeLists.txt", "cuda_add_executable(gpu_test a.cu b.cc)\n")
    assert init_wizard._find_cmake_targets(tmp_path) == []


def test_suggest_config_cmake_targets_become_modules_not_layers(tmp_path, write_file):
    # a named CMake target ("myapp") is a sibling deliverable/package, not a recurring
    # architectural role - that's the module dimension (module_rules), the same reasoning
    # already applied to Rust crate names, not layer (which Rust [[bin]] targets earn
    # instead, since a bin has no directory of its own to be grouped into as a module)
    write_file(tmp_path, "CMakeLists.txt", "add_executable(myapp src/main.c)\n")
    write_file(tmp_path, "src/main.c", "int main(void) { return 0; }\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    by_name = {m["name"]: m for m in draft["modules"]}
    assert by_name["myapp"]["files"] == ["main.c"]
    assert "myapp" not in {l["name"] for l in draft["layers"]}


def test_suggest_config_cmake_heuristics_skipped_without_c(tmp_path, write_file):
    # #25 gates CMake target detection on "C" alone (not yet "C++") - documents the
    # intentional sequencing with #26, which broadens this to include C++ projects
    write_file(tmp_path, "CMakeLists.txt", "add_executable(myapp src/main.cpp)\n")
    write_file(tmp_path, "src/main.cpp", "int main() { return 0; }\n")
    draft = init_wizard.suggest_project_config(tmp_path)
    assert "C++" in draft["_detected_languages"]
    assert "C" not in draft["_detected_languages"]
    assert draft["modules"] == []


def test_write_draft_config_writes_when_absent(tmp_path):
    draft = {"layers": [], "default_layer": "backend/other", "translations": {"files": []},
              "extra_excluded_dirs": []}
    out = init_wizard.write_draft_config(tmp_path, draft)
    assert out == tmp_path / ".layergrep.json"
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["layers"] == []


def test_write_draft_config_never_overwrites_existing(tmp_path, write_file):
    write_file(tmp_path, ".layergrep.json", json.dumps({"layers": [{"name": "custom", "dirs": [], "files": []}]}))
    draft = {"layers": [{"name": "should_not_appear", "dirs": [], "files": []}],
              "default_layer": "backend/other", "translations": {"files": []}, "extra_excluded_dirs": []}
    out = init_wizard.write_draft_config(tmp_path, draft)
    assert out is None
    on_disk = json.loads((tmp_path / ".layergrep.json").read_text(encoding="utf-8"))
    assert on_disk["layers"][0]["name"] == "custom"
