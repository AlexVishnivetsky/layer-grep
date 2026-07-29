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
