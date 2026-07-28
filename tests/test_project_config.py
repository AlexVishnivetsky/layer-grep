from __future__ import annotations

import json
from pathlib import Path

import project_config as pconfig


def test_load_project_config_defaults_when_no_file(tmp_path):
    cfg = pconfig.load_project_config(tmp_path)
    assert cfg.layer_rules == []
    assert cfg.default_layer == pconfig._BUILTIN_DEFAULT_LAYER
    assert cfg.translations_files == []
    assert cfg.extra_excluded_dirs == frozenset()
    assert cfg.forced_add == frozenset()
    assert cfg.module_depth == 1
    assert cfg.module_rules == []
    assert "no .layergrep.json" in cfg.source or cfg.source == "defaults (no .layergrep.json found)"


def test_load_project_config_parses_full_file(tmp_path, write_file):
    raw = {
        "layers": [
            {"name": "backend/api", "dirs": ["api"], "files": ["views.py"]},
            {"name": "models", "dirs": ["models"], "files": []},
        ],
        "default_layer": "backend/other",
        "translations": {"files": ["langs/ru.json"]},
        "extra_excluded_dirs": ["plugins", "engine/libraries/ck4"],
        "forced_add": ["target"],
        "module_depth": 2,
        "modules": [
            {"name": "audio-engine", "dirs": [], "files": ["engine.rs", "Deck.tsx"]},
        ],
    }
    write_file(tmp_path, ".layergrep.json", json.dumps(raw))

    cfg = pconfig.load_project_config(tmp_path)
    assert cfg.layer_rules == [
        ("backend/api", ("api",), ("views.py",)),
        ("models", ("models",), ()),
    ]
    assert cfg.default_layer == "backend/other"
    assert cfg.translations_files == ["langs/ru.json"]
    assert cfg.extra_excluded_dirs == frozenset({"plugins", "engine/libraries/ck4"})
    assert cfg.forced_add == frozenset({"target"})
    assert cfg.module_depth == 2
    assert cfg.module_rules == [("audio-engine", (), ("engine.rs", "Deck.tsx"))]
    assert str(tmp_path) in cfg.source


def test_load_project_config_is_cached(tmp_path, write_file):
    # First call with no config file caches the empty-defaults result; writing a config
    # file afterward must NOT change what a second call for the same project_root returns -
    # this mirrors the documented tradeoff (load once per process per project_root).
    first = pconfig.load_project_config(tmp_path)
    write_file(tmp_path, ".layergrep.json", json.dumps({"layers": [{"name": "x", "dirs": ["x"], "files": []}]}))
    second = pconfig.load_project_config(tmp_path)
    assert first is second
    assert second.layer_rules == []


def test_classify_layer_dir_rule_match():
    cfg = pconfig.ProjectConfig(layer_rules=[("backend/api", ("api",), ())])
    assert pconfig.classify_layer(Path("/proj/webapp/api/handlers.py"), cfg) == "backend/api"


def test_classify_layer_file_rule_match():
    cfg = pconfig.ProjectConfig(layer_rules=[("constants", (), ("const.py",))])
    assert pconfig.classify_layer(Path("/proj/webapp/const.py"), cfg) == "constants"


def test_classify_layer_falls_back_to_default():
    cfg = pconfig.ProjectConfig(layer_rules=[("backend/api", ("api",), ())], default_layer="backend/other")
    assert pconfig.classify_layer(Path("/proj/webapp/daemons/worker.py"), cfg) == "backend/other"


def test_classify_layer_first_matching_rule_wins():
    # "models" dir rule comes first - a file matching both should get the earlier rule
    cfg = pconfig.ProjectConfig(layer_rules=[
        ("models", ("models",), ()),
        ("backend/api", ("api",), ()),
    ])
    assert pconfig.classify_layer(Path("/proj/api/models/user.py"), cfg) == "models"


def test_classify_module_default_depth_one(tmp_path):
    (tmp_path / "webapp" / "api").mkdir(parents=True)
    f = tmp_path / "webapp" / "api" / "handlers.py"
    f.write_text("")
    assert pconfig.classify_module(f, tmp_path) == "webapp"


def test_classify_module_top_level_file(tmp_path):
    f = tmp_path / "manage.py"
    f.write_text("")
    assert pconfig.classify_module(f, tmp_path) == "manage.py"


def test_classify_module_outside_project_root_returns_empty(tmp_path):
    other = tmp_path.parent / "elsewhere.py"
    assert pconfig.classify_module(other, tmp_path) == ""


def test_classify_module_depth_two_monorepo(tmp_path):
    (tmp_path / "packages" / "service-a" / "src").mkdir(parents=True)
    f = tmp_path / "packages" / "service-a" / "src" / "index.js"
    f.write_text("")
    assert pconfig.classify_module(f, tmp_path, module_depth=2) == "packages/service-a"


def test_classify_module_depth_never_includes_filename(tmp_path):
    (tmp_path / "packages").mkdir()
    f = tmp_path / "packages" / "index.js"
    f.write_text("")
    assert pconfig.classify_module(f, tmp_path, module_depth=2) == "packages"


def test_classify_module_rule_crosses_directory_boundary(tmp_path):
    # the whole reason module_rules exists: a business entity split across a separate
    # frontend/backend layout, where no single leading-path-components depth could ever
    # unify "backend/engine.rs" and "frontend/Deck.tsx" into one module
    (tmp_path / "backend").mkdir()
    (tmp_path / "frontend").mkdir()
    engine = tmp_path / "backend" / "engine.rs"
    engine.write_text("")
    deck = tmp_path / "frontend" / "Deck.tsx"
    deck.write_text("")
    rules = [("audio-engine", (), ("engine.rs", "Deck.tsx"))]
    assert pconfig.classify_module(engine, tmp_path, module_rules=rules) == "audio-engine"
    assert pconfig.classify_module(deck, tmp_path, module_rules=rules) == "audio-engine"


def test_classify_module_rule_falls_back_to_depth_when_unmatched(tmp_path):
    (tmp_path / "webapp" / "api").mkdir(parents=True)
    f = tmp_path / "webapp" / "api" / "handlers.py"
    f.write_text("")
    rules = [("audio-engine", (), ("engine.rs",))]
    assert pconfig.classify_module(f, tmp_path, module_rules=rules) == "webapp"
