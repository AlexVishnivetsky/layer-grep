from __future__ import annotations

import db_schema
import graph_analytics
import model_config


def _conn(tmp_path):
    return db_schema.open_db(tmp_path / "t.db", model_config.MODEL_NAME, tmp_path)


def test_compute_module_stats_foundation_role(tmp_path):
    # engine: only imported BY others, imports nothing back -> foundation
    conn = _conn(tmp_path)
    conn.execute("INSERT INTO imports (source_file, target_file, module) VALUES (?, ?, 'x')",
                 (str(tmp_path / "webapp" / "app.py"), str(tmp_path / "engine" / "core.py")))
    conn.commit()

    stats = graph_analytics.compute_module_stats(conn, tmp_path)
    assert stats["engine"]["role"] == "foundation"
    assert stats["webapp"]["role"] == "extension"


def test_compute_module_stats_application_role_when_both_directions(tmp_path):
    conn = _conn(tmp_path)
    conn.execute("INSERT INTO imports (source_file, target_file, module) VALUES (?, ?, 'x')",
                 (str(tmp_path / "webapp" / "app.py"), str(tmp_path / "engine" / "core.py")))
    conn.execute("INSERT INTO imports (source_file, target_file, module) VALUES (?, ?, 'x')",
                 (str(tmp_path / "extension" / "ext.py"), str(tmp_path / "webapp" / "helpers.py")))
    conn.commit()

    stats = graph_analytics.compute_module_stats(conn, tmp_path)
    assert stats["webapp"]["role"] == "application"  # imports engine AND is imported by extension
    assert stats["engine"]["role"] == "foundation"
    assert stats["extension"]["role"] == "extension"


def test_compute_module_stats_ignores_same_module_edges(tmp_path):
    # an import within the same module (e.g. webapp/a.py -> webapp/b.py) isn't a
    # cross-module edge and must not count toward role classification
    conn = _conn(tmp_path)
    conn.execute("INSERT INTO imports (source_file, target_file, module) VALUES (?, ?, 'x')",
                 (str(tmp_path / "webapp" / "a.py"), str(tmp_path / "webapp" / "b.py")))
    conn.commit()

    stats = graph_analytics.compute_module_stats(conn, tmp_path)
    assert stats == {}


def test_compute_module_stats_respects_module_depth(tmp_path, write_file):
    import json
    write_file(tmp_path, ".layergrep.json", json.dumps({"module_depth": 2}))
    conn = _conn(tmp_path)
    conn.execute("INSERT INTO imports (source_file, target_file, module) VALUES (?, ?, 'x')",
                 (str(tmp_path / "packages" / "service-a" / "app.py"),
                  str(tmp_path / "packages" / "service-b" / "engine.py")))
    conn.commit()

    stats = graph_analytics.compute_module_stats(conn, tmp_path)
    assert set(stats) == {"packages/service-a", "packages/service-b"}


def test_find_runtime_unification_detects_shared_entry_point(tmp_path, write_file):
    write_file(tmp_path, "webapp/__init__.py", "")
    write_file(tmp_path, "webapp/app.py", "def create_app():\n    pass\n")
    write_file(tmp_path, "extension/__init__.py", "")
    write_file(tmp_path, "extension/app.py", "def create_app():\n    pass\n")
    write_file(tmp_path, "wsgi.py", "import webapp.app\nimport extension.app\n")

    unification = graph_analytics.find_runtime_unification(tmp_path)
    entry = str(tmp_path / "wsgi.py")
    assert entry in unification
    assert set(unification[entry]) == {"webapp", "extension"}


def test_find_runtime_unification_ignores_single_module_entry_points(tmp_path, write_file):
    write_file(tmp_path, "webapp/__init__.py", "")
    write_file(tmp_path, "webapp/app.py", "def create_app():\n    pass\n")
    write_file(tmp_path, "manage.py", "import webapp.app\n")

    unification = graph_analytics.find_runtime_unification(tmp_path)
    assert unification == {}
