from __future__ import annotations

import json
import sqlite3

import pytest

import db_schema
import model_config


def _make_conn(tmp_path, name="test.db"):
    return db_schema.open_db(tmp_path / name, model_config.MODEL_NAME, tmp_path)


def test_open_db_creates_expected_tables(tmp_path):
    conn = _make_conn(tmp_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"meta", "files", "chunks", "imports", "json_entries"} <= tables
    conn.close()


def test_open_db_stamps_embedding_model_on_first_open(tmp_path):
    conn = _make_conn(tmp_path)
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
    assert row[0] == model_config.MODEL_NAME
    conn.close()


def test_open_db_raises_on_model_mismatch(tmp_path):
    conn = _make_conn(tmp_path)
    conn.close()
    with pytest.raises(db_schema.ModelMismatchError):
        db_schema.open_db(tmp_path / "test.db", "some/other-model", tmp_path)


def test_open_db_reopen_is_idempotent(tmp_path):
    conn1 = _make_conn(tmp_path)
    conn1.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text) VALUES (?, 1, 1, 'function', 'foo', 'x')",
        (str(tmp_path / "a.py"),),
    )
    conn1.commit()
    conn1.close()

    conn2 = _make_conn(tmp_path)
    count = conn2.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 1
    conn2.close()


def test_open_db_backfills_layer_and_module_for_existing_chunks(tmp_path, write_file):
    write_file(tmp_path, ".layergrep.json", json.dumps({
        "layers": [{"name": "backend/api", "dirs": ["api"], "files": []}],
        "default_layer": "backend/other",
    }))
    conn = _make_conn(tmp_path)
    file_path = str(tmp_path / "webapp" / "api" / "handlers.py")
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text) VALUES (?, 1, 1, 'function', 'foo', 'x')",
        (file_path,),
    )
    conn.commit()
    conn.close()

    # reopen: open_db's backfill loop runs over already-indexed chunks each time it's called
    conn2 = _make_conn(tmp_path)
    layer, module = conn2.execute("SELECT layer, module FROM chunks WHERE file_path = ?", (file_path,)).fetchone()
    assert layer == "backend/api"
    assert module == "webapp"
    conn2.close()


def test_open_db_warns_when_module_dimension_not_discriminating(tmp_path, caplog):
    conn = _make_conn(tmp_path)
    file_a = str(tmp_path / "onlymod" / "a.py")
    file_b = str(tmp_path / "onlymod" / "sub" / "b.py")
    conn.executemany(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text) VALUES (?, 1, 1, 'function', 'x', 'y')",
        [(file_a,), (file_b,)],
    )
    conn.commit()
    conn.close()

    with caplog.at_level("WARNING", logger="layergrep.db_schema"):
        conn2 = _make_conn(tmp_path)
    assert any("module dimension isn't discriminating" in r.message for r in caplog.records)
    conn2.close()


def test_open_db_no_warning_when_module_depth_discriminates(tmp_path, write_file, caplog):
    write_file(tmp_path, ".layergrep.json", json.dumps({"module_depth": 2}))
    conn = _make_conn(tmp_path)
    file_a = str(tmp_path / "onlymod" / "a.py")
    file_b = str(tmp_path / "onlymod" / "sub" / "b.py")
    conn.executemany(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text) VALUES (?, 1, 1, 'function', 'x', 'y')",
        [(file_a,), (file_b,)],
    )
    conn.commit()
    conn.close()

    with caplog.at_level("WARNING", logger="layergrep.db_schema"):
        conn2 = _make_conn(tmp_path)
    assert not any("module dimension isn't discriminating" in r.message for r in caplog.records)
    modules = {row[0] for row in conn2.execute("SELECT module FROM chunks")}
    assert modules == {"onlymod", "onlymod/sub"}
    conn2.close()


def test_open_db_warns_when_no_layer_rules_matched(tmp_path, write_file, caplog):
    write_file(tmp_path, ".layergrep.json", json.dumps({
        "layers": [{"name": "backend/api", "dirs": ["nonexistent_dir_name"], "files": []}],
    }))
    conn = _make_conn(tmp_path)
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text) VALUES (?, 1, 1, 'function', 'x', 'y')",
        (str(tmp_path / "a.py"),),
    )
    conn.commit()
    conn.close()

    with caplog.at_level("WARNING", logger="layergrep.db_schema"):
        conn2 = _make_conn(tmp_path)
    assert any("NONE matched" in r.message for r in caplog.records)
    conn2.close()


def test_apply_version_migration_calls_callback_on_mismatch(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()

    calls = []
    db_schema._apply_version_migration(conn, "some_version", 3, lambda c: calls.append(c) or 1)

    assert calls == [conn]
    assert conn.execute("SELECT value FROM meta WHERE key = 'some_version'").fetchone()[0] == "3"


def test_apply_version_migration_skips_callback_when_version_matches(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta (key, value) VALUES ('some_version', '3')")
    conn.commit()

    calls = []
    db_schema._apply_version_migration(conn, "some_version", 3, lambda c: calls.append(c))

    assert calls == []


def test_apply_version_migration_reruns_on_version_bump(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta (key, value) VALUES ('some_version', '3')")
    conn.commit()

    calls = []
    db_schema._apply_version_migration(conn, "some_version", 4, lambda c: calls.append(c) or 1)

    assert calls == [conn]
    assert conn.execute("SELECT value FROM meta WHERE key = 'some_version'").fetchone()[0] == "4"


def test_apply_version_migration_returns_true_only_when_callback_wiped_something(tmp_path):
    # issue #37: the return value drives whether a "you should reindex" notice gets recorded -
    # a version mismatch that had nothing real to wipe (e.g. a brand-new project, or a
    # brand-new extension group with zero matching files yet) must not report True
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()

    assert db_schema._apply_version_migration(conn, "some_version", 1, lambda c: 0) is False

    conn.execute("UPDATE meta SET value = '1'")
    assert db_schema._apply_version_migration(conn, "some_version", 2, lambda c: 3) is True


def test_wipe_for_imports_version_group_only_touches_its_own_extensions(tmp_path):
    # issue #35: a version bump in one group (e.g. rust) must not force reprocessing of, or
    # drop the already-correct import edges belonging to, a completely different group
    # (e.g. python) - the whole point of splitting the version by group in the first place
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE files (path TEXT PRIMARY KEY, content_hash TEXT NOT NULL)")
    conn.execute("CREATE TABLE imports (id INTEGER PRIMARY KEY, source_file TEXT NOT NULL, "
                  "target_file TEXT NOT NULL, module TEXT NOT NULL)")
    py_file = str(tmp_path / "a.py")
    rs_file = str(tmp_path / "b.rs")
    conn.executemany("INSERT INTO files (path, content_hash) VALUES (?, 'h')", [(py_file,), (rs_file,)])
    conn.executemany(
        "INSERT INTO imports (source_file, target_file, module) VALUES (?, 'x', 'm')",
        [(py_file,), (rs_file,)],
    )
    conn.commit()

    wipe_rust_only = db_schema._make_wipe_for_imports_version_group(frozenset({".rs"}))
    wipe_rust_only(conn)

    assert conn.execute("SELECT COUNT(*) FROM files WHERE path = ?", (py_file,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM imports WHERE source_file = ?", (py_file,)).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM files WHERE path = ?", (rs_file,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM imports WHERE source_file = ?", (rs_file,)).fetchone()[0] == 0


def test_open_db_migrates_each_import_version_group_independently(tmp_path, write_file):
    write_file(tmp_path, "a.py", "import os\n")
    write_file(tmp_path, "b.rs", "fn main() {}\n")
    conn = _make_conn(tmp_path)
    for group in db_schema.IMPORTS_VERSIONS:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (f"imports_version_{group}",)
        ).fetchone()
        assert row is not None
    conn.close()


def test_open_db_fresh_project_produces_no_pending_reindex_notice(tmp_path):
    # a brand-new db has nothing to "re"-index yet - every version key starts at `row is
    # None`, which must NOT be mistaken for "something just got wiped" (issue #37)
    conn = _make_conn(tmp_path)
    assert db_schema.get_pending_reindex_notice(conn) is None
    conn.close()


def test_open_db_records_notice_when_an_existing_group_actually_gets_wiped(tmp_path, write_file):
    write_file(tmp_path, "a.py", "import os\n")
    conn = _make_conn(tmp_path)
    conn.execute(
        "INSERT INTO files (path, content_hash) VALUES (?, 'h')", (str(tmp_path / "a.py"),)
    )
    # simulate "layergrep code changed since this was indexed" - downgrade the stamped
    # version for a group this project actually has a file in
    conn.execute("UPDATE meta SET value = '0' WHERE key = 'imports_version_python'")
    conn.commit()
    conn.close()

    conn2 = _make_conn(tmp_path)
    notice = db_schema.get_pending_reindex_notice(conn2)
    assert notice is not None
    assert "imports:python" in notice
    conn2.close()


def test_open_db_no_notice_when_bumped_group_has_no_matching_files(tmp_path, write_file):
    # the false-positive guard: a version bump for a group with zero files of that type in
    # this project (e.g. rust, in a pure-Python project) must not produce a notice - there's
    # really nothing to reindex
    write_file(tmp_path, "a.py", "import os\n")
    conn = _make_conn(tmp_path)
    conn.execute(
        "INSERT INTO files (path, content_hash) VALUES (?, 'h')", (str(tmp_path / "a.py"),)
    )
    conn.execute("UPDATE meta SET value = '0' WHERE key = 'imports_version_rust'")
    conn.commit()
    conn.close()

    conn2 = _make_conn(tmp_path)
    assert db_schema.get_pending_reindex_notice(conn2) is None
    conn2.close()


def test_clear_pending_reindex_notice_removes_it(tmp_path, write_file):
    write_file(tmp_path, "a.py", "import os\n")
    conn = _make_conn(tmp_path)
    conn.execute(
        "INSERT INTO files (path, content_hash) VALUES (?, 'h')", (str(tmp_path / "a.py"),)
    )
    conn.execute("UPDATE meta SET value = '0' WHERE key = 'imports_version_python'")
    conn.commit()
    conn.close()

    conn2 = _make_conn(tmp_path)
    assert db_schema.get_pending_reindex_notice(conn2) is not None
    db_schema.clear_pending_reindex_notice(conn2)
    assert db_schema.get_pending_reindex_notice(conn2) is None
    conn2.close()
