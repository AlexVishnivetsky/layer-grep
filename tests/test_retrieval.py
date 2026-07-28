from __future__ import annotations

import pytest

import db_schema
import indexer
import model_config
import retrieval


def test_escape_like_escapes_wildcards():
    assert retrieval.escape_like("a_b%c") == "a\\_b\\%c"
    assert retrieval.escape_like("no_wildcards_here_") == "no\\_wildcards\\_here\\_"
    assert retrieval.escape_like("literal\\backslash") == "literal\\\\backslash"


def test_escape_like_makes_underscore_match_literally(tmp_path):
    conn = db_schema.open_db(tmp_path / "t.db", model_config.MODEL_NAME, tmp_path)
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text) VALUES ('a.py', 1, 1, 'function', 'get_user', 'x')"
    )
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text) VALUES ('b.py', 1, 1, 'function', 'getXuser', 'x')"
    )
    conn.commit()

    like = f"%{retrieval.escape_like('get_user')}%"
    rows = conn.execute("SELECT name FROM chunks WHERE name LIKE ? ESCAPE '\\'", (like,)).fetchall()
    assert {r[0] for r in rows} == {"get_user"}  # "getXuser" must NOT match


@pytest.fixture
def json_conn(tmp_path):
    conn = db_schema.open_db(tmp_path / "t.db", model_config.MODEL_NAME, tmp_path)
    conn.execute(
        "INSERT INTO json_entries (file_path, key_path, value) VALUES ('ru.json', 'a.greeting', 'Привет мир')"
    )
    conn.execute(
        "INSERT INTO json_entries (file_path, key_path, value) VALUES ('ru.json', 'a.farewell', 'Пока')"
    )
    conn.commit()
    return conn


def test_search_json_literal_matches_value_substring(json_conn):
    results = retrieval.search_json_literal(json_conn, "Привет")
    assert len(results) == 1
    assert results[0][1] == "a.greeting"


def test_search_json_literal_matches_key_path_substring(json_conn):
    results = retrieval.search_json_literal(json_conn, "farewell")
    assert len(results) == 1
    assert results[0][2] == "Пока"


def test_search_json_literal_no_match_returns_empty(json_conn):
    assert retrieval.search_json_literal(json_conn, "nonexistent phrase") == []


def test_get_forward_imports_returns_direct_targets_only(tmp_path):
    conn = db_schema.open_db(tmp_path / "t.db", model_config.MODEL_NAME, tmp_path)
    conn.execute("INSERT INTO imports (source_file, target_file, module) VALUES ('a.py', 'b.py', 'b')")
    conn.execute("INSERT INTO imports (source_file, target_file, module) VALUES ('a.py', 'c.py', 'c')")
    conn.execute("INSERT INTO imports (source_file, target_file, module) VALUES ('b.py', 'd.py', 'd')")
    conn.commit()

    result = retrieval.get_forward_imports(conn, "a.py")
    assert set(result) == {("b.py", "b"), ("c.py", "c")}


def test_expand_via_imports_skips_noisy_targets(tmp_path):
    conn = db_schema.open_db(tmp_path / "t.db", model_config.MODEL_NAME, tmp_path)
    # "shared.py" imported by many distinct files (over threshold) - should be treated as
    # a widely-shared utility and skipped, not surfaced as feature-specific context
    for i in range(retrieval._IMPORT_TARGET_NOISE_THRESHOLD + 5):
        conn.execute("INSERT INTO imports (source_file, target_file, module) VALUES (?, 'shared.py', 'shared')",
                     (f"importer{i}.py",))
    conn.execute("INSERT INTO imports (source_file, target_file, module) VALUES ('seed.py', 'narrow.py', 'narrow')")
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text, layer) "
        "VALUES ('shared.py', 1, 1, 'function', 'shared_fn', 'x', 'backend/other')"
    )
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text, layer) "
        "VALUES ('narrow.py', 1, 1, 'function', 'narrow_fn', 'x', 'backend/other')"
    )
    conn.commit()

    expanded = retrieval.expand_via_imports(conn, ["seed.py"])
    assert "seed.py" in expanded
    modules = {entry[0] for entry in expanded["seed.py"]}
    assert modules == {"narrow"}  # "shared" filtered out as noise


def test_expand_via_imports_no_entries_omits_seed_file(tmp_path):
    conn = db_schema.open_db(tmp_path / "t.db", model_config.MODEL_NAME, tmp_path)
    result = retrieval.expand_via_imports(conn, ["lonely.py"])
    assert result == {}


@pytest.mark.slow
class TestSearchByLayersRealModel:
    def _indexed_conn(self, tmp_path, write_file, layers=None):
        if layers:
            import json
            write_file(tmp_path, ".layergrep.json", json.dumps({"layers": layers}))
        db_path = indexer.default_db_path(tmp_path, model_config.MODEL_NAME)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = db_schema.open_db(db_path, model_config.MODEL_NAME, tmp_path)
        return conn

    def test_search_by_layers_groups_by_layer_and_module(self, tmp_path, write_file):
        write_file(tmp_path, "webapp/api/handlers.py",
                   "def set_user_status(user_id, status):\n"
                   "    '''Update the operator status for a user.'''\n"
                   "    return status\n")
        write_file(tmp_path, "engine/api/other.py",
                   "def unrelated_inventory_report():\n"
                   "    '''Generate an inventory report PDF.'''\n"
                   "    return None\n")
        conn = self._indexed_conn(tmp_path, write_file, layers=[{"name": "backend/api", "dirs": ["api"], "files": []}])
        indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)

        by_layer = retrieval.search_by_layers(conn, "update user operator status", tmp_path,
                                               model_name=model_config.MODEL_NAME)
        assert "backend/api" in by_layer
        assert "webapp" in by_layer["backend/api"]

    def test_search_by_layers_empty_layer_is_valid_result(self, tmp_path, write_file):
        write_file(tmp_path, "webapp/api/handlers.py", "def foo():\n    '''Does something.'''\n    return 1\n")
        conn = self._indexed_conn(tmp_path, write_file, layers=[
            {"name": "backend/api", "dirs": ["api"], "files": []},
            {"name": "frontend", "dirs": ["react"], "files": []},
        ])
        indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)

        by_layer = retrieval.search_by_layers(conn, "anything", tmp_path, model_name=model_config.MODEL_NAME)
        assert by_layer["frontend"] == {}  # no frontend code indexed - empty is the honest result

    def test_search_by_layers_respects_k_per_layer_cap(self, tmp_path, write_file):
        body = "\n\n".join(f"def handler_{i}():\n    '''Handles request number {i}.'''\n    return {i}"
                            for i in range(10))
        write_file(tmp_path, "webapp/api/handlers.py", body)
        conn = self._indexed_conn(tmp_path, write_file, layers=[{"name": "backend/api", "dirs": ["api"], "files": []}])
        indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)

        by_layer = retrieval.search_by_layers(conn, "handling a request", tmp_path,
                                               model_name=model_config.MODEL_NAME, k_per_layer=2)
        assert len(by_layer["backend/api"]["webapp"]) <= 2
