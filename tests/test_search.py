from __future__ import annotations

import pytest

import db_schema
import indexer
import model_config
import search


def test_extract_literal_tokens_quoted_string():
    tokens = search.extract_literal_tokens('find the handler with route "list_task_statuses"')
    assert "list_task_statuses" in tokens


def test_extract_literal_tokens_snake_case_identifier():
    tokens = search.extract_literal_tokens("where is update_task_status defined")
    assert "update_task_status" in tokens


def test_extract_literal_tokens_constant_case():
    tokens = search.extract_literal_tokens("what does TASK_QUEUE_RETRY_LIMIT_MODE do")
    assert "TASK_QUEUE_RETRY_LIMIT_MODE" in tokens


def test_extract_literal_tokens_dotted_path():
    tokens = search.extract_literal_tokens("check settings.task.default_status")
    assert "settings.task.default_status" in tokens


def test_extract_literal_tokens_ignores_ordinary_prose():
    tokens = search.extract_literal_tokens("handler that updates the user status")
    assert tokens == []


def test_extract_literal_tokens_longest_first():
    tokens = search.extract_literal_tokens("look for update_task_status or just status")
    assert tokens[0] == "update_task_status"


def test_extract_literal_tokens_dedupes():
    tokens = search.extract_literal_tokens("task_statuses and again task_statuses")
    assert tokens.count("task_statuses") == 1


def test_extract_chunk_literals_pulls_quoted_strings_only():
    text = "def foo():\n    router('/task_statuses', name='list_task_statuses')\n"
    literals = search._extract_chunk_literals(text)
    assert "/task_statuses" in literals
    assert "list_task_statuses" in literals


def test_extract_chunk_literals_skips_short_and_numeric():
    text = "x = '12'\ny = 'ok'\nz = 'a_real_identifier'\n"
    literals = search._extract_chunk_literals(text)
    assert "12" not in literals
    assert "a_real_identifier" in literals


def test_extract_chunk_literals_skips_pure_formatting_strings():
    # A join separator with a real CRLF+indent inside it, and a bare comment-marker string -
    # both are quoted substrings with no alphanumeric character at all (punctuation/whitespace/
    # control chars only), not identifiers.
    text = "parts = x.join(',\r\n    ')\ny = 'ok'\nz = ' # '\n"
    literals = search._extract_chunk_literals(text)
    assert ",\r\n    " not in literals
    assert " # " not in literals
    assert "ok" not in literals  # too short (len < 3), unrelated to this check but confirms parsing
    assert not any(not any(c.isalnum() for c in lit) for lit in literals)


def test_extract_chunk_literals_respects_limit():
    text = "\n".join(f"x{i} = 'literal_value_{i}'" for i in range(10))
    literals = search._extract_chunk_literals(text, limit=3)
    assert len(literals) == 3


@pytest.fixture
def conn(tmp_path):
    return db_schema.open_db(tmp_path / "t.db", model_config.MODEL_NAME, tmp_path)


def test_search_literal_matches_name(conn):
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text, layer) "
        "VALUES ('a.py', 1, 1, 'function', 'update_task_status', 'x', 'backend/api')"
    )
    conn.commit()
    results = search.search_literal(conn, "update_task_status")
    assert len(results) == 1
    assert results[0][4] == "update_task_status"
    assert results[0][7] == 0  # exact name match -> tier 0


def test_search_literal_matches_file_path(conn):
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text, layer) "
        "VALUES ('daemons/queue_worker__starter.py', 1, 1, 'function', 'main', 'x', 'daemons')"
    )
    conn.commit()
    results = search.search_literal(conn, "queue_worker__starter")
    assert len(results) == 1
    assert results[0][0] == "daemons/queue_worker__starter.py"


def test_search_literal_no_tokens_returns_empty(conn):
    assert search.search_literal(conn, "just some ordinary prose") == []


def test_search_literal_no_match_returns_empty(conn):
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text, layer) "
        "VALUES ('a.py', 1, 1, 'function', 'unrelated_name', 'x', 'backend/api')"
    )
    conn.commit()
    assert search.search_literal(conn, "completely_different_token") == []


def _seed_chunk(conn, file_path, name, text, layer):
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text, layer) "
        "VALUES (?, 1, 1, 'function', ?, ?, ?)",
        (file_path, name, text, layer),
    )


def test_cross_layer_literal_links_finds_shared_literal_in_other_layer(conn):
    _seed_chunk(conn, "frontend/status.jsx", "saveStatus",
                "fetch('update_task_status')", "frontend")
    _seed_chunk(conn, "backend/handlers.py", "UpdateTaskStatus",
                "route = 'update_task_status'", "backend/api")
    conn.commit()

    by_layer = {"frontend": {"app": [("frontend/status.jsx", 1, 1, "function", "saveStatus", 0.1,
                                       "fetch('update_task_status')")]}}
    links = search.cross_layer_literal_links(conn, by_layer)
    assert "status.jsx::saveStatus" in links
    literal, matches = links["status.jsx::saveStatus"][0]
    assert literal == "update_task_status"
    assert matches[0][4] == "UpdateTaskStatus"


def test_cross_layer_literal_links_skips_noisy_literal(conn):
    _seed_chunk(conn, "frontend/x.jsx", "caller", "doThing('GET')", "frontend")
    for i in range(search._LITERAL_NOISE_THRESHOLD + 5):
        _seed_chunk(conn, f"backend/f{i}.py", f"fn{i}", "method = 'GET'", "backend/api")
    conn.commit()

    by_layer = {"frontend": {"app": [("frontend/x.jsx", 1, 1, "function", "caller", 0.1, "doThing('GET')")]}}
    links = search.cross_layer_literal_links(conn, by_layer)
    assert links == {}


def test_cross_layer_literal_links_does_not_link_within_same_layer(conn):
    _seed_chunk(conn, "backend/a.py", "caller", "call('specific_marker_value')", "backend/api")
    _seed_chunk(conn, "backend/b.py", "other", "x = 'specific_marker_value'", "backend/api")
    conn.commit()

    by_layer = {"backend/api": {"app": [("backend/a.py", 1, 1, "function", "caller", 0.1,
                                          "call('specific_marker_value')")]}}
    links = search.cross_layer_literal_links(conn, by_layer)
    assert links == {}


def test_format_results_no_literal_matches_shows_none():
    result = {"literal": [], "json_literal": [], "by_layer": {}, "expanded_via_imports": {},
              "cross_layer_literal_links": {}}
    out = search.format_results("some query", result)
    assert "(none)" in out


def test_format_results_renders_layer_module_grouping():
    result = {
        "literal": [],
        "json_literal": [],
        "by_layer": {"backend/api": {"webapp": [("a.py", 1, 5, "function", "foo", 0.1234, "def foo(): pass")]}},
        "expanded_via_imports": {},
        "cross_layer_literal_links": {},
    }
    out = search.format_results("q", result)
    assert "--- backend/api ---" in out
    assert "[module: webapp]" in out
    assert "0.1234" in out
    assert "def foo(): pass" in out


def test_format_results_uses_clickable_single_number_line_format():
    result = {
        "literal": [], "json_literal": [],
        "by_layer": {"x": {"m": [("a.py", 10, 20, "function", "foo", 0.5, "text")]}},
        "expanded_via_imports": {}, "cross_layer_literal_links": {},
    }
    out = search.format_results("q", result)
    assert "a.py:10, ends 20" in out
    assert "a.py:10-20" not in out


@pytest.mark.slow
def test_search_hybrid_full_pipeline(tmp_path, write_file):
    write_file(tmp_path, "webapp/api/handlers.py",
               "def update_task_status(user_id, status):\n"
               "    '''Set the operator status for a user.'''\n"
               "    return status\n")
    write_file(tmp_path, "webapp/langs/ru.json", '{"status_error": "Ошибка установки статуса"}')

    db_path = indexer.default_db_path(tmp_path, model_config.MODEL_NAME)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn2 = db_schema.open_db(db_path, model_config.MODEL_NAME, tmp_path)
    indexer.index_project(conn2, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)
    indexer.index_json_translations(conn2, tmp_path / "webapp" / "langs" / "ru.json", tmp_path,
                                     model_name=model_config.MODEL_NAME)

    result = search.search_hybrid(conn2, "update_task_status", tmp_path,
                                   model_name=model_config.MODEL_NAME)
    assert len(result["literal"]) == 1
    assert result["literal"][0][4] == "update_task_status"

    formatted = search.format_results("update_task_status", result)
    assert "update_task_status" in formatted
