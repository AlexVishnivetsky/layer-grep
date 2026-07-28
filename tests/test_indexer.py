from __future__ import annotations

import pytest

import db_schema
import indexer
import model_config


def test_iter_source_files_finds_supported_extensions(tmp_path, write_file):
    write_file(tmp_path, "a.py", "def foo(): pass\n")
    write_file(tmp_path, "b.js", "function bar() {}\n")
    write_file(tmp_path, "c.txt", "not code\n")
    found = {p.name for p in indexer.iter_source_files(tmp_path)}
    assert found == {"a.py", "b.js"}


def test_iter_source_files_excludes_base_dir_names(tmp_path, write_file):
    write_file(tmp_path, "node_modules/pkg/index.js", "function foo() {}\n")
    write_file(tmp_path, ".venv/Lib/site-packages/thing.py", "def x(): pass\n")
    write_file(tmp_path, "real/app.py", "def y(): pass\n")
    found = {p.name for p in indexer.iter_source_files(tmp_path)}
    assert found == {"app.py"}


def test_iter_source_files_extra_excluded_bare_name(tmp_path, write_file):
    write_file(tmp_path, "plugins/a.py", "def a(): pass\n")
    write_file(tmp_path, "app/b.py", "def b(): pass\n")
    found = {p.name for p in indexer.iter_source_files(tmp_path, extra_excluded_dirs=frozenset({"plugins"}))}
    assert found == {"b.py"}


def test_iter_source_files_extra_excluded_path_prefix_is_scoped(tmp_path, write_file):
    # "engine/libraries/ck4" as a prefix must not exclude "webapp/libraries" - the whole
    # reason extra_excluded_dirs supports path-prefix entries, not just bare names (see
    # iter_source_files' own docstring: "libraries" isn't uniformly vendored-vs-first-party)
    write_file(tmp_path, "engine/libraries/ck4/vendor.js", "function v() {}\n")
    write_file(tmp_path, "webapp/libraries/real.py", "def real(): pass\n")
    found = {p.name for p in indexer.iter_source_files(tmp_path, extra_excluded_dirs=frozenset({"engine/libraries/ck4"}))}
    assert found == {"real.py"}


def test_iter_source_files_forced_add_overrides_builtin_exclusion(tmp_path, write_file):
    # "target" is in EXCLUDED_DIR_NAMES (Cargo's build dir) - a project whose own first-party
    # code happens to live in a directory of that name needs a way to un-exclude it.
    write_file(tmp_path, "target/a.py", "def a(): pass\n")
    write_file(tmp_path, "app/b.py", "def b(): pass\n")
    found = {p.name for p in indexer.iter_source_files(tmp_path, forced_add=frozenset({"target"}))}
    assert found == {"a.py", "b.py"}


def test_iter_source_files_excludes_minified_js(tmp_path, write_file):
    write_file(tmp_path, "lib.min.js", "function x(){}\n")
    write_file(tmp_path, "lib.js", "function y(){}\n")
    found = {p.name for p in indexer.iter_source_files(tmp_path)}
    assert found == {"lib.js"}


def test_iter_source_files_excludes_oversized_files(tmp_path):
    big = tmp_path / "big.py"
    big.write_text("x = 1\n" * 100_000)
    assert big.stat().st_size > indexer.MAX_FILE_SIZE
    found = list(indexer.iter_source_files(tmp_path))
    assert found == []


def test_file_hash_changes_with_content(tmp_path, write_file):
    p = write_file(tmp_path, "a.py", "def foo(): pass\n")
    h1 = indexer.file_hash(p)
    write_file(tmp_path, "a.py", "def foo(): return 1\n")
    h2 = indexer.file_hash(p)
    assert h1 != h2


def test_default_db_path_scopes_by_model():
    root = __import__("pathlib").Path("/proj")
    p1 = indexer.default_db_path(root, "intfloat/multilingual-e5-small")
    p2 = indexer.default_db_path(root, "BAAI/bge-m3")
    assert p1 != p2
    assert p1.name == "multilingual-e5-small.db"
    assert p2.name == "bge-m3.db"


@pytest.mark.slow
class TestIndexProjectRealModel:
    def _open(self, tmp_path):
        db_path = indexer.default_db_path(tmp_path, model_config.MODEL_NAME)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_schema.open_db(db_path, model_config.MODEL_NAME, tmp_path)

    def test_index_project_chunks_and_embeds(self, tmp_path, write_file):
        write_file(tmp_path, "app/main.py", "def handler():\n    return 1\n")
        conn = self._open(tmp_path)
        stats = indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)

        assert stats["changed_files"] == 1
        assert stats["unchanged_files"] == 0
        assert stats["new_chunks"] == 1
        assert stats["parse_errors"] == []

        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        vector_count = conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]
        assert chunk_count == 1
        assert vector_count == 1

    def test_index_project_skips_unchanged_files_on_rerun(self, tmp_path, write_file):
        write_file(tmp_path, "app/main.py", "def handler():\n    return 1\n")
        conn = self._open(tmp_path)
        indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)

        stats2 = indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)
        assert stats2["changed_files"] == 0
        assert stats2["unchanged_files"] == 1
        assert stats2["new_chunks"] == 0

    def test_index_project_reindexes_changed_file(self, tmp_path, write_file):
        write_file(tmp_path, "app/main.py", "def handler():\n    return 1\n")
        conn = self._open(tmp_path)
        indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)

        write_file(tmp_path, "app/main.py", "def handler():\n    return 2\n\n\ndef other():\n    return 3\n")
        stats2 = indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)
        assert stats2["changed_files"] == 1
        assert stats2["new_chunks"] == 2

        names = {row[0] for row in conn.execute("SELECT name FROM chunks")}
        assert names == {"handler", "other"}

    def test_index_project_removes_chunks_for_deleted_file(self, tmp_path, write_file):
        p = write_file(tmp_path, "app/main.py", "def handler():\n    return 1\n")
        conn = self._open(tmp_path)
        stats1 = indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)
        assert stats1["new_chunks"] == 1

        p.unlink()
        stats2 = indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)
        assert stats2["deleted_files"] == 1
        assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0] == 0

    def test_index_project_records_import_edges(self, tmp_path, write_file):
        write_file(tmp_path, "pkg/__init__.py", "")
        write_file(tmp_path, "pkg/helpers.py", "def foo():\n    pass\n")
        write_file(tmp_path, "main.py", "import pkg.helpers\n\ndef entry():\n    pass\n")
        conn = self._open(tmp_path)
        indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)

        rows = conn.execute("SELECT source_file, target_file FROM imports").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == str(tmp_path / "pkg" / "helpers.py")

    def test_index_project_continues_after_parse_error(self, tmp_path, write_file, monkeypatch):
        # tree-sitter itself is error-tolerant (recovers via ERROR nodes rather than raising),
        # so a real parse failure has to be forced to exercise index_project's own
        # try/except-around-chunk_file path - what matters here is that one bad file doesn't
        # abort chunking of the rest (the exact bug process_class's None-body guard exists to
        # prevent, one level up: at the whole-file granularity instead of one class).
        write_file(tmp_path, "bad.py", "def broken(): pass\n")
        write_file(tmp_path, "good.py", "def good(): pass\n")

        real_chunk_file = indexer.chunk_file

        def fake_chunk_file(path):
            if path.name == "bad.py":
                raise RuntimeError("simulated parse failure")
            return real_chunk_file(path)

        monkeypatch.setattr(indexer, "chunk_file", fake_chunk_file)

        conn = self._open(tmp_path)
        stats = indexer.index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)

        assert len(stats["parse_errors"]) == 1
        assert "bad.py" in stats["parse_errors"][0]
        names = {row[0] for row in conn.execute("SELECT name FROM chunks")}
        assert names == {"good"}

    def test_index_json_translations_creates_literal_and_vector_entries(self, tmp_path, write_file):
        write_file(tmp_path, "langs/ru.json", '{"greeting": "Привет", "farewell": "Пока"}')
        conn = self._open(tmp_path)
        stats = indexer.index_json_translations(conn, tmp_path / "langs" / "ru.json", tmp_path,
                                                  model_name=model_config.MODEL_NAME)
        assert stats["changed"] is True
        assert stats["new_entries"] == 2
        assert stats["new_chunks"] >= 1

        entries = dict(conn.execute("SELECT key_path, value FROM json_entries").fetchall())
        assert entries == {"greeting": "Привет", "farewell": "Пока"}

    def test_index_json_translations_skips_unchanged(self, tmp_path, write_file):
        write_file(tmp_path, "langs/ru.json", '{"a": "b"}')
        conn = self._open(tmp_path)
        indexer.index_json_translations(conn, tmp_path / "langs" / "ru.json", tmp_path,
                                         model_name=model_config.MODEL_NAME)
        stats2 = indexer.index_json_translations(conn, tmp_path / "langs" / "ru.json", tmp_path,
                                                   model_name=model_config.MODEL_NAME)
        assert stats2["changed"] is False
