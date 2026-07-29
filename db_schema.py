from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path

import sqlite_vec

import model_config

# alias: this module's own `project_config` local variable (the *loaded* ProjectConfig
# instance) would otherwise shadow the module name within the same scope
import project_config as pconfig
from chunker import CHUNKER_VERSION
from imports import IMPORT_VERSION_GROUPS, IMPORTS_VERSIONS
from json_index import JSON_INDEX_VERSION

logger = logging.getLogger("layergrep.db_schema")


class ModelMismatchError(Exception):
    pass


def _apply_version_migration(
    conn: sqlite3.Connection,
    meta_key: str,
    current_version: int,
    on_mismatch: Callable[[sqlite3.Connection], int],
) -> bool:
    """Shared shape for the "version stamped in meta, wipe-and-reprocess-on-mismatch"
    pattern used by CHUNKER_VERSION/IMPORTS_VERSIONS (once per group)/JSON_INDEX_VERSION. Factoring
    "read meta -> compare -> wipe callback -> re-stamp -> commit" into one place means a
    future 4th versioned feature only has to write its own wipe logic, not duplicate the
    bookkeeping around it.

    `on_mismatch` does whatever table-specific wiping/reprocessing is needed on a version
    mismatch, returning how many already-indexed rows/paths it actually deleted; it must not
    commit itself - this function commits once afterward, matching every original migration
    block's own trailing conn.commit().

    Returns True only if `on_mismatch` reported actually deleting something real - not just
    whether the stamped version differed (issue #37): a brand-new project, or a brand-new
    extension group added to an existing project with zero files of that type yet, has
    `row is None` too, but there's nothing to "re"-index in either case. The wipe *count*
    (not the raw version-mismatch fact) is what should drive a "you should reindex" notice."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (meta_key,)).fetchone()
    version_str = str(current_version)
    wiped = 0
    if row is None or row[0] != version_str:
        wiped = on_mismatch(conn)
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (meta_key, version_str),
        )
    conn.commit()
    return wiped > 0


def _wipe_for_chunker_version(conn: sqlite3.Connection) -> int:
    # covers both a version bump and a pre-versioning db (row is None but may already
    # hold stale chunks) - wiping an empty table on a genuinely fresh db is a no-op
    affected = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    conn.execute("DELETE FROM chunk_vectors")
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM files")
    return affected


def _make_wipe_for_imports_version_group(extensions: frozenset[str]) -> Callable[[sqlite3.Connection], int]:
    """Builds a wipe callback scoped to just one import-version group's own extensions
    (issue #35) - a version bump in one group (say, Rust) must not touch another group's
    (say, Python's) already-correct `imports` rows, which is why this deletes per-path
    rather than the old single-group code's unconditional `DELETE FROM imports`."""
    def wipe(conn: sqlite3.Connection) -> int:
        # every file of this group's extensions needs reprocessing to (re)populate imports,
        # not just ones already in the `imports` table - on the very first introduction of
        # import-graph tracking that table starts empty, so scoping this to "already has
        # import rows" would never force anything and silently leave the graph empty forever.
        affected_paths = [
            r[0] for r in conn.execute("SELECT path FROM files")
            if Path(r[0]).suffix.lower() in extensions
        ]
        for p in affected_paths:
            conn.execute("DELETE FROM imports WHERE source_file = ?", (p,))
            conn.execute("DELETE FROM files WHERE path = ?", (p,))
        return len(affected_paths)
    return wipe


def _wipe_for_json_index_version(conn: sqlite3.Connection) -> int:
    # force these files to be treated as new next time index_json_translations() runs -
    # capture affected paths before wiping json_entries itself
    stale_json_paths = [r[0] for r in conn.execute("SELECT DISTINCT file_path FROM json_entries")]
    json_chunk_paths = [r[0] for r in conn.execute(
        "SELECT DISTINCT file_path FROM chunks WHERE chunk_source_kind = 'json_key'"
    )]
    for p in json_chunk_paths:
        if p not in stale_json_paths:
            stale_json_paths.append(p)

    conn.execute("DELETE FROM json_entries")
    json_chunk_ids = [r[0] for r in conn.execute(
        "SELECT id FROM chunks WHERE chunk_source_kind = 'json_key'"
    )]
    if json_chunk_ids:
        conn.executemany("DELETE FROM chunk_vectors WHERE chunk_id = ?", [(i,) for i in json_chunk_ids])
        conn.execute("DELETE FROM chunks WHERE chunk_source_kind = 'json_key'")
    for p in stale_json_paths:
        conn.execute("DELETE FROM files WHERE path = ?", (p,))
    return len(stale_json_paths)


_PENDING_REINDEX_NOTICE_KEY = "pending_reindex_notice"


def get_pending_reindex_notice(conn: sqlite3.Connection) -> str | None:
    """Set by open_db() (issue #37) whenever a version-triggered wipe actually deleted
    already-indexed rows - callers that open the db without immediately reindexing (a search,
    calibrate-thresholds) should surface this rather than silently return results against a
    now-incomplete index. Cleared by clear_pending_reindex_notice() once a real reindex runs."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (_PENDING_REINDEX_NOTICE_KEY,)).fetchone()
    return row[0] if row is not None else None


def clear_pending_reindex_notice(conn: sqlite3.Connection) -> None:
    """Called by callers (cli.py's _run_index, mcp_server.py's index_codebase) right after a
    successful indexer.index_project() run - not by indexer.py itself, which stays agnostic
    of this notice concept entirely."""
    conn.execute("DELETE FROM meta WHERE key = ?", (_PENDING_REINDEX_NOTICE_KEY,))
    conn.commit()


def open_db(db_path: Path, model_name: str, project_root: Path) -> sqlite3.Connection:
    # check_same_thread=False: mcp_server.py holds one connection open for the lifetime of
    # the process (see its module-level _CONN) rather than reopening per call - defensive
    # against MCPServer dispatching a tool call on a different thread than the one that
    # opened it.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    row = conn.execute("SELECT value FROM meta WHERE key = 'embedding_model'").fetchone()
    if row is None:
        conn.execute("INSERT INTO meta (key, value) VALUES ('embedding_model', ?)", (model_name,))
        conn.commit()
    elif row[0] != model_name:
        raise ModelMismatchError(
            f"{db_path} was built with model {row[0]!r}, not {model_name!r}. "
            f"Vectors from different models aren't comparable/interchangeable - "
            f"use a separate .db file per model instead of mixing them."
        )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            node_type TEXT NOT NULL,
            name TEXT NOT NULL,
            text TEXT NOT NULL,
            layer TEXT NOT NULL DEFAULT '',
            chunk_source_kind TEXT NOT NULL DEFAULT '',
            module TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path)")

    # `chunks.module` (top-level dir name relative to project_root, e.g. "inventory") is an
    # unrelated concept from `imports.module` below (a Python import's dotted name, e.g.
    # "inventory.api.account.helpers") - same column name, different table, don't confuse the
    # two when reading this schema.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
    if "layer" not in existing_columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN layer TEXT NOT NULL DEFAULT ''")
    if "chunk_source_kind" not in existing_columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN chunk_source_kind TEXT NOT NULL DEFAULT ''")
    if "module" not in existing_columns:
        conn.execute("ALTER TABLE chunks ADD COLUMN module TEXT NOT NULL DEFAULT ''")

    # Both are pure functions of file_path (+ project_root for module, + the project config
    # for layer) - recomputed for every distinct file_path on every open_db() call, not just
    # once as a one-time migration. Cheap (grouped by distinct path, not per-chunk) and, now
    # that open_db() is only called once per process lifetime (persistent connection - see
    # mcp_server.py), this only costs at server startup, never per search. This also means
    # editing .layergrep.json's layer rules takes effect on next server restart without
    # needing a full force-reindex.
    project_config = pconfig.load_project_config(project_root)
    distinct_paths = [row[0] for row in conn.execute("SELECT DISTINCT file_path FROM chunks")]
    distinct_modules: set[str] = set()
    for file_path in distinct_paths:
        module = pconfig.classify_module(
            Path(file_path), project_root, project_config.module_depth, project_config.module_rules,
        )
        distinct_modules.add(module)
        conn.execute(
            "UPDATE chunks SET layer = ?, module = ? WHERE file_path = ?",
            (pconfig.classify_layer(Path(file_path), project_config), module, file_path),
        )
    if distinct_paths and len(distinct_modules) <= 1:
        # Mirrors the "layer rules didn't match" warning below, one dimension over: every
        # indexed file collapsed to the same module value (or none), so search_by_layers'
        # per-module grouping (see retrieval.py) has nothing to actually group by - degrades
        # silently to "one bucket per layer" rather than failing, but worth flagging since a
        # monorepo whose packages sit deeper than module_depth accounts for is an easy miss.
        logger.warning(
            f"open_db: module dimension isn't discriminating - all {len(distinct_paths)} indexed "
            f"files classified as module {next(iter(distinct_modules), '')!r} at "
            f"module_depth={project_config.module_depth} ({project_config.source}). If this is a "
            f"monorepo/workspace layout with packages nested deeper than the project root, try "
            f"raising \"module_depth\" in .layergrep.json."
        )
    if not project_config.layer_rules:
        logger.info(f"open_db: project_config has no layer rules ({project_config.source}) - "
                    f"every chunk will fall through to default_layer={project_config.default_layer!r}")
    elif distinct_paths:
        # Rules exist but never matched anything real for this project - e.g. a
        # .layergrep.json copy-pasted from a different repo whose directory names don't
        # exist here. Not fatal (default_layer degradation is still a valid, honest result),
        # but worth flagging since it's easy to miss silently.
        non_default = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE layer != ?", (project_config.default_layer,)
        ).fetchone()[0]
        if non_default == 0:
            logger.warning(
                f"open_db: {len(project_config.layer_rules)} layer rule(s) configured "
                f"({project_config.source}) but NONE matched any of {len(distinct_paths)} indexed "
                f"files - every chunk fell through to default_layer={project_config.default_layer!r}. "
                f"Check that the configured dirs/files actually exist under this project root."
            )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_layer ON chunks(layer)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_module ON chunks(module)")
    conn.commit()

    # import graph (Python only for now) - "file -> what it imports", resolved to real
    # project files by imports.extract_python_imports. Used to expand a search hit to the
    # model/config/constants it actually imports, a structural link independent of vector
    # similarity between the handler's and the model's embeddings.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS imports (
            id INTEGER PRIMARY KEY,
            source_file TEXT NOT NULL,
            target_file TEXT NOT NULL,
            module TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_imports_source ON imports(source_file)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_imports_target ON imports(target_file)")

    # Collected across every version migration below (issue #37) - human-readable scope
    # names for whichever ones actually wiped real, already-indexed rows, surfaced as a
    # single pending-reindex notice at the end rather than left as a silent side effect.
    stale_scopes: list[str] = []

    # One migration per group (not one global one) - see IMPORT_VERSION_GROUPS/
    # IMPORTS_VERSIONS in imports.py for why: a version bump in one language's resolver
    # must not force reprocessing of every other language's already-correct import edges.
    for group, version in IMPORTS_VERSIONS.items():
        if _apply_version_migration(
            conn, f"imports_version_{group}", version,
            _make_wipe_for_imports_version_group(IMPORT_VERSION_GROUPS[group]),
        ):
            stale_scopes.append(f"imports:{group}")

    # literal lookup side-table for JSON translations (json_index.literal_entries) - exact/
    # substring search, deliberately outside chunk_vectors: a short translation string gives
    # a weak, barely-distinguishable embedding, so this is meant for the literal-grep layer,
    # not vector search
    conn.execute("""
        CREATE TABLE IF NOT EXISTS json_entries (
            id INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            key_path TEXT NOT NULL,
            value TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_json_entries_file_path ON json_entries(file_path)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_json_entries_key_path ON json_entries(key_path)")

    if _apply_version_migration(conn, "json_index_version", JSON_INDEX_VERSION, _wipe_for_json_index_version):
        stale_scopes.append("json_index")

    dim = model_config.get_embedding_dim(model_name)
    try:
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dim}] distance_metric=cosine
            )
        """)
    except sqlite3.OperationalError:
        # older sqlite-vec without distance_metric - fine, we store unit vectors so
        # L2 ordering is equivalent to cosine ordering
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dim}]
            )
        """)
    conn.commit()

    # file-content hashes alone can't tell that chunker.py's logic changed for unchanged
    # source files - track the chunker version separately and force a full reindex on mismatch
    if _apply_version_migration(conn, "chunker_version", CHUNKER_VERSION, _wipe_for_chunker_version):
        stale_scopes.append("chunker")

    if stale_scopes:
        logger.warning(f"open_db: layergrep code changed since this project was last indexed "
                        f"({', '.join(stale_scopes)}) - some results may be stale until a reindex runs")
        notice = (
            f"layergrep code changed since this project was last indexed ({', '.join(stale_scopes)}) - "
            f"some search results may be incomplete until you run `layergrep index` (CLI) or "
            f"call index_codebase (MCP)."
        )
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_PENDING_REINDEX_NOTICE_KEY, notice),
        )
        conn.commit()
    return conn
