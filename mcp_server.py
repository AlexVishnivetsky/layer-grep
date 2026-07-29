from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from mcp.server.mcpserver import MCPServer

import db_schema
import indexer
import model_config
import project_config as pconfig  # alias: avoids shadowing this module's own PROJECT_CONFIG
import search

# LAYERGREP_ROOT: project root - where .layergrep/ lives and the import graph's resolution
# base (a monorepo's sibling packages can be imported from here without living inside
# LAYERGREP_SUBDIR). Defaults to cwd, matching cli.py.
PROJECT_ROOT = Path(os.environ.get("LAYERGREP_ROOT", ".")).resolve()
INDEX_SUBDIR = os.environ.get("LAYERGREP_SUBDIR")  # e.g. "backend"; None = whole project_root
MODEL_NAME = os.environ.get("LAYERGREP_MODEL", model_config.MODEL_NAME)

INDEX_ROOT = (PROJECT_ROOT / INDEX_SUBDIR) if INDEX_SUBDIR else PROJECT_ROOT
DB_PATH = indexer.default_db_path(PROJECT_ROOT, MODEL_NAME)

# Loaded once at module level (not just inside index_codebase) so layergrep's tool
# description below can list this project's *actual* configured layers - the exact same
# mcp_server.py file gets pointed at different projects via LAYERGREP_ROOT/.mcp.json, each
# with its own .layergrep.json layer set - hardcoding one project's layer names into this
# shared file would silently mislead whichever LLM client reads this tool's description on
# every other project.
PROJECT_CONFIG = pconfig.load_project_config(PROJECT_ROOT)

# File, not stdout/stderr: stdout is the stdio MCP transport itself (writing to it would
# corrupt the protocol), and stderr only shows up via `claude --debug mcp` in the parent's
# terminal - a file survives process exit and lets two concurrently-spawned server
# instances (seen in practice: Claude Code launching this twice at startup) be told apart
# by PID in one place, after the fact.
LOG_PATH = DB_PATH.parent / "mcp_server.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Configured on the shared "layergrep" parent, not just this module's own logger:
# model_config.py and retrieval.py each log checkpoints (model_config.load_model's
# import/constructor timings, retrieval.search_by_layers' model/vector-query timings) under
# "layergrep.model_config"/"layergrep.retrieval", which propagate up to this handler by
# default - so a stuck call's sub-step can be pinned down from this one log file.
_root_logger = logging.getLogger("layergrep")
_root_logger.setLevel(logging.INFO)
_root_logger.propagate = False  # file only - avoid duplicating into whatever the root logger does
_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
_handler.setFormatter(logging.Formatter(
    f"%(asctime)s pid={os.getpid()} %(name)s %(levelname)s %(message)s"
))
_root_logger.addHandler(_handler)

logger = logging.getLogger("layergrep.mcp_server")

logger.info(f"server process starting: project_root={PROJECT_ROOT} index_root={INDEX_ROOT} "
            f"model={MODEL_NAME} db={DB_PATH} argv={sys.argv} executable={sys.executable}")

# Allowlist, not a full os.environ dump: this process inherits whatever Claude Code's own
# environment carries (which can include API keys/tokens - not something to ever write to a
# log file). Only variables relevant to diagnosing an environment-dependent hang (proxy
# config, offline flags, HF/torch cache locations).
_ENV_ALLOWLIST = [
    "PATH", "PYTHONPATH", "PYTHONHOME",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
    "HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TRANSFORMERS_OFFLINE",
    "TRANSFORMERS_CACHE", "SENTENCE_TRANSFORMERS_HOME",
    "SSL_CERT_FILE", "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE",
    "TEMP", "TMP", "USERPROFILE", "NUMBER_OF_PROCESSORS", "OS", "COMPUTERNAME",
    "LAYERGREP_ROOT", "LAYERGREP_SUBDIR", "LAYERGREP_MODEL",
]
_env_snapshot = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
logger.info(f"server process env (allowlisted): {_env_snapshot}")

# Warm the model cache here, before MCPServer spawns its own worker threads (mcp.run() below):
# on Windows, a Python import that triggers OpenBLAS's DLL load can deadlock via loader-lock
# contention if *other* threads already exist in the process and might touch the loader
# concurrently (OpenBLAS spawns its own worker threads from inside DllMain, a well-known
# OpenBLAS-on-Windows anti-pattern - DllMain must never create threads, because the loader
# lock is held during it) - exactly the situation once MCPServer's AnyIO worker-thread pool is
# running. Doing the heavy import here, before any of those threads exist, sidesteps the
# race entirely - by the time a real layergrep call runs model_config.load_model(), it's
# just an in-process cache hit (_MODEL_CACHE), not a fresh import.
logger.info("warming model cache before starting MCPServer...")
_t_warm = time.monotonic()
try:
    model_config.load_model(MODEL_NAME)
    logger.info(f"model cache warmed after {time.monotonic() - _t_warm:.3f}s")
except model_config.ModelNotCachedError:
    # Not fatal: a fresh install has no cached model yet. Start up anyway; layergrep/
    # index_codebase check for this same condition themselves and return a message pointing
    # at the install_model tool instead of raising.
    logger.warning(f"model {MODEL_NAME!r} isn't downloaded yet - starting anyway; "
                    f"layergrep/index_codebase will report this until install_model is called")

mcp = MCPServer("layergrep")

# Opened once for the process lifetime, not per call: open_db() reruns every migration/
# version check (CREATE TABLE IF NOT EXISTS x N, PRAGMA table_info, meta-table version
# comparisons) on every invocation, which is real, avoidable overhead paid on every single
# layergrep/index_codebase call for no reason - none of that state changes between calls
# in the same long-lived server process. Symmetric with the model warm-up above.
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
logger.info("opening persistent db connection...")
_t_db = time.monotonic()
_CONN = db_schema.open_db(DB_PATH, MODEL_NAME, PROJECT_ROOT)
logger.info(f"db connection opened after {time.monotonic() - _t_db:.3f}s")


def _connect():
    return _CONN


def _describe_configured_layers(cfg: pconfig.ProjectConfig) -> str:
    """Layer names for the docstring below - read from *this* project's PROJECT_CONFIG,
    not a fixed list, since different projects this same server gets pointed at can have
    a completely different layer set (or none at all)."""
    names = [name for name, _, _ in cfg.layer_rules]
    if cfg.default_layer not in names:
        names.append(cfg.default_layer)
    return ", ".join(names)


# Built once at import time (from PROJECT_CONFIG, loaded above) and passed via
# mcp.tool(description=...) rather than a plain docstring literal: MCPServer reads fn.__doc__
# at decoration time, which only ever sees a string that's a compile-time constant if written
# as a normal """docstring""" - it can't be an f-string. Computing it here lets the layer list
# reflect whatever project PROJECT_ROOT actually points at, instead of hardcoding one
# project's layer names into a file meant to be reused unmodified across projects.
_LAYERGREP_DESCRIPTION = f"""
Semantic + literal search for a feature's implementation from a natural-language
description (e.g. "handler that updates user status"). Returns markdown with
findings grouped by layer: {_describe_configured_layers(PROJECT_CONFIG)}. An empty
list for a layer is a valid result (the feature doesn't touch that layer, or nothing
was found there).
The whole project is indexed (not just one application) - within each layer,
findings are further grouped by module (the project's top-level directory the file
lives under, shown as "[module: <name>]") so a large module doesn't drown out
findings from a small one in the same layer.
Also includes exact (literal) matches on route/field/constant names, structural
links via the import graph (handler -> the model/config/constants files it
imports), and cross-layer links via shared string literals (a route path/field
name/constant that appears verbatim in chunks from OTHER layers - this also
works across languages, where the import graph plainly doesn't apply, e.g.
JS frontend -> Python backend). Each per-layer search hit includes a code
snippet (up to 15 lines/800 chars) - vector distance alone doesn't guarantee
relevance, so judge by the actual code, not just the distance number.

A codebase has two potentially different languages: its code language (the
language identifiers/names read as - usually English by convention, regardless
of project) and its comment language (whatever language its docstrings/comments
are actually written in - varies by project). Phrase the query in both, if they
differ (e.g. "handler that updates user status. хендлер, который обновляет
статус пользователя." illustrates pairing English with Russian, a comment
language this particular project happens to use in places - substitute whatever
language this project's own comments are actually in). A multilingual embedding
model has a measurable bias toward text in the query's own language - even an
exact translation of the same sentence scores noticeably worse than text in
the query's language - and a chunk's most informative text may be in either
language (a native-language docstring, or
just an English identifier with no docstring at all) - a query combining both
languages empirically narrows this gap a lot without hurting the match on
either one.
"""


_MODEL_NOT_READY_MESSAGE = (
    f'Model "{MODEL_NAME}" isn\'t downloaded yet on this machine. Call the install_model '
    f"tool first (one-time, needs network access), then retry."
)


@mcp.tool(description=_LAYERGREP_DESCRIPTION)
def layergrep(query: str) -> str:
    """See _LAYERGREP_DESCRIPTION above - passed via mcp.tool(description=...) instead of
    this docstring so the layer list can be computed from PROJECT_CONFIG (see its comment)."""
    logger.info(f"layergrep start query={query!r}")
    t0 = time.monotonic()
    try:
        conn = _connect()
        result = search.search_hybrid(conn, query, PROJECT_ROOT, model_name=MODEL_NAME)
        formatted = search.format_results(query, result)
    except model_config.ModelNotCachedError:
        # Returned as a normal result, not raised: this is routine first-run guidance (see
        # install_model below), not a bug - a raised exception would surface to the calling
        # LLM as a tool error rather than a plain instruction to follow.
        logger.info(f"layergrep query={query!r}: model not cached, pointing at install_model")
        return _MODEL_NOT_READY_MESSAGE
    except Exception:
        logger.exception(f"layergrep FAILED query={query!r} after {time.monotonic() - t0:.3f}s")
        raise
    logger.info(f"layergrep done query={query!r} took={time.monotonic() - t0:.3f}s "
                f"literal_hits={len(result['literal'])} json_literal_hits={len(result['json_literal'])}")
    return formatted


@mcp.tool()
def index_codebase(force: bool = False) -> str:
    """
    Incrementally (re)build the project index: chunking, embeddings, import
    graph, JSON translations (paths from this project's .layergrep.json
    translations config, if any). Skips files whose content hasn't changed
    since last time (by hash). force=True forces a full reindex, ignoring the
    hash cache (needed e.g. after a chunker logic change).
    layergrep never reindexes on its own - call this explicitly after code changes.
    """
    logger.info(f"index_codebase start force={force}")
    t0 = time.monotonic()
    conn = _connect()
    try:
        stats = indexer.index_project(conn, INDEX_ROOT, model_name=MODEL_NAME,
                                       project_root=PROJECT_ROOT, force=force)

        lines = [
            (f"code: changed={stats['changed_files']} unchanged={stats['unchanged_files']} "
             f"deleted={stats['deleted_files']} new_chunks={stats['new_chunks']}"),
        ]
        if stats["parse_errors"]:
            lines.append(f"parse errors: {len(stats['parse_errors'])} (first: {stats['parse_errors'][0]})")

        for rel_path in PROJECT_CONFIG.translations_files:
            translations_file = PROJECT_ROOT / rel_path
            if translations_file.is_file():
                json_stats = indexer.index_json_translations(conn, translations_file, PROJECT_ROOT,
                                                               model_name=MODEL_NAME, force=force)
                lines.append(f"{rel_path}: changed={json_stats['changed']} "
                              f"entries={json_stats['new_entries']} chunks={json_stats['new_chunks']}")
    except model_config.ModelNotCachedError:
        # index_project() writes files/imports rows for the changed set *before* the embed
        # step that actually needs the model (see indexer.py), all in the same uncommitted
        # transaction on this long-lived connection - roll back explicitly here, since (unlike
        # a genuine crash) the process keeps running and would otherwise carry an open
        # transaction into the next call, where those files would wrongly look "unchanged"
        # (their new hash already written, uncommitted) and get silently skipped with no
        # chunks ever created for them.
        conn.rollback()
        logger.info(f"index_codebase force={force}: model not cached, pointing at install_model "
                    f"(rolled back)")
        return _MODEL_NOT_READY_MESSAGE
    except Exception:
        logger.exception(f"index_codebase FAILED force={force} after {time.monotonic() - t0:.3f}s")
        raise

    logger.info(f"index_codebase done force={force} took={time.monotonic() - t0:.3f}s: "
                + " | ".join(lines))
    return "\n".join(lines)


@mcp.tool()
def install_model() -> str:
    """
    Downloads and caches this server's configured embedding model (this project's
    LAYERGREP_MODEL, or the tool's own built-in default) from HuggingFace. One-time
    setup step for a fresh install with no cached models yet - call this if
    layergrep/index_codebase report the model isn't downloaded. Needs network
    access for the download; safe to call again afterwards (just confirms it's ready).
    Only ever installs this one already-configured model, not an arbitrary one - changing
    which model a project uses is a deliberate reindexing decision (a different model
    needs its own separate index), not something to pick mid-conversation.
    """
    logger.info(f"install_model start model={MODEL_NAME!r}")
    t0 = time.monotonic()
    try:
        model_config.load_model(MODEL_NAME, allow_download=True)
    except Exception:
        logger.exception(f"install_model FAILED model={MODEL_NAME!r} after {time.monotonic() - t0:.3f}s")
        raise
    logger.info(f"install_model done model={MODEL_NAME!r} took={time.monotonic() - t0:.3f}s")
    return f'Model "{MODEL_NAME}" is downloaded and ready ({time.monotonic() - t0:.1f}s).'


def main() -> None:
    """Entry point for the `layergrep-mcp-server` console script (see pyproject.toml) as well
    as `python mcp_server.py` directly - everything before this point (project root/model/db
    setup, @mcp.tool() registration) already ran at import time either way, so this just needs
    to start the actual server loop."""
    import atexit

    atexit.register(lambda: logger.info("server process exiting"))
    try:
        mcp.run()
    except BaseException:
        logger.exception("server process crashed out of mcp.run()")
        raise


if __name__ == "__main__":
    main()
