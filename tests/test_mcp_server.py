from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def _call_tool(project_root: Path, tool_name: str, arguments: dict, env_extra: dict | None = None) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {"LAYERGREP_ROOT": str(project_root)}
    if env_extra:
        env.update(env_extra)
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_server.py")],
        env=env,
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool(tool_name, arguments)
        return "".join(part.text for part in result.content if hasattr(part, "text"))


async def _run_layergrep(project_root: Path, query: str) -> str:
    return await _call_tool(project_root, "layergrep", {"query": query})


@pytest.mark.slow
def test_layergrep_over_real_mcp_protocol(tmp_path, write_file):
    write_file(tmp_path, "webapp/api/handlers.py",
               "def update_task_status(user_id, status):\n"
               "    '''Set the operator status for a user.'''\n"
               "    return status\n")

    import model_config
    from db_schema import open_db
    from indexer import default_db_path, index_project

    db_path = default_db_path(tmp_path, model_config.MODEL_NAME)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = open_db(db_path, model_config.MODEL_NAME, tmp_path)
    index_project(conn, tmp_path, model_name=model_config.MODEL_NAME, project_root=tmp_path)
    conn.close()

    output = asyncio.run(asyncio.wait_for(
        _run_layergrep(tmp_path, "update_task_status"), timeout=60
    ))
    assert "update_task_status" in output


# jinaai/jina-embeddings-v2-base-code: registered in model_config.MODEL_PREFIXES/EMBEDDING_DIMS
# (so db_schema.open_db's get_embedding_dim() finds its dimension in the dict without needing to
# load the model at all - see model_config.get_embedding_dim) but not downloaded in this
# environment - the realistic "fresh install, default model not cached yet" case, without
# actually needing network access to prove the server doesn't crash. A genuinely unregistered
# model name would instead crash inside open_db() itself (get_embedding_dim falling back to
# load_model for an unknown dimension), which is a different, less interesting failure mode
# than the one being tested here.
_UNCACHED_MODEL = "jinaai/jina-embeddings-v2-base-code"


@pytest.mark.slow
def test_layergrep_reports_missing_model_instead_of_crashing(tmp_path, write_file):
    write_file(tmp_path, "handlers.py", "def foo():\n    return 1\n")

    output = asyncio.run(asyncio.wait_for(
        _call_tool(tmp_path, "layergrep", {"query": "foo"}, {"LAYERGREP_MODEL": _UNCACHED_MODEL}),
        timeout=60,
    ))
    assert "install_model" in output
    assert _UNCACHED_MODEL in output


@pytest.mark.slow
def test_index_codebase_reports_missing_model_and_rolls_back(tmp_path, write_file):
    write_file(tmp_path, "handlers.py", "def foo():\n    return 1\n")

    output = asyncio.run(asyncio.wait_for(
        _call_tool(tmp_path, "index_codebase", {}, {"LAYERGREP_MODEL": _UNCACHED_MODEL}),
        timeout=60,
    ))
    assert "install_model" in output

    # Verify the rollback actually happened, not just that a friendly message was returned:
    # index_project() writes to `files` *before* the embed step that needs the model (see
    # indexer.py), all in one transaction on the server's long-lived connection - without an
    # explicit rollback, handlers.py's new hash would be left committed with zero chunks ever
    # created for it, and a later successful index_codebase would wrongly treat it as
    # "unchanged" and skip it forever (see mcp_server.py's index_codebase for the full story).
    from db_schema import open_db
    from indexer import default_db_path

    db_path = default_db_path(tmp_path, _UNCACHED_MODEL)
    conn = open_db(db_path, _UNCACHED_MODEL, tmp_path)
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    conn.close()


@pytest.mark.slow
def test_install_model_reports_ready_for_already_cached_model(tmp_path, write_file):
    write_file(tmp_path, "handlers.py", "def foo():\n    return 1\n")

    # No LAYERGREP_MODEL override - uses model_config.MODEL_NAME, already cached from every
    # other test in this suite, so this proves the "already downloaded" path works without
    # actually needing network access to exercise a real first-time download.
    output = asyncio.run(asyncio.wait_for(
        _call_tool(tmp_path, "install_model", {}),
        timeout=60,
    ))
    assert "ready" in output.lower()
