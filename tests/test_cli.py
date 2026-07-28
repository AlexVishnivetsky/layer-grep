from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CLI = str(Path(__file__).resolve().parent.parent / "cli.py")


def run_cli(*args, cwd=None):
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True, cwd=cwd, timeout=60,
                           check=False)


def test_no_args_prints_usage_and_exits_nonzero():
    result = run_cli()
    assert result.returncode != 0
    assert "usage:" in result.stderr


def test_help_flag_prints_usage():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_init_config_writes_draft(tmp_path, write_file):
    write_file(tmp_path, "api/handlers.py", "def foo(): pass\n")
    result = run_cli("init-config", "--root", str(tmp_path))
    assert result.returncode == 0
    assert (tmp_path / ".layergrep.json").is_file()
    assert "backend/api" in result.stdout


def test_init_config_does_not_overwrite_existing(tmp_path, write_file):
    write_file(tmp_path, ".layergrep.json", '{"layers": [{"name": "custom", "dirs": [], "files": []}]}')
    result = run_cli("init-config", "--root", str(tmp_path))
    assert result.returncode == 0
    assert "already exists - not overwritten" in result.stderr
    import json
    on_disk = json.loads((tmp_path / ".layergrep.json").read_text(encoding="utf-8"))
    assert on_disk["layers"][0]["name"] == "custom"


def test_search_without_index_fails_clearly(tmp_path):
    result = run_cli("some query", "--root", str(tmp_path))
    assert result.returncode != 0
    assert "run `layergrep index` first" in result.stderr


@pytest.mark.slow
def test_index_then_search_round_trip(tmp_path, write_file):
    write_file(tmp_path, "webapp/api/handlers.py",
               "def update_task_status(user_id):\n"
               "    '''Set operator status for a user.'''\n"
               "    return user_id\n")

    index_result = run_cli("index", "--root", str(tmp_path))
    assert index_result.returncode == 0
    assert (tmp_path / ".layergrep" / "multilingual-e5-small.db").is_file()

    search_result = run_cli("update_task_status", "--root", str(tmp_path))
    assert search_result.returncode == 0
    assert "update_task_status" in search_result.stdout


@pytest.mark.slow
def test_double_dash_escapes_the_word_index_as_a_query(tmp_path, write_file):
    write_file(tmp_path, "app.py", "def index():\n    '''Renders the index page.'''\n    return None\n")
    run_cli("index", "--root", str(tmp_path))

    result = run_cli("--", "index", "--root", str(tmp_path))
    assert result.returncode == 0
    assert "literal matches for 'index'" in result.stdout


def test_calibrate_thresholds_without_index_fails_clearly(tmp_path):
    result = run_cli("calibrate-thresholds", "--root", str(tmp_path))
    assert result.returncode != 0
    assert "run `layergrep index` first" in result.stderr


@pytest.mark.slow
def test_calibrate_thresholds_reports_current_values_and_distribution(tmp_path, write_file):
    write_file(tmp_path, "a.py", "def f():\n    x = 'rare_literal_xyz'\n    return x\n")
    run_cli("index", "--root", str(tmp_path))

    result = run_cli("calibrate-thresholds", "--root", str(tmp_path))
    assert result.returncode == 0
    assert "currently configured: 30" in result.stdout
    assert "currently configured: 15" in result.stdout


@pytest.mark.slow
def test_install_model_reports_ready_for_already_cached_model():
    # No --model override - uses model_config.MODEL_NAME, already cached from every other
    # slow test in this suite, so this proves the "already downloaded" path works without
    # actually needing network access to exercise a real first-time download (mirrors
    # test_install_model_reports_ready_for_already_cached_model in test_mcp_server.py).
    result = run_cli("install-model")
    assert result.returncode == 0
    assert "is downloaded and ready" in result.stdout
