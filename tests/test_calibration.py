from __future__ import annotations

import pytest

import calibration
import db_schema
import model_config


def _make_conn(tmp_path):
    return db_schema.open_db(tmp_path / "t.db", model_config.MODEL_NAME, tmp_path)


def test_literal_match_counts_distinguishes_rare_from_common(tmp_path):
    conn = _make_conn(tmp_path)
    conn.execute(
        "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text) "
        "VALUES ('a.py', 1, 1, 'function', 'f', \"route = 'rare_literal_xyz'\")"
    )
    for i in range(10):
        conn.execute(
            "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text) "
            "VALUES (?, 1, 1, 'function', ?, \"kind = 'common_literal'\")",
            (f"b{i}.py", f"f{i}"),
        )
    conn.commit()

    counts = dict(calibration.literal_match_counts(conn))
    assert counts["rare_literal_xyz"] == 1
    assert counts["common_literal"] == 10


def test_literal_match_counts_respects_sample_cap(tmp_path):
    conn = _make_conn(tmp_path)
    for i in range(20):
        conn.execute(
            "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text) "
            "VALUES (?, 1, 1, 'function', ?, ?)",
            (f"c{i}.py", f"f{i}", f"kind = 'unique_literal_{i}'"),
        )
    conn.commit()

    counts = calibration.literal_match_counts(conn, sample=5)
    assert len(counts) <= 5


def test_import_target_counts_groups_by_distinct_source(tmp_path):
    conn = _make_conn(tmp_path)
    for src in ("a.py", "b.py", "c.py"):
        conn.execute(
            "INSERT INTO imports (source_file, target_file, module) VALUES (?, 'widely_shared.py', 'x')",
            (src,),
        )
    conn.execute(
        "INSERT INTO imports (source_file, target_file, module) VALUES ('a.py', 'narrow.py', 'x')"
    )
    conn.commit()

    counts = dict(calibration.import_target_counts(conn))
    assert counts["widely_shared.py"] == 3
    assert counts["narrow.py"] == 1


def test_percentile_basic():
    values = list(range(1, 101))  # 1..100, ascending
    assert calibration.percentile(values, 50) == 51  # nearest-rank, not interpolated
    assert calibration.percentile(values, 100) == 100
    assert calibration.percentile(values, 0) == 1


def test_percentile_empty_list():
    assert calibration.percentile([], 50) == 0


def test_fraction_at_or_below():
    values = sorted([1, 1, 1, 2, 2, 50, 60])
    # 5 of 7 values are <= 30
    assert calibration.fraction_at_or_below(values, 30) == pytest.approx(5 / 7)
    assert calibration.fraction_at_or_below(values, 0) == 0.0
    assert calibration.fraction_at_or_below(values, 1000) == 1.0


def test_fraction_at_or_below_empty_list():
    assert calibration.fraction_at_or_below([], 30) == 0.0


def test_format_report_includes_current_values_and_percentiles():
    report = calibration.format_report(
        literal_counts=[("rare", 2), ("common", 500)],
        import_counts=[("narrow.py", 1), ("shared.py", 100)],
        current_literal_threshold=30,
        current_import_threshold=15,
    )
    assert "currently configured: 30" in report
    assert "currently configured: 15" in report
    assert "'rare'" in report
    assert "shared.py" in report
    assert "percentiles:" in report
    assert "keeps" in report and "filters" in report


def test_format_report_handles_empty_corpus():
    report = calibration.format_report([], [], 30, 15)
    assert "nothing to calibrate" in report
