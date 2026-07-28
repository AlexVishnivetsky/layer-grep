from __future__ import annotations

import bisect
import sqlite3

import retrieval
from search import _extract_chunk_literals

# literal_noise_threshold/import_noise_threshold are hand-picked per project. Judging whether
# a given value fits a specific project requires seeing its actual match-count distribution,
# which isn't practical without database access and SQL - this module reproduces that data.
# Deliberately reports percentiles rather than searching for "the widest gap" and picking a
# number automatically: real corpora are Zipfian (a huge pile of count=1 literals, then a long
# thinning tail up to a few very common ones), so there usually isn't a clean two-cluster
# split to find - only a continuum. A percentile view is the honest equivalent of "eyeballing
# where the bulk of the distribution sits", left for a human to judge.


def literal_match_counts(conn: sqlite3.Connection, sample: int = 500) -> list[tuple[str, int]]:
    """(literal, total_matches) for up to `sample` distinct literals actually extracted from
    the indexed corpus's own chunks (not synthetic/guessed strings) - using the exact same
    _extract_chunk_literals() + name/text LIKE match query that cross_layer_literal_links()
    runs at search time, so this reflects the real distribution a literal_noise_threshold
    would filter, not an approximation. Sampling (not every literal in a large corpus) keeps
    this a one-off manual calibration tool, not something with production-time cost - each
    entry costs one full-table LIKE scan (no index can help a leading-wildcard '%x%' query),
    so thousands of literals against a large corpus would be slow for no added insight beyond
    a few hundred: the whole point is reading off the shape of the distribution, not an exact
    census of every literal ever chunked."""
    literals: set[str] = set()
    for (text,) in conn.execute("SELECT text FROM chunks"):
        literals.update(_extract_chunk_literals(text, limit=8))
        if len(literals) >= sample:
            break

    counts: list[tuple[str, int]] = []
    for literal in literals:
        like = f"%{retrieval.escape_like(literal)}%"
        total = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE name LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\'",
            (like, like),
        ).fetchone()[0]
        counts.append((literal, total))
    counts.sort(key=lambda pair: -pair[1])
    return counts


def import_target_counts(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """(target_file, distinct_importer_count) for every real import edge in the corpus - a
    single cheap aggregate query (unlike literal_match_counts, no sampling needed: this is the
    same metric expand_via_imports() already computes per-target at search time, just grouped
    once up front instead of once per seed file)."""
    return conn.execute(
        "SELECT target_file, COUNT(DISTINCT source_file) AS n FROM imports "
        "GROUP BY target_file ORDER BY n DESC"
    ).fetchall()


def percentile(ascending_values: list[int], p: float) -> int:
    """Value at percentile `p` (0-100) of an already-ascending-sorted list - nearest-rank,
    not interpolated (simple and good enough for a rough calibration read, not a statistics
    tool)."""
    if not ascending_values:
        return 0
    idx = min(len(ascending_values) - 1, int(len(ascending_values) * p / 100))
    return ascending_values[idx]


def fraction_at_or_below(ascending_values: list[int], threshold: int) -> float:
    """Fraction of values <= threshold - answers "how much of the sampled corpus would this
    threshold keep vs. filter out", so a *currently configured* value can be judged against
    real data instead of guessed at."""
    if not ascending_values:
        return 0.0
    return bisect.bisect_right(ascending_values, threshold) / len(ascending_values)


_PERCENTILES = (50, 75, 90, 95, 99, 100)


def _distribution_lines(counts: list[int], current_threshold: int) -> list[str]:
    ascending = sorted(counts)
    lines = [f"n={len(ascending)}, percentiles: " +
             ", ".join(f"p{p}={percentile(ascending, p)}" for p in _PERCENTILES)]
    kept_fraction = fraction_at_or_below(ascending, current_threshold)
    lines.append(f"current threshold {current_threshold} keeps {kept_fraction:.0%} of sampled "
                 f"values, filters the most common {1 - kept_fraction:.0%} as noise")
    return lines


def format_report(
    literal_counts: list[tuple[str, int]],
    import_counts: list[tuple[str, int]],
    current_literal_threshold: int,
    current_import_threshold: int,
    top_n: int = 15,
) -> str:
    lines = ["=== literal_noise_threshold ==="]
    lines.append(f"currently configured: {current_literal_threshold}")
    if not literal_counts:
        lines.append("(no literals found in this corpus - nothing to calibrate)")
    else:
        lines.append(f"sampled {len(literal_counts)} distinct literals, top {top_n} by match count:")
        for literal, count in literal_counts[:top_n]:
            lines.append(f"  {count:6d}  {literal!r}")
        lines.extend(_distribution_lines([c for _, c in literal_counts], current_literal_threshold))

    lines.append("")
    lines.append("=== import_noise_threshold ===")
    lines.append(f"currently configured: {current_import_threshold}")
    if not import_counts:
        lines.append("(no imports found in this corpus - nothing to calibrate)")
    else:
        lines.append(f"{len(import_counts)} import targets, top {top_n} by distinct importer count:")
        for target_file, count in import_counts[:top_n]:
            lines.append(f"  {count:6d}  {target_file}")
        lines.extend(_distribution_lines([c for _, c in import_counts], current_import_threshold))

    lines.append("")
    lines.append("A threshold keeping ~80-95% of values (filtering only the most common tail as "
                 "noise) is a reasonable range to judge by - not a hard rule. Review before "
                 "writing anything: set literal_noise_threshold/import_noise_threshold in "
                 ".layergrep.json by hand if you want to change them.")
    return "\n".join(lines)
