from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import project_config as pconfig
import retrieval

_QUOTED_RE = re.compile(r'''["']([^"']{2,})["']''')
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_./]*")


def extract_literal_tokens(query: str) -> list[str]:
    """Pull candidate exact-match tokens out of a natural-language query: quoted substrings
    (highest confidence), plus identifier-looking words - snake_case, CONSTANT_CASE, dotted
    paths, route-like slashes - as opposed to ordinary lowercase prose words which aren't
    worth a literal search."""
    tokens: list[str] = []

    for m in _QUOTED_RE.finditer(query):
        tokens.append(m.group(1))

    for m in _TOKEN_RE.finditer(query):
        word = m.group(0)
        if len(word) < 3:
            continue
        looks_like_code = (
            "_" in word
            or "/" in word
            or "." in word
            or word.isupper()
            or (any(c.isupper() for c in word) and any(c.islower() for c in word))
        )
        if looks_like_code:
            tokens.append(word)

    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    # longest first: a specific token (`update_task_status_handler`) shouldn't be
    # crowded out in the result cap by a short substring of itself (`status`)
    unique.sort(key=len, reverse=True)
    return unique


def _extract_chunk_literals(text: str, limit: int = 4) -> list[str]:
    """Quoted string literals from a *found chunk's own code* - route paths, field/param
    names passed as strings, i18n keys, constant names - the highest-precision signal that
    the same concrete thing is referenced elsewhere, as opposed to running the query-oriented
    extract_literal_tokens() heuristic against a whole function body (which would flag nearly
    every local variable name as a 'code-looking' token - way too noisy to fan out a search
    per hit). Capped short (default 4) since this runs once per per-layer result, not once
    per query. Also skips quoted strings with no alphanumeric character at all - formatting
    separators like ',\r\n    ' or ' # ' picked up from things like a multi-line ", ".join(...)
    call or an inline comment marker are not identifiers."""
    seen: set[str] = set()
    tokens: list[str] = []
    for m in _QUOTED_RE.finditer(text):
        s = m.group(1)
        if s in seen or s.isdigit() or len(s) < 3 or not any(c.isalnum() for c in s):
            continue
        seen.add(s)
        tokens.append(s)
    tokens.sort(key=len, reverse=True)
    return tokens[:limit]


# Above this many total chunk matches (name/text, any layer), a literal is treated as too
# generic to be a meaningful cross-layer "bridge" and skipped - genuinely specific
# identifiers cluster at a handful of matches, while generic words (e.g. common field names)
# cluster in the hundreds or thousands, with a wide gap between the two. Corpus-size-
# dependent - exposed as project_config.py's literal_noise_threshold, this module constant
# is just the fallback default for direct callers without a ProjectConfig at hand (e.g.
# tests).
_LITERAL_NOISE_THRESHOLD = 30


def cross_layer_literal_links(
    conn: sqlite3.Connection,
    by_layer: dict[str, dict[str, list[tuple]]],
    limit_per_literal: int = 3,
    noise_threshold: int = _LITERAL_NOISE_THRESHOLD,
) -> dict[str, list[tuple[str, list[tuple]]]]:
    """For each per-layer result, search its own literal strings (route paths, field/constant
    names - see _extract_chunk_literals) in chunks from *other* layers - complements, doesn't
    replace, the import-graph link: semantic similarity between layers can be weak (different
    language, different code style) even for the same feature, but a route path or field name
    is often physically the same string in both places.

    Skips literals above noise_threshold total matches - generic words produce links that say
    nothing meaningful and eat context without proportional value; see
    _LITERAL_NOISE_THRESHOLD's comment for the rationale. Callers with a ProjectConfig should
    pass its literal_noise_threshold instead of relying on the default, since the right
    cutoff depends on corpus size."""
    links: dict[str, list[tuple[str, list[tuple]]]] = {}
    for source_layer, module_buckets in by_layer.items():
        for rows in module_buckets.values():
            for file_path, start, end, node_type, name, dist, text in rows:
                source_key = f"{Path(file_path).name}::{name}"
                for literal in _extract_chunk_literals(text):
                    like = f"%{retrieval.escape_like(literal)}%"
                    total_matches = conn.execute(
                        "SELECT COUNT(*) FROM chunks WHERE name LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\'",
                        (like, like),
                    ).fetchone()[0]
                    if total_matches > noise_threshold:
                        continue
                    matches = conn.execute(
                        """
                        SELECT file_path, start_line, end_line, node_type, name, layer
                        FROM chunks
                        WHERE layer != ? AND (name LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\')
                        ORDER BY LENGTH(text) ASC
                        LIMIT ?
                        """,
                        (source_layer, like, like, limit_per_literal),
                    ).fetchall()
                    if matches:
                        links.setdefault(source_key, []).append((literal, matches))
    return links


def search_literal(conn: sqlite3.Connection, query: str, limit: int = 15) -> list[tuple]:
    """Exact/substring match against indexed chunk name/text - the fast, high-confidence
    path for queries that already contain the literal string being searched for."""
    tokens = extract_literal_tokens(query)
    if not tokens:
        return []

    seen_ids: set[int] = set()
    results: list[tuple] = []
    for token in tokens:
        like = f"%{retrieval.escape_like(token)}%"
        # file_path LIKE too, not just name/text: a query for a filename the user already
        # knows (e.g. a daemon's own module name) should still find it even if that name
        # never appears as a chunk's `name`/`text`. Ranked alongside a name match (tier 1),
        # not above an exact `name` match (tier 0) or below a plain text hit.
        rows = conn.execute(
            """
            SELECT id, file_path, start_line, end_line, node_type, name, layer,
                   CASE WHEN name = ? THEN 0
                        WHEN name LIKE ? ESCAPE '\\' OR file_path LIKE ? ESCAPE '\\' THEN 1
                        ELSE 2 END AS rank
            FROM chunks
            WHERE name LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\' OR file_path LIKE ? ESCAPE '\\'
            ORDER BY rank ASC, LENGTH(text) ASC
            LIMIT ?
            """,
            (token, like, like, like, like, like, limit),
        ).fetchall()
        for chunk_id, file_path, start, end, node_type, name, layer, rank in rows:
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            results.append((file_path, start, end, node_type, name, layer, token, rank))

    results.sort(key=lambda r: r[7])
    return results[:limit]


def search_hybrid(
    conn: sqlite3.Connection,
    query: str,
    project_root: Path,
    model_name: str | None = None,
    k_per_layer: int = 3,
    literal_limit: int = 10,
    expand_imports: bool = True,
) -> dict:
    """Full pipeline: literal grep first (exact strings win outright), per-layer vector
    search second (so no domain gets drowned out), import-graph expand third (structural
    handler -> model/config/constants links, independent of embedding similarity)."""
    project_config = pconfig.load_project_config(project_root)
    literal_hits = search_literal(conn, query, limit=literal_limit)
    # Whole-query substring, not extract_literal_tokens(): translation values are natural-
    # language phrases, not code identifiers, so the code-oriented token heuristic (snake_case/
    # CONSTANT_CASE/dotted) that search_literal() uses would extract nothing from a plain query
    # like "create a new task".
    json_literal_hits = retrieval.search_json_literal(conn, query)
    by_layer = retrieval.search_by_layers(conn, query, project_root, model_name=model_name, k_per_layer=k_per_layer)

    expanded: dict[str, list[tuple]] = {}
    if expand_imports:
        seed_files = {row[0] for row in literal_hits}
        for module_buckets in by_layer.values():
            for rows in module_buckets.values():
                seed_files.update(row[0] for row in rows)
        expanded = retrieval.expand_via_imports(conn, list(seed_files), chunks_per_file=2,
                                                 noise_threshold=project_config.import_noise_threshold)

    # Complements expand_via_imports: that one follows real import edges (structural,
    # Python-only); this one follows shared literal strings (route paths, field/constant
    # names) across layers regardless of language - a JS frontend chunk can never "import"
    # a Python backend chunk, so this is the only structural (non-embedding) bridge for
    # exactly that cross-language case.
    literal_links = cross_layer_literal_links(conn, by_layer,
                                               noise_threshold=project_config.literal_noise_threshold)

    return {
        "literal": literal_hits,
        "json_literal": json_literal_hits,
        "by_layer": by_layer,
        "expanded_via_imports": expanded,
        "cross_layer_literal_links": literal_links,
    }


_SNIPPET_MAX_LINES = 15
_SNIPPET_MAX_CHARS = 800


def _indent_snippet(text: str) -> str:
    """Render a chunk's own code inline under its result line, capped in both dimensions -
    lets the consumer (an LLM reading search results) judge relevance directly from the code
    rather than from name + a vector distance number alone, which isn't reliable enough on
    its own."""
    truncated = False
    body = text
    if len(body) > _SNIPPET_MAX_CHARS:
        body = body[:_SNIPPET_MAX_CHARS]
        truncated = True
    snippet_lines = body.splitlines()
    if len(snippet_lines) > _SNIPPET_MAX_LINES:
        snippet_lines = snippet_lines[:_SNIPPET_MAX_LINES]
        truncated = True
    rendered = "\n".join(f"        {line}" for line in snippet_lines)
    if truncated:
        rendered += "\n        ..."
    return rendered


def format_results(query: str, result: dict) -> str:
    lines = [f"=== literal matches for {query!r} ==="]
    if not result["literal"]:
        lines.append("  (none)")
    for file_path, start, end, node_type, name, layer, token, rank in result["literal"]:
        lines.append(f"  [{layer}] [{node_type}] {name}  ({Path(file_path).name}:{start}, ends {end})  matched {token!r}")

    if result.get("json_literal"):
        lines.append("\n=== literal matches in translations (ru.json) ===")
        for file_path, key_path, value in result["json_literal"]:
            lines.append(f"  {key_path} = {value!r}  ({Path(file_path).name})")

    lines.append("\n=== per-layer vector search ===")
    for layer, module_buckets in result["by_layer"].items():
        lines.append(f"--- {layer} ---")
        if not module_buckets:
            lines.append("  (empty)")
        # Sub-grouped by module: within one layer, a high-volume module could otherwise
        # crowd out a genuinely relevant hit from a smaller one.
        for module, rows in module_buckets.items():
            lines.append(f"  [module: {module}]")
            for file_path, start, end, node_type, name, dist, text in rows:
                # "path:N" (single trailing number), not "path:start-end" - Claude Code's
                # terminal linkifies the former into a clickable file/line reference but not
                # a range (observed directly: a paired file:267 became a link, file:243-551
                # didn't).
                lines.append(f"    {dist:.4f}  [{node_type}] {name}  ({Path(file_path).name}:{start}, ends {end})")
                lines.append(_indent_snippet(text))

    if result["expanded_via_imports"]:
        lines.append("\n=== expanded via imports ===")
        for source_file, entries in result["expanded_via_imports"].items():
            lines.append(f"{Path(source_file).name}:")
            for module, target_file, start, end, node_type, name, layer in entries:
                lines.append(f"  -> {module}  =>  [{layer}] [{node_type}] {name}  ({Path(target_file).name})")

    if result.get("cross_layer_literal_links"):
        lines.append("\n=== cross-layer links via shared literal strings ===")
        for source_key, literal_matches in result["cross_layer_literal_links"].items():
            lines.append(f"{source_key}:")
            for literal, matches in literal_matches:
                for file_path, start, end, node_type, name, layer in matches:
                    lines.append(f"  {literal!r} also in [{layer}] [{node_type}] {name}  "
                                 f"({Path(file_path).name}:{start}, ends {end})")

    return "\n".join(lines)


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    import db_schema
    import model_config

    db_path = Path(sys.argv[1])
    query = sys.argv[2]
    model_name = model_config.MODEL_NAME
    # db_path is always <project_root>/.layergrep/<model_slug>.db (see
    # indexer.default_db_path) - derive project_root from it rather than adding a third argv.
    project_root = db_path.parent.parent

    conn = db_schema.open_db(db_path, model_name, project_root)
    result = search_hybrid(conn, query, project_root, model_name=model_name)
    print(format_results(query, result))


if __name__ == "__main__":
    main()
