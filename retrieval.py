from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

import sqlite_vec

import model_config

# alias: avoids shadowing the local `project_config` variable (the loaded ProjectConfig
# instance) used throughout search_by_layers
import project_config as pconfig

# Child of "layergrep" (see mcp_server.py, which attaches the single FileHandler to that
# parent logger and lets this one propagate up to it) - diagnostic checkpoints around the
# model-touching calls in search_by_layers(), logged per sub-step so a stuck call can be
# pinned down precisely.
logger = logging.getLogger("layergrep.retrieval")


def escape_like(s: str) -> str:
    """Escape SQL LIKE's own wildcards (`%` = any run of chars, `_` = any single char) so a
    literal query is matched literally. Without this, any `_` in the query - which is most
    real queries, since snake_case identifiers are exactly what literal search is for -
    silently means 'any one character' instead of 'this underscore', quietly breaking the
    'exact match' guarantee the feature is named for."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_json_literal(conn: sqlite3.Connection, query: str) -> list[tuple[str, str, str]]:
    """Exact/substring match on translation key paths or values - the literal-grep side of
    search for constants/translations, deliberately not going through embeddings at all."""
    like = f"%{escape_like(query)}%"
    return conn.execute(
        "SELECT file_path, key_path, value FROM json_entries WHERE key_path LIKE ? ESCAPE '\\' "
        "OR value LIKE ? ESCAPE '\\' ORDER BY LENGTH(value) ASC",
        (like, like),
    ).fetchall()


def search_by_layers(
    conn: sqlite3.Connection,
    query: str,
    project_root: Path,
    model_name: str | None = None,
    k_per_layer: int = 3,
    module_distance_margin: float = 0.15,
    layers: list[str] | None = None,
    candidate_pool: int = 600,
) -> dict[str, dict[str, list[tuple]]]:
    """Per-layer retrieval: a flat top-k over the whole corpus lets a high-volume layer (e.g.
    frontend) drown out a thin but equally relevant one (e.g. config) - empty-handed for a
    layer is itself a useful signal, and shouldn't be indistinguishable from "not in the
    top-k slots that filled up".

    Rather than a separate `k = ?` vec0 query per layer (which risks a layer coming back
    under-filled, since vec0's KNN runs before any join-side filter on `layer`), pull one
    generously sized candidate pool by pure vector distance, then group/truncate by layer
    in Python - correct and plenty fast at this corpus size (thousands, not millions).

    Sub-grouped by `module` within each layer: the same "noisy bucket drowns a thin one"
    problem per-layer search solves reappears one level down when a project has multiple
    modules of very different sizes indexed together - a flat top-k *within* one layer alone
    would let a large module's volume crowd out a genuinely relevant hit from a small one.
    Unlike layers (a fixed, enumerable list from project config), modules aren't pre-
    enumerated here - only modules that actually appear in the candidate pool for a given
    layer get a bucket, since the full module list is open-ended and most layer/module
    combinations are legitimately empty.

    module_distance_margin caps how many distinct modules get a bucket at all (not just how
    many results per module): a module's best candidate must be within `module_distance_margin`
    (fractional, default 0.15 = 15%) of the single best distance already found for that layer,
    or it's dropped as noise from the wide candidate pool rather than a genuinely additional
    relevant module. A fixed top-N-modules cap doesn't work here since the right module count
    varies per query - a relative margin from the layer's own best distance adapts to how
    tightly or loosely the real candidates cluster. Rows are distance-sorted (the underlying
    query is already ORDER BY v.distance), so the first-seen distance for a layer is exactly
    its best, computed lazily with no separate pass."""
    model_name = model_name or model_config.MODEL_NAME
    t0 = time.monotonic()
    logger.info(f"search_by_layers: calling load_model({model_name!r})")
    model = model_config.load_model(model_name)
    logger.info(f"search_by_layers: load_model returned after {time.monotonic() - t0:.3f}s")
    query_prefix, _ = model_config.get_prefixes(model_name)

    t1 = time.monotonic()
    logger.info("search_by_layers: calling model.encode() for query embedding")
    query_embedding = model.encode([query_prefix + query], normalize_embeddings=True)[0]
    logger.info(f"search_by_layers: model.encode() returned after {time.monotonic() - t1:.3f}s")

    t2 = time.monotonic()
    logger.info(f"search_by_layers: running vec0 MATCH query (candidate_pool={candidate_pool})")
    rows = conn.execute(
        """
        SELECT c.file_path, c.start_line, c.end_line, c.node_type, c.name, c.layer, c.module, v.distance, c.text
        FROM chunk_vectors v
        JOIN chunks c ON c.id = v.chunk_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (sqlite_vec.serialize_float32(query_embedding.tolist()), candidate_pool),
    ).fetchall()
    logger.info(f"search_by_layers: vec0 MATCH query returned {len(rows)} rows after "
                f"{time.monotonic() - t2:.3f}s")

    if layers is None:
        project_config = pconfig.load_project_config(project_root)
        layers = [name for name, _, _ in project_config.layer_rules] + [project_config.default_layer]

    grouped: dict[str, dict[str, list[tuple]]] = {layer: {} for layer in layers}
    layer_best_distance: dict[str, float] = {}
    for file_path, start_line, end_line, node_type, name, layer, module, distance, text in rows:
        layer_bucket = grouped.get(layer)
        if layer_bucket is None:
            continue
        # rows are distance-sorted, so the first hit seen for a layer is necessarily its best
        best = layer_best_distance.setdefault(layer, distance)
        if module not in layer_bucket and distance > best * (1 + module_distance_margin):
            continue  # this module's best candidate trails the layer's best match by more
                      # than the margin - treated as noise from a wide candidate pool, not a
                      # genuinely additional relevant module (see module_distance_margin doc)
        module_bucket = layer_bucket.setdefault(module, [])
        if len(module_bucket) < k_per_layer:
            module_bucket.append((file_path, start_line, end_line, node_type, name, distance, text))

    logger.info(f"search_by_layers: total {time.monotonic() - t0:.3f}s")
    return grouped


def get_forward_imports(conn: sqlite3.Connection, file_path: str) -> list[tuple[str, str]]:
    """(target_file, module) pairs this file imports directly - not transitive."""
    return conn.execute(
        "SELECT DISTINCT target_file, module FROM imports WHERE source_file = ?",
        (file_path,),
    ).fetchall()


# Above this many distinct importing files, a target is treated as a widely-shared utility
# module (a generic constants/config/helpers file) rather than something specific to the
# seed file's own feature, and skipped. Corpus-size-dependent - exposed as project_config.py's
# import_noise_threshold, this module constant is just the fallback default for direct
# callers that don't have a ProjectConfig at hand (e.g. tests).
_IMPORT_TARGET_NOISE_THRESHOLD = 15


def expand_via_imports(conn: sqlite3.Connection, file_paths: list[str],
                        chunks_per_file: int = 1,
                        noise_threshold: int = _IMPORT_TARGET_NOISE_THRESHOLD) -> dict[str, list[tuple]]:
    """For each seed file (typically a search hit's file_path), pull the files it imports
    directly and a representative chunk from each (its first class/function chunk) - a
    structural link independent of vector similarity between the handler's and the model's
    own embeddings. Complements literal-match cross-layer linking, doesn't replace it. Skips
    targets imported by more than noise_threshold distinct files - callers with a
    ProjectConfig should pass its import_noise_threshold instead of relying on the default,
    since the right cutoff depends on corpus size."""
    expanded: dict[str, list[tuple]] = {}
    for file_path in file_paths:
        entries = []
        for target_file, module in get_forward_imports(conn, file_path):
            importer_count = conn.execute(
                "SELECT COUNT(DISTINCT source_file) FROM imports WHERE target_file = ?", (target_file,)
            ).fetchone()[0]
            if importer_count > noise_threshold:
                continue
            rows = conn.execute(
                """
                SELECT file_path, start_line, end_line, node_type, name, layer FROM chunks
                WHERE file_path = ?
                ORDER BY CASE node_type WHEN 'class' THEN 0 WHEN 'function' THEN 1 ELSE 2 END, start_line
                LIMIT ?
                """,
                (target_file, chunks_per_file),
            ).fetchall()
            for row in rows:
                entries.append((module, *row))
        if entries:
            expanded[file_path] = entries
    return expanded
