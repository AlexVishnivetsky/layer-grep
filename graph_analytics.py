from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import project_config as pconfig
from imports import extract_python_imports

logger = logging.getLogger("layergrep.graph_analytics")


def compute_module_stats(conn: sqlite3.Connection, project_root: Path) -> dict[str, dict]:
    """Derives each top-level module's role purely by aggregating the *already collected*
    `imports` table by (source module, target module) pairs - no new indexing needed, since
    extract_python_imports() resolves import edges against project_root regardless of
    whether the target file itself is inside the currently-indexed subtree (so a sibling
    module can have edges pointing at it even if it isn't itself chunked/embedded yet).

    A module heavy on incoming, light on outgoing cross-module edges is foundation-like
    (other code imports it, it imports little/nothing from siblings - e.g. a shared engine
    module). Heavy outgoing, light incoming is an extension/leaf (builds on other modules,
    nothing imports it back). Anything with both is an application/middle layer. This is a
    coarse, purely structural heuristic, not a judgment about code quality or importance - a
    module could be "foundation" by this measure just because nothing else in the repo
    happens to import from it yet."""
    project_config = pconfig.load_project_config(project_root)
    incoming: dict[str, set[str]] = {}
    outgoing: dict[str, set[str]] = {}
    for source_file, target_file in conn.execute("SELECT source_file, target_file FROM imports"):
        source_module = pconfig.classify_module(
            Path(source_file), project_root, project_config.module_depth, project_config.module_rules,
        )
        target_module = pconfig.classify_module(
            Path(target_file), project_root, project_config.module_depth, project_config.module_rules,
        )
        if not source_module or not target_module or source_module == target_module:
            continue
        outgoing.setdefault(source_module, set()).add(target_module)
        incoming.setdefault(target_module, set()).add(source_module)

    stats: dict[str, dict] = {}
    for mod in sorted(set(incoming) | set(outgoing)):
        in_mods = sorted(incoming.get(mod, set()))
        out_mods = sorted(outgoing.get(mod, set()))
        # foundation = other modules import FROM it, it imports nothing back;
        # extension = it imports from others, nothing (visible) imports it back.
        if in_mods and not out_mods:
            role = "foundation"
        elif out_mods and not in_mods:
            role = "extension"
        else:
            role = "application"
        stats[mod] = {"role": role, "imported_by_modules": in_mods, "imports_from_modules": out_mods}
    return stats


def find_runtime_unification(project_root: Path) -> dict[str, list[str]]:
    """Which modules share one deployed application via a common entry-point file (e.g.
    a wsgi.py importing from two different top-level modules). Entry points are .py files
    directly at project_root, not
    nested under any module dir - scanned from the filesystem directly (not the `imports`
    table), since they live outside whatever subtree is currently indexed and were never
    walked by index_project()/extract_python_imports() in the first place."""
    project_config = pconfig.load_project_config(project_root)
    unification: dict[str, list[str]] = {}
    for entry in sorted(project_root.glob("*.py")):
        if not entry.is_file():
            continue
        try:
            edges = extract_python_imports(entry, project_root)
        except Exception:
            # a broken/unparseable entry-point file shouldn't abort scanning the rest;
            # a narrower type isn't practical since tree-sitter parsing can fail in
            # several unrelated ways
            logger.warning(f"find_runtime_unification: failed to parse {entry}, skipping", exc_info=True)
            continue
        modules = sorted({
            m for m in (
                pconfig.classify_module(e.target, project_root, project_config.module_depth, project_config.module_rules)
                for e in edges
            ) if m
        })
        if len(modules) > 1:
            unification[str(entry)] = modules
    return unification
