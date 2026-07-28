import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("layergrep.project_config")

# Per-project config file, not a hardcoded constant: layer rules, translation file paths and
# extra excluded dirs are all project-specific. Fallback default is empty, not some other
# project's rules - silently reusing one project's naming conventions on an unconfigured
# different project would misclassify it confidently and wrongly, which is worse than the
# honest "everything is DEFAULT_LAYER" degradation this produces instead.
_BUILTIN_DEFAULT_LAYER = "backend/other"
_CONFIG_FILENAME = ".layergrep.json"


@dataclass(frozen=True)
class ProjectConfig:
    layer_rules: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = field(default_factory=list)
    default_layer: str = _BUILTIN_DEFAULT_LAYER
    translations_files: list[str] = field(default_factory=list)
    extra_excluded_dirs: frozenset[str] = frozenset()
    # Un-excludes a bare directory name that indexer.EXCLUDED_DIR_NAMES treats as vendored/
    # build output everywhere (e.g. "target" is Cargo's build dir, but a Python project could
    # have its own first-party directory that happens to be named "target" too) - without
    # this, a project could only ever narrow what gets indexed (extra_excluded_dirs), never
    # widen it back past the universal built-in list.
    forced_add: frozenset[str] = frozenset()
    # How many leading path components (relative to project_root) make up a "module" - see
    # classify_module(). 1 (default) is right for sibling packages that each live one dir
    # deep under project_root. A monorepo/workspace layout that nests packages one level
    # deeper (e.g. packages/service-a/...) needs 2, or classify_module would collapse every
    # package down to the shared "packages" component and lose the per-package grouping
    # search_by_layers relies on.
    module_depth: int = 1
    # Overrides module_depth's structural default for a business entity that doesn't align
    # with any single directory depth - e.g. a concern spread across a separate frontend/
    # backend split (`Deck.tsx` + `engine.rs`), which don't share a parent directory at all,
    # so no fixed leading-path-components count could ever unify them. Same (name, dirs,
    # files) shape as layer_rules, checked the same way - mirrors that mechanism instead of
    # depth because the whole reason it's needed is depth alone can't express this. Empty by
    # default: a monorepo of sibling packages (module_depth alone) needs no rules at all.
    module_rules: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = field(default_factory=list)
    # Corpus-size-dependent noise filters for the two cross-layer linking mechanisms
    # (search._LITERAL_NOISE_THRESHOLD / retrieval._IMPORT_TARGET_NOISE_THRESHOLD module
    # constants, kept as the function-level defaults) - a much smaller corpus (dozens to a
    # few hundred chunks) or a much larger one may want different cutoffs, so this is exposed
    # here instead of staying a fixed constant that only fits one project's scale.
    literal_noise_threshold: int = 30
    import_noise_threshold: int = 15
    source: str = "defaults (no .layergrep.json found)"


_PROJECT_CONFIG_CACHE: dict[Path, ProjectConfig] = {}


def load_project_config(project_root: Path) -> ProjectConfig:
    """Reads <project_root>/.layergrep.json once per process per project_root and caches
    it (same pattern as model_config._MODEL_CACHE) - cheap to call repeatedly from
    indexer.py/search.py without threading a loaded-config object through every call site."""
    project_root = project_root.resolve()
    if project_root in _PROJECT_CONFIG_CACHE:
        return _PROJECT_CONFIG_CACHE[project_root]

    config_path = project_root / _CONFIG_FILENAME
    if not config_path.is_file():
        project_config = ProjectConfig()
        logger.info(f"load_project_config({project_root}): no {_CONFIG_FILENAME} found, using empty defaults")
    else:
        with config_path.open(encoding="utf-8") as f:
            raw = json.load(f)
        layer_rules = [
            (entry["name"], tuple(entry.get("dirs", [])), tuple(entry.get("files", [])))
            for entry in raw.get("layers", [])
        ]
        module_rules = [
            (entry["name"], tuple(entry.get("dirs", [])), tuple(entry.get("files", [])))
            for entry in raw.get("modules", [])
        ]
        project_config = ProjectConfig(
            layer_rules=layer_rules,
            default_layer=raw.get("default_layer", _BUILTIN_DEFAULT_LAYER),
            translations_files=raw.get("translations", {}).get("files", []),
            extra_excluded_dirs=frozenset(raw.get("extra_excluded_dirs", [])),
            forced_add=frozenset(raw.get("forced_add", [])),
            module_depth=raw.get("module_depth", 1),
            module_rules=module_rules,
            literal_noise_threshold=raw.get("literal_noise_threshold", 30),
            import_noise_threshold=raw.get("import_noise_threshold", 15),
            source=str(config_path),
        )
        logger.info(f"load_project_config({project_root}): loaded {len(layer_rules)} layer rules from {config_path}")

    _PROJECT_CONFIG_CACHE[project_root] = project_config
    return project_config


def classify_layer(file_path: Path, project_config: ProjectConfig) -> str:
    parts_lower = {p.lower() for p in file_path.parts}
    name_lower = file_path.name.lower()
    for layer, dir_names, file_names in project_config.layer_rules:
        if any(d.lower() in parts_lower for d in dir_names):
            return layer
        if any(name_lower == f.lower() for f in file_names):
            return layer
    return project_config.default_layer


_RESOLVED_ROOT_CACHE: dict[str, Path] = {}


def _resolved_root(project_root: Path) -> Path:
    """project_root.resolve() is a filesystem syscall (resolves symlinks), and
    classify_module() is called once per CHUNK (not once per file) from
    indexer._embed_and_store_chunks's hot loop - project_root never changes across that
    whole call, so re-resolving it on every single chunk was pure redundant work (~5787
    avoidable syscalls on one real reindex run). Cached by string key, same pattern as
    model_config._MODEL_CACHE/_PROJECT_CONFIG_CACHE above."""
    key = str(project_root)
    resolved = _RESOLVED_ROOT_CACHE.get(key)
    if resolved is None:
        resolved = project_root.resolve()
        _RESOLVED_ROOT_CACHE[key] = resolved
    return resolved


def classify_module(
    file_path: Path,
    project_root: Path,
    module_depth: int = 1,
    module_rules: Sequence[tuple[str, tuple[str, ...], tuple[str, ...]]] = (),
) -> str:
    """Two mechanisms, checked in order. module_rules (optional, same shape as layer_rules,
    matched the same way as classify_layer) comes first, for a business entity that cuts
    across a directory boundary a fixed depth could never express - e.g. a concern spread
    across a separate frontend/backend split, which don't even share a parent directory.

    Falling through, the zero-heuristic default: the leading `module_depth` directory
    component(s) relative to project_root, joined with "/" - e.g. a sibling package name at
    the default depth 1, or "packages/service-a" at depth 2 for a monorepo/workspace layout.
    This is what a project with no rules configured relies on entirely, and it's the right
    call whenever layer nests *inside* each module's own directory (a sibling package that
    has its own frontend/ and api/ subdirs) - depth and naming-convention matching are
    independent dimensions there and compose for free. It stops being enough only when layer
    and module-shaped structure collapse onto the same top-level split (module_rules' case).

    Caps the depth at len(rel.parts) - 1 so a shallow file never has its own filename folded
    into the "module" (e.g. project_root/packages/index.js with module_depth=2 must not
    become module "packages/index.js" - that's not a package, it's a lone file one level
    below project_root). Falls back to "" for a file that IS project_root itself (no parent
    component) - shouldn't happen for real source files but avoids an IndexError on a
    degenerate input."""
    if module_rules:
        parts_lower = {p.lower() for p in file_path.parts}
        name_lower = file_path.name.lower()
        for name, dir_names, file_names in module_rules:
            if any(d.lower() in parts_lower for d in dir_names):
                return name
            if any(name_lower == f.lower() for f in file_names):
                return name

    try:
        rel = file_path.resolve().relative_to(_resolved_root(project_root))
    except ValueError:
        return ""
    if len(rel.parts) <= 1:
        return rel.parts[0] if rel.parts else ""
    depth = min(module_depth, len(rel.parts) - 1)
    return "/".join(rel.parts[:depth])
