from __future__ import annotations

import json
import re
from pathlib import Path

from indexer import EXCLUDED_DIR_NAMES

# Extra directories to skip while *scanning for structure* (not indexing) - dependency/venv
# directories that would otherwise dominate the directory-name scan with irrelevant noise
# (e.g. a Python venv's own site-packages tree contains "models"/"config"-named dirs from
# unrelated third-party packages).
_SCAN_EXCLUDED_DIR_NAMES = EXCLUDED_DIR_NAMES | {"venv", ".venv", "site-packages", "vendor"}

# layer -> (directory names, file basenames) that suggest it, matched anywhere in the tree.
# Deliberately just common cross-framework conventions (Flask/Django/FastAPI/Express/React-
# ish naming): this catches the *structural* case for free; genuinely project-specific
# naming still needs a human to add or rename entries - no heuristic can guess
# project-specific vocabulary.
#
# File-basename candidates exist because directory-name matching alone misses a whole other
# convention family: classic Django apps put one concern per *file* at the app root
# (models.py/views.py/urls.py/admin.py), not per *directory*. project_config.classify_layer()
# already supports file-basename rules - this dict is what actually populates that half of
# the schema.
_LAYER_STRUCTURE_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "frontend": {
        "dirs": ["frontend", "client", "web", "ui", "react", "vue", "angular"],
        "files": [],
    },
    "backend/api": {
        "dirs": ["api", "routes", "handlers", "views", "endpoints", "controllers"],
        "files": ["views.py", "urls.py", "serializers.py", "routes.py", "endpoints.py", "controllers.py"],
    },
    "models": {
        "dirs": ["models", "schemas", "entities"],
        "files": ["models.py", "schemas.py", "entities.py"],
    },
    "config": {
        # "core" deliberately NOT included as a dir candidate - too ambiguous a name across
        # arbitrary projects (frequently means "central business logic", unrelated to
        # configuration) to treat as a config signal on its own. The more specific
        # "config.py" filename match below catches that convention instead (e.g. a
        # pydantic-settings Settings class living in core/config.py).
        "dirs": ["config", "configs", "settings"],
        "files": ["settings.py", "config.py"],
    },
    "constants": {
        "dirs": ["constants", "consts"],
        "files": ["constants.py", "consts.py"],
    },
    "daemons": {
        "dirs": ["daemons", "workers", "jobs", "cron", "tasks"],
        "files": ["tasks.py", "jobs.py", "cron.py", "worker.py", "workers.py"],
    },
}
# Rust-specific layer candidates - the web-framework dict above is actively wrong here (it
# guessed "frontend"/"models"/"config" from unrelated Flutter/Xcode folder names on a real
# Rust project), since Rust code doesn't follow Flask/Django-style naming at all. These are
# genuine Cargo/ecosystem conventions instead: tests/examples/benches are directories Cargo
# itself treats specially (not just a naming guess), and commands/handlers is the common
# shape of a Tauri app's IPC-exposed surface (the closest Rust analogue to "backend/api").
_RUST_LAYER_STRUCTURE_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "tests": {"dirs": ["tests"], "files": []},
    "examples": {"dirs": ["examples"], "files": []},
    "benches": {"dirs": ["benches"], "files": []},
    "commands": {"dirs": ["commands", "handlers"], "files": ["commands.rs", "handlers.rs"]},
}
_TRANSLATIONS_DIR_CANDIDATES = ["langs", "locales", "i18n", "translations"]
_FRONTEND_EXTENSIONS = {".jsx", ".tsx"}

# C/C++-family layer candidates - the web-framework dict above doesn't apply (a C/C++
# codebase has no "views.py"/"models.py"-style naming), and unlike Rust there's no single
# dominant ecosystem-wide project generator either, so these are genuine cross-project
# conventions instead: "include" (public headers, separate from "src"'s implementation) is
# close to universal, "tests"/"test" likewise (both spellings seen in real projects - curl
# uses "tests", PuTTY uses "test"). "platform" was added after inspecting real GUI-shaped C
# projects (PuTTY): native cross-platform C/C++ apps split their platform/GUI integration
# code into dedicated directories (windows/unix/gtk/macosx/...) the same way a web project
# splits out "frontend" - this is the closest C/C++ analogue to that layer, not something the
# issue text called out explicitly but confirmed empirically. "vendor" makes bundled
# third-party code its own visible layer instead of only ever being silently excluded
# (extra_excluded_dirs) - for a project that wants to *see* vendored code as its own bucket
# rather than hide it entirely.
#
# Shared verbatim between C and C++ (issue #24): this is really "a C-family project"
# convention, not distinctly a C or a C++ one - a namespace-based layer signal (the one
# genuinely C++-specific idea worth considering) isn't expressible through directory/file
# naming at all (namespaces are a source-level construct, not a filesystem one) and would
# need real AST inspection, not init_wizard.py's name-matching approach - out of scope here.
_C_LAYER_STRUCTURE_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "include": {"dirs": ["include", "inc", "public"], "files": []},
    "tests": {"dirs": ["tests", "test"], "files": []},
    "platform": {
        "dirs": ["windows", "unix", "macosx", "mac", "linux", "gtk", "android", "ios", "win32"],
        "files": [],
    },
    "vendor": {"dirs": ["third_party", "third-party", "vendor", "external", "deps"], "files": []},
}

# Purely informational (surfaced in the draft for a human to see at a glance) except for
# "Rust", which also gates whether the Cargo.toml-driven heuristic below bothers running -
# scanning for manifests on every init-config call would be wasted work on a project with no
# Rust in it at all. "C" covers both .c and .h - a header alone (no .c anywhere near it) still
# counts, since a header-only C library is a real and not even particularly rare case. "C++"
# deliberately does NOT include .h (already covered by "C" - a project can easily have both
# flags true at once, e.g. a C++ project using plain .h headers alongside .cpp sources; see
# _C_LAYER_STRUCTURE_CANDIDATES's single shared application below, gated on either being true,
# not each independently - applying the same dict twice would double every matched layer).
_LANGUAGE_EXTENSIONS: dict[str, frozenset[str]] = {
    "Python": frozenset({".py"}),
    "JavaScript": frozenset({".js", ".jsx"}),
    "TypeScript": frozenset({".ts", ".tsx"}),
    "Rust": frozenset({".rs"}),
    "C": frozenset({".c", ".h"}),
    "C++": frozenset({".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"}),
}


def _scan_tree(project_root: Path, max_depth: int = 5) -> tuple[dict[str, str], dict[str, str]]:
    """Single pass collecting both (lowercase dir name -> first-seen actual-case name) and
    (lowercase file basename -> first-seen actual-case name), anywhere under project_root up
    to max_depth (limits cost and avoids wandering into deeply nested vendored trees that
    slipped past the exclusion list). One combined walk instead of two separate rglob passes
    since both need the same traversal/exclusion logic anyway."""
    dir_names: dict[str, str] = {}
    file_names: dict[str, str] = {}
    root_depth = len(project_root.parts)
    for path in project_root.rglob("*"):
        if len(path.parts) - root_depth > max_depth:
            continue
        if any(part in _SCAN_EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.is_dir():
            dir_names.setdefault(path.name.lower(), path.name)
        elif path.is_file():
            file_names.setdefault(path.name.lower(), path.name)
    return dir_names, file_names


def _has_frontend_extension_signal(project_root: Path, max_files: int = 2000) -> bool:
    """Fallback signal when no directory matches the common frontend naming conventions -
    .jsx/.tsx files are an almost unambiguous frontend marker regardless of what the
    directory happens to be called. Capped scan (max_files) since this walks the whole tree
    without the layer-name shortcut; good enough for a yes/no signal, not exhaustive."""
    checked = 0
    for path in project_root.rglob("*"):
        if any(part in _SCAN_EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if path.is_file():
            checked += 1
            if path.suffix.lower() in _FRONTEND_EXTENSIONS:
                return True
            if checked >= max_files:
                break
    return False


def _detect_languages(project_root: Path, max_files: int = 5000) -> list[str]:
    """Which of the languages layergrep can chunk are actually present in this project, by
    file extension. Capped scan, same reasoning as _has_frontend_extension_signal - stops
    early once every known language has been seen, since there's nothing left to learn."""
    ext_to_lang = {ext: lang for lang, exts in _LANGUAGE_EXTENSIONS.items() for ext in exts}
    found: set[str] = set()
    checked = 0
    for path in project_root.rglob("*"):
        if any(part in _SCAN_EXCLUDED_DIR_NAMES for part in path.parts):
            continue
        if not path.is_file():
            continue
        checked += 1
        lang = ext_to_lang.get(path.suffix.lower())
        if lang:
            found.add(lang)
        if len(found) == len(_LANGUAGE_EXTENSIONS) or checked >= max_files:
            break
    return sorted(found)


def _find_rust_crates(project_root: Path) -> list[tuple[str, Path]]:
    """(crate_name, dir_relative_to_project_root) for every Cargo.toml under project_root
    that declares a [package] - a workspace-root manifest with no package of its own isn't a
    layer candidate, just a grouping declaration. Reads the crate's own name from the
    manifest rather than guessing it from the directory name, so e.g. a `libs/scrap`
    directory whose crate is actually named `screen-capture` still gets labelled correctly."""
    import tomllib
    crates: list[tuple[str, Path]] = []
    for cargo_path in sorted(project_root.rglob("Cargo.toml")):
        if any(part in _SCAN_EXCLUDED_DIR_NAMES for part in cargo_path.relative_to(project_root).parts):
            continue
        try:
            with cargo_path.open("rb") as f:
                data = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            continue
        name = data.get("package", {}).get("name")
        if not name:
            continue
        crates.append((name, cargo_path.parent.relative_to(project_root)))
    return crates


def _find_rust_bin_targets(project_root: Path) -> list[tuple[str, str]]:
    """(bin_name, file_basename) for every [[bin]] target declared across all Cargo.toml
    manifests under project_root - Cargo's own path default (src/bin/<name>.rs) applies when
    a [[bin]] entry omits "path". Each bin is a genuinely separate deliverable/entry point
    (unlike a plain module inside the crate it's built from), which is exactly the kind of
    cross-cutting distinction "layer" is meant to capture - e.g. rustdesk's Cargo.toml
    declares "naming" and "service" bins alongside its main app, each its own concern."""
    import tomllib
    bins: list[tuple[str, str]] = []
    for cargo_path in sorted(project_root.rglob("Cargo.toml")):
        if any(part in _SCAN_EXCLUDED_DIR_NAMES for part in cargo_path.relative_to(project_root).parts):
            continue
        try:
            with cargo_path.open("rb") as f:
                data = tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError):
            continue
        for entry in data.get("bin", []):
            name = entry.get("name")
            if not name:
                continue
            path = entry.get("path") or f"src/bin/{name}.rs"
            bins.append((name, Path(path).name))
    return bins


# #25/#26: CMake is C/C++'s closest equivalent to Cargo.toml being worth targeting first -
# the only one of the fragmented C/C++ build-tool landscape (CMake/Makefiles/Meson/Bazel/
# Autotools, no dominant standard) whose target declarations (add_library/add_executable)
# are parseable for the common literal case without a full CMake evaluator.
#
# Anchored on a preceding non-identifier boundary so a bare substring match doesn't fire
# inside an unrelated custom macro that happens to end the same way, e.g. fmt's CUDA test
# tree calls a distinct `cuda_add_executable(...)` macro with no guaranteed argument shape.
_CMAKE_TARGET_CALL_RE = re.compile(r"(?<!\w)add_(?:library|executable)\s*\(", re.IGNORECASE)

# Non-source keyword arguments that can appear in an add_library/add_executable call,
# combined into one set since neither list overlaps with the other's real source files.
_CMAKE_NON_SOURCE_KEYWORDS = frozenset({
    # add_library
    "STATIC", "SHARED", "MODULE", "OBJECT", "UNKNOWN", "ALIAS", "IMPORTED", "GLOBAL",
    "EXCLUDE_FROM_ALL", "INTERFACE",
    # add_executable
    "WIN32", "MACOSX_BUNDLE",
})
# A variable reference (${...}) or generator expression ($<...>) - can't resolve either
# without evaluating CMake, so any token matching this is treated as non-literal.
_CMAKE_DYNAMIC_TOKEN_RE = re.compile(r"[${}<>]")


def _extract_cmake_call_body(text: str, open_paren_idx: int) -> str | None:
    """Text between a call's open paren (at open_paren_idx) and its matching close paren,
    with any `#`-to-end-of-line comments stripped - or None if the parens never balance
    (truncated/malformed file, not worth guessing at). CMake command arguments don't nest
    parens in the common case, so plain depth-counting is enough - no real tokenizing
    parser needed. Comment-tracking has to share this same character-by-character scan
    rather than run as a separate pass afterward: real multi-line calls (seen in PuTTY's
    CMakeLists.txt) carry trailing `# comment` text inside the parens, and a stray `)`
    inside a comment must not prematurely end the match."""
    depth = 0
    in_comment = False
    body_chars: list[str] = []
    for i in range(open_paren_idx, len(text)):
        ch = text[i]
        if in_comment:
            if ch == "\n":
                in_comment = False
                body_chars.append(ch)
            continue
        if ch == "#":
            in_comment = True
        elif ch == "(":
            depth += 1
            if depth > 1:
                body_chars.append(ch)
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return "".join(body_chars)
            body_chars.append(ch)
        else:
            body_chars.append(ch)
    return None


def _tokenize_cmake_args(body: str) -> list[str]:
    """Splits a call's argument text on whitespace, treating a "..."-quoted run as one
    token with the quotes stripped - not CMake's full quoting/escaping rules, just enough
    for the common case (e.g. "${_target}.c", "curlinfo.c" both seen in real manifests)."""
    tokens = re.findall(r'"[^"]*"|\S+', body)
    return [t[1:-1] if len(t) >= 2 and t[0] == '"' and t[-1] == '"' else t for t in tokens]


def _find_cmake_targets(project_root: Path) -> list[tuple[str, list[str]]]:
    """(target_name, [source_basenames]) for every add_library/add_executable call found
    across all CMakeLists.txt under project_root - CMake's closest equivalent to Cargo's
    [[bin]]/[lib] targets. Each detected target becomes its own layer entry, the same way
    _find_rust_bin_targets' results do.

    Deliberately not a full CMake evaluator - only resolves calls whose target name is
    literal text, skipping anything built from a variable/generator-expression entirely
    (e.g. curl's real libcurl/CLI targets, all declared via
    `add_library(${LIB_STATIC} ...)` - expected to go undetected here, not a bug to chase;
    nothing downstream is resolvable without evaluating CMake once the name itself isn't
    literal). Individual *source* tokens that aren't literal are dropped one at a time
    rather than discarding the whole target - real-world calls often mix a handful of
    literal sources with one dynamic entry (e.g. PuTTY's
    `$<TARGET_OBJECTS:logging>` alongside a dozen literal .c files in the same call) -  a
    target left with zero real sources this way is skipped by the empty-result check
    below, so no separate all-or-nothing rule is needed.

    Multiple manifests declaring the same target name are merged (union of basenames),
    not treated as a collision - the expected, common case for cross-platform C, e.g.
    PuTTY declares "pageant" once in unix/CMakeLists.txt and once in windows/CMakeLists.txt
    with overlapping-but-not-identical sources; both genuinely belong to the same
    logical deliverable."""
    targets: dict[str, list[str]] = {}
    for cmake_path in sorted(project_root.rglob("CMakeLists.txt")):
        if any(part in _SCAN_EXCLUDED_DIR_NAMES for part in cmake_path.relative_to(project_root).parts):
            continue
        try:
            text = cmake_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in _CMAKE_TARGET_CALL_RE.finditer(text):
            body = _extract_cmake_call_body(text, match.end() - 1)
            if body is None:
                continue
            tokens = _tokenize_cmake_args(body)
            if not tokens:
                continue
            name, *rest = tokens
            if _CMAKE_DYNAMIC_TOKEN_RE.search(name):
                continue
            basenames = []
            for token in rest:
                if token in _CMAKE_NON_SOURCE_KEYWORDS or _CMAKE_DYNAMIC_TOKEN_RE.search(token):
                    continue
                source_path = cmake_path.parent / token
                if source_path.is_file():
                    basenames.append(source_path.name)
            if not basenames:
                continue
            existing = targets.setdefault(name, [])
            for basename in basenames:
                if basename not in existing:
                    existing.append(basename)
    return [(name, sorted(basenames)) for name, basenames in sorted(targets.items())]


def suggest_project_config(project_root: Path) -> dict:
    """Draft .layergrep.json content from structural heuristics alone - no LLM, no content
    analysis beyond directory/file names and one extension check. Meant to be reviewed and
    edited by a human before use, not applied blindly: a directory literally named "api" (or
    a file named "views.py") strongly suggests a backend/api layer in most Flask/Django/
    FastAPI/Express-shaped projects, but this has no way to know if a specific project's
    naming is unconventional, or to guess genuinely project-specific concepts."""
    dir_names, file_names = _scan_tree(project_root)
    detected_languages = _detect_languages(project_root)

    layers = []
    for layer_name, candidates in _LAYER_STRUCTURE_CANDIDATES.items():
        matched_dirs = [dir_names[c] for c in candidates["dirs"] if c in dir_names]
        matched_files = [file_names[c] for c in candidates["files"] if c in file_names]
        if matched_dirs or matched_files:
            layers.append({"name": layer_name, "dirs": matched_dirs, "files": matched_files})

    frontend_layer = next((entry for entry in layers if entry["name"] == "frontend"), None)
    if frontend_layer is None and _has_frontend_extension_signal(project_root):
        layers.insert(0, {
            "name": "frontend",
            "dirs": [],
            "files": [],
            "_comment": "no directory matched common frontend naming, but .jsx/.tsx files "
                        "were found somewhere in the tree - fill in \"dirs\" by hand once "
                        "you've located the actual frontend root",
        })

    # A Cargo crate is a "sibling package" the same way a top-level package is in a Python
    # monorepo - that's the module dimension (classify_module), not layer: layer is a
    # cross-cutting architectural role (frontend/backend/config), which crate names aren't.
    rust_crates = _find_rust_crates(project_root) if "Rust" in detected_languages else []

    if "Rust" in detected_languages:
        for layer_name, candidates in _RUST_LAYER_STRUCTURE_CANDIDATES.items():
            matched_dirs = [dir_names[c] for c in candidates["dirs"] if c in dir_names]
            matched_files = [file_names[c] for c in candidates["files"] if c in file_names]
            if matched_dirs or matched_files:
                layers.append({"name": layer_name, "dirs": matched_dirs, "files": matched_files})

        for bin_name, file_basename in _find_rust_bin_targets(project_root):
            layers.append({"name": bin_name, "dirs": [], "files": [file_basename]})

    # "C" and "C++" share this single application (not one `if` block per language) - both
    # being true at once is common (plain .h headers alongside .cpp sources), and applying
    # the same dict under each language's own `if` would double every matched layer entry.
    if "C" in detected_languages or "C++" in detected_languages:
        for layer_name, candidates in _C_LAYER_STRUCTURE_CANDIDATES.items():
            matched_dirs = [dir_names[c] for c in candidates["dirs"] if c in dir_names]
            matched_files = [file_names[c] for c in candidates["files"] if c in file_names]
            if matched_dirs or matched_files:
                layers.append({"name": layer_name, "dirs": matched_dirs, "files": matched_files})

    # #25: CMakeLists.txt target detection - gated on "C" alone (not C++ yet) even though
    # _find_cmake_targets itself doesn't care what language a target's sources are, to keep
    # this ticket's scope to what #25 asks for. #26 broadens this to "C" or "C++" once
    # validated separately against a real C++/CMake project, mirroring how
    # _C_LAYER_STRUCTURE_CANDIDATES above was itself extended from C to C++ (#24).
    if "C" in detected_languages:
        for target_name, basenames in _find_cmake_targets(project_root):
            layers.append({"name": target_name, "dirs": [], "files": basenames})

    # Computed only after every layer source (general dict, frontend fallback, Rust/C
    # conventions, bin targets) has had a chance to populate `layers` - checking this any
    # earlier could wrongly report "other" for a project whose only matches come from a
    # language-specific block below the check.
    default_layer = "backend/other" if layers else "other"

    translations_files: list[str] = []
    for cand in _TRANSLATIONS_DIR_CANDIDATES:
        actual = dir_names.get(cand)
        if actual is None:
            continue
        # Re-filter by _SCAN_EXCLUDED_DIR_NAMES here too, not just when first deciding
        # `actual` is a real candidate - this is a fresh, unrelated rglob over the whole
        # tree for any directory bearing that bare name, so it can still walk straight into
        # a vendored copy (e.g. node_modules/some-i18n-pkg/locales/en.json) even though the
        # dir_names dict itself only ever recorded a legitimate, non-vendored occurrence.
        for jf in project_root.rglob(f"{actual}/*.json"):
            if any(part in _SCAN_EXCLUDED_DIR_NAMES for part in jf.parts):
                continue
            translations_files.append(str(jf.relative_to(project_root)).replace("\\", "/"))

    draft: dict[str, object] = {
        "_generated_by": "layergrep init - REVIEW BEFORE USE, this is a draft, not a "
                          "verified config",
        "_detected_languages": detected_languages,
        "layers": layers,
        "default_layer": default_layer,
        "translations": {"files": translations_files},
        "extra_excluded_dirs": [],
    }
    # a nested crate (workspace member, or a rustdesk-style path-dependency sub-crate) means
    # module_depth=1 would collapse every file under it down to the shared top-level dir name
    # (e.g. everything under libs/ becomes module "libs") - bump it so classify_module's
    # per-module grouping actually distinguishes one sub-crate from another.
    if any(rel != Path(".") for _, rel in rust_crates):
        draft["module_depth"] = 2
    return draft


def write_draft_config(project_root: Path, draft: dict, out=None) -> Path | None:
    """Writes the draft to <project_root>/.layergrep.json unless one already exists (never
    silently overwrite a human-reviewed config with a heuristic guess) - returns the path if
    written, None if skipped. `out` is the stream status messages go to (defaults to
    sys.stderr so stdout stays clean for the JSON itself, matching cli.py's other commands)."""
    import sys
    out = sys.stderr if out is None else out
    out_path = project_root / ".layergrep.json"
    if out_path.is_file():
        print(f"NOTE: {out_path} already exists - not overwritten. Review the draft above "
              f"and merge by hand if useful.", file=out)
        return None
    out_path.write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Draft written to {out_path} - review it (especially layer names and "
          f"translations paths) before relying on it; this is a heuristic guess, not a "
          f"verified config.", file=out)
    return out_path


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    project_root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    draft = suggest_project_config(project_root)
    print(json.dumps(draft, indent=2, ensure_ascii=False))
    print(file=sys.stderr)
    write_draft_config(project_root, draft)


if __name__ == "__main__":
    main()
