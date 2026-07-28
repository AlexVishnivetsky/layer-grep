from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import sqlite_vec
from tqdm import tqdm

import json_index
import model_config

# alias: this module's own `project_config` local variable (the *loaded* ProjectConfig
# instance, used throughout index_project/_embed_and_store_chunks/etc.) would otherwise
# shadow the module name within the same scope
import project_config as pconfig
from chunker import EXTENSION_LANGS, Chunk, chunk_file
from db_schema import open_db
from imports import extract_python_imports

# Universal vendored/build directory names - true across ~any JS/Python/Rust project, not
# tied to one repo's layout. Project-specific additions (e.g. a vendored "plugins" dir that
# happens to live under one particular project's frontend assets) belong in that project's
# .layergrep.json extra_excluded_dirs, not hardcoded here for every project to inherit.
# ".venv"/"venv"/"site-packages" belong here rather than there: a Python project whose venv
# lives inside project_root would otherwise have its entire third-party dependency tree
# chunked/embedded as if it were first-party code. "target" is Cargo's build-output dir,
# same role as node_modules/dist/build.
EXCLUDED_DIR_NAMES = {
    ".git", "__pycache__", "node_modules", "dist", "build", ".idea",
    ".venv", "venv", "site-packages", "target",
}
MAX_FILE_SIZE = 200_000  # bytes - anything bigger in this codebase is a vendored/bundled file, not hand-written source
SUPPORTED_EXTENSIONS = set(EXTENSION_LANGS.keys())


def iter_source_files(
    root: Path,
    extra_excluded_dirs: frozenset[str] = frozenset(),
    forced_add: frozenset[str] = frozenset(),
):
    """extra_excluded_dirs entries with no "/" are bare directory names, excluded wherever
    they occur (like EXCLUDED_DIR_NAMES). Entries containing "/" are root-relative path
    prefixes, excluded only at that specific location - needed because a directory name can
    be vendored in one sibling module and legitimate first-party code in another (e.g. a
    vendored UI library bundled under one module's `libraries/`, while a different module's
    own `libraries/` holds real first-party code) - a bare-name exclusion would wrongly drop
    both.

    forced_add is the inverse escape hatch: a bare name in EXCLUDED_DIR_NAMES (or in
    extra_excluded_dirs) that this particular project wants indexed anyway - e.g. "target"
    is Cargo's build output in a Rust project, but nothing stops an unrelated Python project
    from having its own first-party directory that happens to share that name."""
    excluded_names = (EXCLUDED_DIR_NAMES | {e for e in extra_excluded_dirs if "/" not in e}) - forced_add
    excluded_prefixes = [e.strip("/") for e in extra_excluded_dirs if "/" in e]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if any(part in excluded_names for part in path.parts):
            continue
        if excluded_prefixes:
            rel = path.relative_to(root).as_posix()
            if any(rel == p or rel.startswith(p + "/") for p in excluded_prefixes):
                continue
        if path.name.endswith(".min.js"):
            continue
        try:
            if path.stat().st_size > MAX_FILE_SIZE:
                continue
        except OSError:
            continue
        yield path


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _delete_file_chunks(conn: sqlite3.Connection, file_path: str) -> None:
    ids = [row[0] for row in conn.execute("SELECT id FROM chunks WHERE file_path = ?", (file_path,))]
    if ids:
        conn.executemany("DELETE FROM chunk_vectors WHERE chunk_id = ?", [(i,) for i in ids])
        conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))


def index_project(conn: sqlite3.Connection, root: Path, model_name: str | None = None,
                   project_root: Path | None = None, force: bool = False) -> dict:
    model_name = model_name or model_config.MODEL_NAME
    # import-graph resolution base: a monorepo's sibling packages can be imported from
    # within `root` without living inside it - falls back to `root` itself for standalone
    # use where that distinction doesn't apply
    project_root = project_root or root
    project_config = pconfig.load_project_config(project_root)

    known_hashes = dict(conn.execute("SELECT path, content_hash FROM files"))
    seen_paths: set[str] = set()
    new_hashes: dict[str, str] = {}
    to_chunk: list[Path] = []
    unchanged = 0

    for path in iter_source_files(root, project_config.extra_excluded_dirs, project_config.forced_add):
        key = str(path)
        seen_paths.add(key)
        try:
            h = file_hash(path)
        except OSError:
            continue
        new_hashes[key] = h
        if not force and known_hashes.get(key) == h:
            unchanged += 1
        else:
            to_chunk.append(path)

    # only consider paths this scan is actually responsible for - `files` also tracks
    # non-code sources indexed through a separate path (e.g. ru.json via
    # index_json_translations), which iter_source_files() never yields and would otherwise
    # look "deleted" here on every single run
    deleted_paths = [
        p for p in known_hashes
        if p not in seen_paths and Path(p).suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    for p in deleted_paths:
        _delete_file_chunks(conn, p)
        conn.execute("DELETE FROM imports WHERE source_file = ?", (p,))
        conn.execute("DELETE FROM files WHERE path = ?", (p,))

    all_chunks: list[Chunk] = []
    parse_errors: list[str] = []
    for path in tqdm(to_chunk, desc="chunking", unit="file"):
        key = str(path)
        _delete_file_chunks(conn, key)
        try:
            file_chunks = chunk_file(path)
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't abort the whole
            # index run; a narrower type isn't practical since tree-sitter parsing can fail
            # in several unrelated ways. Recorded in parse_errors, not silently dropped.
            parse_errors.append(f"{path}: {e!r}")
            continue
        all_chunks.extend(file_chunks)

        conn.execute("DELETE FROM imports WHERE source_file = ?", (key,))
        if path.suffix.lower() == ".py":
            for edge in extract_python_imports(path, project_root):
                conn.execute(
                    "INSERT INTO imports (source_file, target_file, module) VALUES (?, ?, ?)",
                    (key, str(edge.target), edge.module),
                )

        conn.execute(
            "INSERT INTO files (path, content_hash) VALUES (?, ?) "
            "ON CONFLICT(path) DO UPDATE SET content_hash = excluded.content_hash",
            (key, new_hashes[key]),
        )

    _embed_and_store_chunks(conn, all_chunks, model_name, project_root)
    conn.commit()
    return {
        "changed_files": len(to_chunk),
        "unchanged_files": unchanged,
        "deleted_files": len(deleted_paths),
        "new_chunks": len(all_chunks),
        "parse_errors": parse_errors,
    }


def _embed_and_store_chunks(conn: sqlite3.Connection, chunks: list[Chunk], model_name: str,
                             project_root: Path, batch_size: int = 64) -> None:
    if not chunks:
        return
    _, passage_prefix = model_config.get_prefixes(model_name)
    model = model_config.load_model(model_name)
    project_config = pconfig.load_project_config(project_root)
    texts = [passage_prefix + c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=batch_size)

    cur = conn.cursor()
    for chunk, embedding in zip(chunks, embeddings):
        cur.execute(
            "INSERT INTO chunks (file_path, start_line, end_line, node_type, name, text, layer, "
            "chunk_source_kind, module) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(chunk.file_path), chunk.start_line, chunk.end_line, chunk.node_type, chunk.name, chunk.text,
             pconfig.classify_layer(chunk.file_path, project_config), chunk.chunk_source_kind,
             pconfig.classify_module(
                 chunk.file_path, project_root, project_config.module_depth, project_config.module_rules,
             )),
        )
        cur.execute(
            "INSERT INTO chunk_vectors (chunk_id, embedding) VALUES (?, ?)",
            (cur.lastrowid, sqlite_vec.serialize_float32(embedding.tolist())),
        )


def index_json_translations(conn: sqlite3.Connection, path: Path, project_root: Path,
                             model_name: str | None = None, force: bool = False) -> dict:
    """Which file(s) to call this on is project_config.translations_files - this function
    itself is generic, it just takes whatever path a caller hands it. Two independent
    outputs: literal (dot.path, value) rows for exact/substring lookup (json_entries, no
    embedding), and adaptively-sized chunks for the normal vector/per-layer path
    (chunk_source_kind='json_key', layer via classify_layer like any other chunk)."""
    model_name = model_name or model_config.MODEL_NAME
    key = str(path)

    try:
        current_hash = file_hash(path)
    except OSError:
        return {"changed": False, "new_entries": 0, "new_chunks": 0}

    known_hash = conn.execute("SELECT content_hash FROM files WHERE path = ?", (key,)).fetchone()
    if not force and known_hash is not None and known_hash[0] == current_hash:
        return {"changed": False, "new_entries": 0, "new_chunks": 0}

    _delete_file_chunks(conn, key)
    conn.execute("DELETE FROM json_entries WHERE file_path = ?", (key,))

    entries = json_index.literal_entries(path)
    conn.executemany(
        "INSERT INTO json_entries (file_path, key_path, value) VALUES (?, ?, ?)",
        [(key, key_path, value) for key_path, value in entries],
    )

    chunks = json_index.adaptive_chunks(path)
    _embed_and_store_chunks(conn, chunks, model_name, project_root)

    conn.execute(
        "INSERT INTO files (path, content_hash) VALUES (?, ?) "
        "ON CONFLICT(path) DO UPDATE SET content_hash = excluded.content_hash",
        (key, current_hash),
    )
    conn.commit()
    return {"changed": True, "new_entries": len(entries), "new_chunks": len(chunks)}


def default_db_path(root: Path, model_name: str) -> Path:
    """One project = its own .layergrep/ dir; one model = its own file within it,
    so neither projects nor models ever end up mixed in the same db."""
    model_slug = model_name.split("/")[-1]
    return root / ".layergrep" / f"{model_slug}.db"


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    # project_root: where .db lives + base for import-graph resolution (may contain sibling
    # packages that the indexed subtree imports from). index_root: subtree actually
    # walked/chunked/embedded - may be a subset of project_root while other sibling packages
    # aren't indexed yet.
    project_root = Path(sys.argv[1])
    index_root = project_root / sys.argv[2] if len(sys.argv) > 2 else project_root
    model_name = model_config.MODEL_NAME
    db_path = default_db_path(project_root, model_name)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = open_db(db_path, model_name, project_root)
    stats = index_project(conn, index_root, model_name=model_name, project_root=project_root)
    print(f"changed={stats['changed_files']} unchanged={stats['unchanged_files']} "
          f"deleted={stats['deleted_files']} new_chunks={stats['new_chunks']}")
    if stats["parse_errors"]:
        print(f"parse errors ({len(stats['parse_errors'])}):")
        for e in stats["parse_errors"][:20]:
            print(f"  {e}")

    project_config = pconfig.load_project_config(project_root)
    for rel_path in project_config.translations_files:
        translations_file = project_root / rel_path
        if translations_file.is_file():
            json_stats = index_json_translations(conn, translations_file, project_root, model_name=model_name)
            print(f"{rel_path}: changed={json_stats['changed']} "
                  f"entries={json_stats['new_entries']} chunks={json_stats['new_chunks']}")


if __name__ == "__main__":
    main()
