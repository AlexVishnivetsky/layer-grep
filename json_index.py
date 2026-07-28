from __future__ import annotations

import json
from pathlib import Path

from chunker import Chunk

# Bump on any change to literal_entries()/adaptive_chunks() that would produce different
# output for unchanged input - mirrors chunker.CHUNKER_VERSION's role for code chunking.
JSON_INDEX_VERSION = 3

# Neither "one chunk per top-level key" nor "one chunk per leaf" is a workable fixed
# granularity for a real translation file: a single top-level key can expand to thousands of
# leaves (one unfocused mega-chunk), while chunking every leaf individually explodes into
# singleton chunks whose short text gives a weak, barely-distinguishable embedding.
LEAF_BUDGET = 25
CHAR_BUDGET = 1200


def _flatten_leaves(node, prefix: str = ""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _flatten_leaves(value, f"{prefix}.{key}" if prefix else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _flatten_leaves(value, f"{prefix}[{i}]")
    else:
        yield prefix, "" if node is None else str(node)


def _subtree_size(node) -> tuple[int, int]:
    leaves = list(_flatten_leaves(node))
    total_chars = sum(len(p) + len(v) for p, v in leaves)
    return len(leaves), total_chars


def literal_entries(path: Path) -> list[tuple[str, str]]:
    """Flat (dot.path, value) pairs for exact/substring lookup - bypasses chunking/embedding
    entirely, since a short translation string gives a weak, barely-distinguishable vector
    but is trivial to find by literal match on its path or text."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return list(_flatten_leaves(data))


def adaptive_chunks(path: Path, leaf_budget: int = LEAF_BUDGET, char_budget: int = CHAR_BUDGET) -> list[Chunk]:
    """Recursive budget-based descent for the vector/semantic-search side: a subtree small
    enough becomes one chunk (dot-path prefix + all values under it as context); an
    oversized one descends into its nested-dict children and repeats - so `telephony` (1616
    leaves) splits into `telephony.validation`, `telephony.errors`, etc. instead of being one
    unfocused mega-chunk.

    Some dicts are oversized *without* being nested further (e.g. `menu` here is ~150 flat
    `"key": "label"` pairs, no sub-dicts to descend into) - "descend a level" has nowhere to
    go, so naively recursing would explode into one singleton chunk per leaf, exactly the
    over-fragmentation this is meant to avoid. Those sibling leaves are batched by budget
    instead, in key order, alongside any recursion into real sub-dicts."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    raw_text = path.read_text(encoding="utf-8")

    chunks: list[Chunk] = []
    # Best-effort line number, not exact: json.load() discards source positions, so recover
    # one by finding the chunk's first leaf key as a quoted string in the raw file text. A
    # monotonically-advancing cursor (never searches backward) keeps this correct even when
    # the same short key name recurs elsewhere in the file, since walk()/emit() process leaves
    # in the same top-to-bottom order they appear in the file.
    search_cursor = 0

    def find_line(key_path: str) -> int:
        nonlocal search_cursor
        last_segment = key_path.rsplit(".", 1)[-1].split("[", 1)[0]
        idx = raw_text.find(f'"{last_segment}"', search_cursor)
        if idx == -1:
            return 1
        search_cursor = idx + 1
        return raw_text.count("\n", 0, idx) + 1

    def emit(entries: list[tuple[str, str]], name: str) -> None:
        if not entries:
            return
        text = "\n".join(f"{key_path}: {value}" for key_path, value in entries)
        line = find_line(entries[0][0])
        chunks.append(Chunk(
            file_path=path,
            start_line=line,
            end_line=line,
            node_type="json_group",
            name=name,
            text=text,
            chunk_source_kind="json_key",
        ))

    def _child_entries(node, prefix: str):
        """(child_prefix, value) pairs for either a dict or a list - lists get the same
        `[i]` suffix convention _flatten_leaves already uses, so a large list is just
        another container that can recurse/batch instead of being emitted as one
        unsplittable mega-chunk."""
        if isinstance(node, dict):
            for key, value in node.items():
                yield (f"{prefix}.{key}" if prefix else key), value
        elif isinstance(node, list):
            for i, value in enumerate(node):
                yield f"{prefix}[{i}]", value

    def walk(node, prefix: str) -> None:
        n_leaves, n_chars = _subtree_size(node)
        if n_leaves == 0:
            return
        if not isinstance(node, (dict, list)) or (n_leaves <= leaf_budget and n_chars <= char_budget):
            emit(list(_flatten_leaves(node, prefix)), prefix or path.stem)
            return

        pending: list[tuple[str, str]] = []
        pending_leaves = 0
        pending_chars = 0
        batch_index = 0

        def flush_pending() -> None:
            nonlocal pending, pending_leaves, pending_chars, batch_index
            if pending:
                batch_index += 1
                base_name = prefix or path.stem
                emit(pending, f"{base_name}#{batch_index}")
            pending, pending_leaves, pending_chars = [], 0, 0

        for child_prefix, value in _child_entries(node, prefix):
            if isinstance(value, (dict, list)) and _subtree_size(value)[0] > 0:
                flush_pending()
                walk(value, child_prefix)
                continue

            leaves = list(_flatten_leaves(value, child_prefix))
            item_leaves = len(leaves)
            item_chars = sum(len(p) + len(v) for p, v in leaves)
            if pending and (pending_leaves + item_leaves > leaf_budget or pending_chars + item_chars > char_budget):
                flush_pending()
            pending.extend(leaves)
            pending_leaves += item_leaves
            pending_chars += item_chars
        flush_pending()

    walk(data, "")
    return chunks


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    target = Path(sys.argv[1])
    entries = literal_entries(target)
    chunks = adaptive_chunks(target)
    print(f"{len(entries)} literal entries, {len(chunks)} adaptive chunks\n")
    for chunk in chunks[:10]:
        n_lines = chunk.text.count("\n") + 1
        print(f"[{chunk.node_type}] {chunk.name}  (~{n_lines} lines, {len(chunk.text)} chars)")
        print("-" * 60)
        print(chunk.text[:300])
        print("=" * 60)


if __name__ == "__main__":
    main()
