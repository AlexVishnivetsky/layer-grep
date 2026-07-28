from __future__ import annotations

import json

from json_index import CHAR_BUDGET, LEAF_BUDGET, adaptive_chunks, literal_entries


def test_literal_entries_flattens_nested_dict(tmp_path, write_file):
    data = {"a": {"b": "hello", "c": "world"}, "top": "value"}
    p = write_file(tmp_path, "t.json", json.dumps(data))
    entries = dict(literal_entries(p))
    assert entries == {"a.b": "hello", "a.c": "world", "top": "value"}


def test_literal_entries_flattens_list(tmp_path, write_file):
    data = {"items": ["x", "y"]}
    p = write_file(tmp_path, "t.json", json.dumps(data))
    entries = dict(literal_entries(p))
    assert entries == {"items[0]": "x", "items[1]": "y"}


def test_literal_entries_none_becomes_empty_string(tmp_path, write_file):
    p = write_file(tmp_path, "t.json", json.dumps({"a": None}))
    entries = dict(literal_entries(p))
    assert entries == {"a": ""}


def test_adaptive_chunks_small_dict_is_one_chunk(tmp_path, write_file):
    data = {"a": "1", "b": "2"}
    p = write_file(tmp_path, "t.json", json.dumps(data))
    chunks = adaptive_chunks(p)
    assert len(chunks) == 1
    assert chunks[0].chunk_source_kind == "json_key"
    assert "a: 1" in chunks[0].text and "b: 2" in chunks[0].text


def test_adaptive_chunks_oversized_dict_splits_by_subtree(tmp_path, write_file):
    # two sub-dicts, each individually under budget, but the whole thing over LEAF_BUDGET -
    # should descend into "big"/"small" rather than emit one mega-chunk
    data = {
        "big": {f"k{i}": f"v{i}" for i in range(LEAF_BUDGET + 5)},
        "small": {"x": "y"},
    }
    p = write_file(tmp_path, "t.json", json.dumps(data))
    chunks = adaptive_chunks(p)
    names = [c.name for c in chunks]
    assert any(n.startswith("big") for n in names)
    assert any(n.startswith("small") for n in names)
    # no single chunk should exceed the leaf budget by much (allows the batching path's
    # own per-item granularity, but a whole 30-leaf "big" dict must not survive as one chunk)
    assert not any(n == "big" for n in names)


def test_adaptive_chunks_flat_oversized_dict_batches_by_budget(tmp_path, write_file):
    # "menu"-style: many flat key/value pairs, nothing to recurse into - must batch by
    # budget instead of exploding into one chunk per leaf
    data = {f"item{i}": f"label{i}" for i in range(LEAF_BUDGET * 2)}
    p = write_file(tmp_path, "t.json", json.dumps(data))
    chunks = adaptive_chunks(p)
    assert 1 < len(chunks) < LEAF_BUDGET * 2  # batched, not one-per-leaf and not one mega-chunk
    for c in chunks:
        assert c.text.count("\n") + 1 <= LEAF_BUDGET


def test_adaptive_chunks_splits_large_list(tmp_path, write_file):
    data = {"items": [f"value{i}" for i in range(200)]}
    p = write_file(tmp_path, "t.json", json.dumps(data))
    chunks = adaptive_chunks(p)
    assert len(chunks) > 1


def test_adaptive_chunks_char_budget_forces_split(tmp_path, write_file):
    # few leaves but each value is huge - char budget should still force a split even
    # though leaf count alone would fit in one chunk
    data = {f"k{i}": "x" * 500 for i in range(5)}
    p = write_file(tmp_path, "t.json", json.dumps(data))
    chunks = adaptive_chunks(p)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= CHAR_BUDGET + 600  # generous margin for one over-budget single item


def test_adaptive_chunks_line_numbers_point_at_real_content(tmp_path, write_file):
    raw = '{\n  "a": {\n    "b": "hello"\n  }\n}\n'
    p = write_file(tmp_path, "t.json", raw)
    chunks = adaptive_chunks(p)
    assert len(chunks) == 1
    # "b" (the first leaf key) is on line 3
    assert chunks[0].start_line == 3


def test_adaptive_chunks_line_numbers_advance_for_repeated_keys(tmp_path, write_file):
    # same leaf key name ("item0") appears in two different sibling groups, each small
    # enough to stay a single chunk on its own - padded with enough flat filler leaves that
    # the *top-level* dict is forced to descend into group_a/group_b as their own chunks
    # (a dict/list child is always recursed into once its parent is expanding, regardless of
    # the child's own size - see walk()'s _child_entries loop). The monotonically advancing
    # search cursor must not report the same (wrong, first) line for the second occurrence.
    data = {
        "group_a": {"item0": "v0"},
        "group_b": {"item0": "v0"},
        **{f"filler{i}": f"v{i}" for i in range(LEAF_BUDGET + 5)},
    }
    p = write_file(tmp_path, "t.json", json.dumps(data, indent=2))
    chunks = adaptive_chunks(p)
    by_name = {c.name: c for c in chunks}
    assert by_name["group_a"].start_line < by_name["group_b"].start_line


def test_json_index_version_is_int():
    import json_index
    assert isinstance(json_index.JSON_INDEX_VERSION, int)
