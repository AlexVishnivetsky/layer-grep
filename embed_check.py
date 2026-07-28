import argparse
import sys
from pathlib import Path

import numpy as np

import model_config
from chunker import chunk_file

MODELS = {
    "bge": "BAAI/bge-small-en-v1.5",
    "e5": "intfloat/multilingual-e5-small",
    "e5-large": "intfloat/multilingual-e5-large",
    "bge-m3": "BAAI/bge-m3",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-8b": "Qwen/Qwen3-Embedding-8B",
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    parser = argparse.ArgumentParser(
        description="Sanity-check an embedding model's ranking quality against real chunks "
                    "from your own project - not part of the normal index/search path, this "
                    "is a one-off manual tool for trying a model before committing to it."
    )
    parser.add_argument("model_key", choices=sorted(MODELS), nargs="?", default="e5")
    parser.add_argument("--root", required=True, type=Path, help="project root")
    parser.add_argument("--file", action="append", required=True, dest="files",
                         help="file to chunk and embed, relative to --root (repeatable)")
    parser.add_argument("--query-comment-lang", required=True,
                         help="query phrased in this project's comment language")
    parser.add_argument("--query-code-lang", required=True,
                         help="the same query phrased in English (the code language)")
    args = parser.parse_args()

    model_name = MODELS[args.model_key]
    query_prefix, passage_prefix = model_config.get_prefixes(model_name)
    print(f"Model: {model_name}")

    model = model_config.load_model(model_name, allow_download=True)  # deliberate "try a model" tool

    chunks = []
    for rel_path in args.files:
        chunks.extend(chunk_file(args.root / rel_path))
    if not chunks:
        print("No chunks found in the given files.", file=sys.stderr)
        sys.exit(1)

    queries = {
        "query_comment_lang": query_prefix + args.query_comment_lang,
        "query_code_lang": query_prefix + args.query_code_lang,
    }
    passages = {f"{c.node_type}:{c.name}": passage_prefix + c.text for c in chunks}

    keys = list(queries) + list(passages)
    texts = list(queries.values()) + list(passages.values())
    embeddings = model.encode(texts, normalize_embeddings=True)
    by_key = dict(zip(keys, embeddings))

    print("\nCosine similarity to each query:\n")
    for qkey, query_text in queries.items():
        print(f"=== {qkey}: {query_text[len(query_prefix):]!r} ===")
        scores = [(k, float(np.dot(by_key[qkey], by_key[k]))) for k in passages]
        for k, s in sorted(scores, key=lambda x: -x[1]):
            print(f"  {s:.4f}  {k}")
        print()


if __name__ == "__main__":
    main()
