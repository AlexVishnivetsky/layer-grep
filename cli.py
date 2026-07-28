from __future__ import annotations

import argparse
import sys
from pathlib import Path

import calibration
import db_schema
import indexer
import init_wizard
import model_config

# alias: avoids shadowing the local `project_config` variable (the loaded ProjectConfig
# instance) in _run_index
import project_config as pconfig
import search

USAGE = 'usage: layergrep index [subdir] [--root PATH] [--model NAME]\n' \
        '       layergrep "<query>" [--root PATH] [--model NAME] [--k N] [--no-expand]\n' \
        '       layergrep -- "<query>"   (forces query mode - needed to search for the\n' \
        '                                  literal word "index" itself, which would otherwise\n' \
        '                                  always be parsed as the index subcommand)\n' \
        '       layergrep init-config [--root PATH]   (draft .layergrep.json from\n' \
        '                                  structural heuristics - review before use)\n' \
        '       layergrep calibrate-thresholds [--root PATH] [--model NAME] [--sample N]\n' \
        '                                  (suggest literal_noise_threshold/import_noise_threshold\n' \
        '                                  for this project\'s actual corpus - requires an existing\n' \
        '                                  index)\n' \
        '       layergrep install-model [--model NAME]   (download and cache an embedding\n' \
        '                                  model - one-time setup step on a machine that has\n' \
        '                                  never used this model before; needs network access)'


def _run_index(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="layergrep index")
    parser.add_argument("subdir", nargs="?", default=None,
                         help="subtree of --root to walk/chunk/embed (default: all of --root)")
    parser.add_argument("--root", default=".",
                         help="project root: where .layergrep/ lives and the base import "
                              "graph resolution uses (default: current directory)")
    parser.add_argument("--model", default=None, help=f"embedding model (default: {model_config.MODEL_NAME})")
    args = parser.parse_args(argv)

    project_root = Path(args.root).resolve()
    index_root = (project_root / args.subdir) if args.subdir else project_root
    model_name = args.model or model_config.MODEL_NAME

    db_path = indexer.default_db_path(project_root, model_name)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = db_schema.open_db(db_path, model_name, project_root)

    stats = indexer.index_project(conn, index_root, model_name=model_name, project_root=project_root)
    print(f"[code] changed={stats['changed_files']} unchanged={stats['unchanged_files']} "
          f"deleted={stats['deleted_files']} new_chunks={stats['new_chunks']}")
    if stats["parse_errors"]:
        print(f"parse errors ({len(stats['parse_errors'])}):")
        for e in stats["parse_errors"][:20]:
            print(f"  {e}")

    project_config = pconfig.load_project_config(project_root)
    for rel_path in project_config.translations_files:
        translations_file = project_root / rel_path
        if translations_file.is_file():
            json_stats = indexer.index_json_translations(conn, translations_file, project_root, model_name=model_name)
            print(f"[{rel_path}] changed={json_stats['changed']} "
                  f"entries={json_stats['new_entries']} chunks={json_stats['new_chunks']}")

    print(f"db: {db_path}")


def _run_init_config(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="layergrep init-config")
    parser.add_argument("--root", default=".", help="project root to scan (default: current directory)")
    args = parser.parse_args(argv)

    project_root = Path(args.root).resolve()
    draft = init_wizard.suggest_project_config(project_root)
    import json
    print(json.dumps(draft, indent=2, ensure_ascii=False))
    print(file=sys.stderr)
    init_wizard.write_draft_config(project_root, draft)


def _run_calibrate_thresholds(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="layergrep calibrate-thresholds")
    parser.add_argument("--root", default=".", help="project root (default: current directory)")
    parser.add_argument("--model", default=None, help=f"embedding model (default: {model_config.MODEL_NAME})")
    parser.add_argument("--sample", type=int, default=500,
                         help="max distinct literals to sample for literal_noise_threshold "
                              "(default: 500 - each one costs a full-table scan)")
    args = parser.parse_args(argv)

    project_root = Path(args.root).resolve()
    model_name = args.model or model_config.MODEL_NAME
    db_path = indexer.default_db_path(project_root, model_name)
    if not db_path.is_file():
        print(f"no index found at {db_path} - run `layergrep index` first", file=sys.stderr)
        sys.exit(1)

    conn = db_schema.open_db(db_path, model_name, project_root)
    project_config = pconfig.load_project_config(project_root)

    literal_counts = calibration.literal_match_counts(conn, sample=args.sample)
    import_counts = calibration.import_target_counts(conn)
    print(calibration.format_report(
        literal_counts, import_counts,
        project_config.literal_noise_threshold, project_config.import_noise_threshold,
    ))


def _run_install_model(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="layergrep install-model")
    parser.add_argument("--model", default=None, help=f"embedding model (default: {model_config.MODEL_NAME})")
    args = parser.parse_args(argv)

    model_name = args.model or model_config.MODEL_NAME
    # CLI counterpart of mcp_server.py's install_model tool, for anyone using `layergrep
    # index`/`layergrep "query"` directly without going through MCP.
    model_config.load_model(model_name, allow_download=True)
    print(f'Model "{model_name}" is downloaded and ready.')


def _run_search(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="layergrep")
    parser.add_argument("query")
    parser.add_argument("--root", default=".", help="project root (default: current directory)")
    parser.add_argument("--model", default=None, help=f"embedding model (default: {model_config.MODEL_NAME})")
    # A layer's results are split by module, and this many are kept per (layer, module) pair -
    # a layer with several qualifying modules can return well over --k total results (see
    # search_by_layers' module_distance_margin docstring in retrieval.py).
    parser.add_argument("--k", type=int, default=3, help="results per module within each layer (default: 3)")
    parser.add_argument("--no-expand", action="store_true", help="skip import-graph expand")
    args = parser.parse_args(argv)

    project_root = Path(args.root).resolve()
    model_name = args.model or model_config.MODEL_NAME
    db_path = indexer.default_db_path(project_root, model_name)
    if not db_path.is_file():
        print(f"no index found at {db_path} - run `layergrep index` first", file=sys.stderr)
        sys.exit(1)

    conn = db_schema.open_db(db_path, model_name, project_root)
    result = search.search_hybrid(conn, args.query, project_root, model_name=model_name, k_per_layer=args.k,
                                   expand_imports=not args.no_expand)
    print(search.format_results(args.query, result))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    if len(sys.argv) < 2:
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    if sys.argv[1] == "--":
        # explicit escape hatch: without it, `layergrep "index"` is indistinguishable from
        # the index subcommand and the word "index" could never be searched for as a query
        _run_search(sys.argv[2:])
    elif sys.argv[1] == "index":
        _run_index(sys.argv[2:])
    elif sys.argv[1] == "init-config":
        _run_init_config(sys.argv[2:])
    elif sys.argv[1] == "calibrate-thresholds":
        _run_calibrate_thresholds(sys.argv[2:])
    elif sys.argv[1] == "install-model":
        _run_install_model(sys.argv[2:])
    elif sys.argv[1] in ("-h", "--help"):
        print(USAGE)
    else:
        _run_search(sys.argv[1:])


if __name__ == "__main__":
    main()
