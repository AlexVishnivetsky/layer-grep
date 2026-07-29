# layergrep

Local, offline semantic + literal code search. Point it at a project, ask it a
question in plain language ("handler that updates user status"), get back
implementation locations grouped by architectural layer and module — not a
flat ranked list.

Available both as a CLI and as an MCP server, so an LLM client (e.g. Claude
Code) can call it directly during a conversation.

## Why "layergrep" and not "codesearch"

Plain "search" implies a single list of equivalent results ranked by score.
That's not how this tool works: it's a matrix — rows are **layers** (a
cross-cutting architectural role: `frontend`, `backend/api`, `models`,
`config`, ...), columns are **business entities/modules** (a sibling
package, a Rust crate, or a feature that spans both frontend and backend
files) — and results are grouped by both, so a large module can't drown out
a smaller one just because it has more code. `-grep` signals "searches
code" the way `ripgrep`/`ast-grep`/`semgrep` do; `layer` is the actual
differentiator.

## How it works

1. **Chunking** — AST-based (tree-sitter), not fixed-size windows. Functions,
   classes/structs, and methods for `.py`/`.js`/`.jsx`/`.ts`/`.tsx`/`.rs`/`.c`/
   `.h`/`.cpp`/`.cc`/`.cxx`/`.hpp`/`.hh`/`.hxx`; adjacent top-level statements
   (constants, route registration blocks) get
   grouped too, so plain data/config files aren't invisible to search. JSON
   files (e.g. translation catalogs) get both an exact literal lookup and
   adaptive size-budgeted chunking for semantic search.
2. **Embedding** — each chunk is embedded with a configurable
   sentence-transformers model (default `intfloat/multilingual-e5-small`;
   also supports `bge-small`/`bge-m3`/`e5-large`/the `Qwen3-Embedding` family).
   Stored in SQLite via [`sqlite-vec`](https://github.com/asg017/sqlite-vec) —
   no external vector database.
3. **Retrieval** — hybrid: literal grep on exact identifiers/paths first,
   then per-`(layer, module)` vector search (top-k *within* each bucket, not
   overall — the reason results don't collapse into a single dominant
   category), then two structural expansion passes: the import graph
   (handler → model/config/constants it imports — Python, JavaScript/
   TypeScript, Rust, C, and C++, see [Language support](#language-support)) and
   shared string literals across layers (language-agnostic, works
   everywhere — e.g. a route path used by both a JS frontend and a Python
   backend).
4. **Everything is project-specific and configurable**, not hardcoded —
   layer rules, translation file paths, module depth, and noise-filtering
   thresholds all live in `.layergrep.json` (see below).

## Language support

Not every feature is at full parity across languages yet.

| Feature | Python | JavaScript/TypeScript | Rust | C | C++ |
|---|---|---|---|---|---|
| AST chunking (functions/classes/methods) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Layer/module classification (path & name based) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cross-layer literal linking | ✅ | ✅ | ✅ | ✅ | ✅ |
| Import-graph expansion (`expand_via_imports`) | ✅ | ✅ | ✅ | ✅ | ✅ |
| `init-config` layer heuristics | ✅ | ✅ | ✅ | ✅ | ✅ |
| `init-config` manifest-based module detection | ❌ | ❌ | ✅ | ✅ | ❌ |

## Install

Requires Python 3.12+. Heavy ML dependencies (torch, sentence-transformers,
tree-sitter grammars) — expect a real download/install on first setup, this
is an embedding-model-backed tool, not a lightweight script.

```bash
git clone <this-repo>
cd layergrep
pip install -e .
```

Or, once published, with [`uv`](https://docs.astral.sh/uv/) — no manual venv
needed:

```bash
uvx --from git+<this-repo-url> layergrep-mcp-server
```

## Quickstart (CLI)

```bash
# 1. Draft a config from the target project's own structure (review it — it's a draft)
layergrep init-config --root /path/to/project

# 2. First time on this machine: download the embedding model (needs network, one-time)
layergrep install-model

# 3. Build the index (incremental — safe to re-run after code changes)
layergrep index --root /path/to/project

# 4. Search
layergrep "handler that updates user status" --root /path/to/project

# Optional: check whether the default noise-filtering thresholds fit this project's corpus size
layergrep calibrate-thresholds --root /path/to/project
```

`layergrep "index"` (the literal word) needs the `--` escape hatch, since
`index` alone is a subcommand: `layergrep -- "index"`.

## Quickstart (MCP server)

Add to `.mcp.json` in the target project:

```json
{
  "mcpServers": {
    "layergrep": {
      "command": "/path/to/layergrep/.venv/Scripts/python.exe",
      "args": ["/path/to/layergrep/mcp_server.py"],
      "env": {
        "LAYERGREP_ROOT": "/path/to/target/project"
      }
    }
  }
}
```

(Or, if installed via `pip`/`uv`, `command` can just be `layergrep-mcp-server`
directly — no venv path needed.)

Optional env vars: `LAYERGREP_SUBDIR` (index only a subtree of the project,
default: the whole root) and `LAYERGREP_MODEL` (override the embedding
model). Three tools are exposed:

- **`layergrep(query)`** — the search itself, returns markdown grouped by
  layer/module.
- **`index_codebase(force=False)`** — incremental (re)index; call after code
  changes.
- **`install_model()`** — one-time model download if the server reports it
  isn't cached yet (it won't crash on a fresh machine — it tells you to call
  this instead).

## Configuration: `.layergrep.json`

Optional, lives at the target project's root (not inside `.layergrep/`,
which holds only generated artifacts — the index DB and log — and should be
gitignored). Missing entirely → everything falls into one `default_layer`,
which is a valid, honest result for a project that hasn't been configured
yet — never silently reuses another project's layer rules.

```json
{
  "layers": [
    {"name": "frontend", "dirs": ["frontend", "client"], "files": []},
    {"name": "backend/api", "dirs": ["api", "routes"], "files": ["views.py"]},
    {"name": "models", "dirs": ["models"], "files": []},
    {"name": "config", "dirs": ["config"], "files": ["settings.py"]}
  ],
  "default_layer": "backend/other",
  "translations": {"files": ["langs/ru.json"]},
  "extra_excluded_dirs": ["vendor"],
  "forced_add": ["target"],
  "module_depth": 1,
  "modules": [
    {"name": "audio-engine", "dirs": [], "files": ["engine.rs", "Deck.tsx"]}
  ],
  "literal_noise_threshold": 30,
  "import_noise_threshold": 15
}
```

| Field | Default | Meaning |
|---|---|---|
| `layers` | `[]` | `(name, dirs, files)` rules, checked in order — first match wins |
| `default_layer` | `"backend/other"` | catch-all for anything no rule matches |
| `translations.files` | `[]` | JSON files to index for literal + semantic translation search |
| `extra_excluded_dirs` | `[]` | extra vendored/generated dirs to skip (bare name, or `a/b` path prefix) |
| `forced_add` | `[]` | un-excludes a bare dir name from the built-in skip list (e.g. `.venv`, `target`) — for when this project has its own first-party dir that happens to share that name |
| `module_depth` | `1` | how many leading path components form a "module" — raise for monorepos where packages are nested (e.g. `packages/service-a/...`) |
| `modules` | `[]` | `(name, dirs, files)` rules, checked before module_depth — for a business entity that crosses a directory boundary depth alone can't express (e.g. a frontend/backend split where the same feature has one file on each side) |
| `literal_noise_threshold` | `30` | max corpus-wide matches before a literal is treated as too generic to be a useful cross-layer link |
| `import_noise_threshold` | `15` | max distinct importers before an import target is treated as a shared utility, not a feature-specific link |

`layergrep init-config` drafts this file from common Flask/Django/FastAPI/
Express/React-shaped conventions — review before relying on it, it can't
guess a project's own vocabulary. It also detects which languages are
present, and for Rust projects reads real crate names straight out of every
`Cargo.toml` under the root (not the directory name — a directory called
`libxdo-sys-stub` can hold a crate actually named `libxdo-sys`) to suggest
`module_depth`; crates aren't suggested as `layers`, since a crate is a
sibling package (the `modules` axis), not an architectural role. For
`layers` specifically, Rust projects get their own candidate set instead —
Cargo conventions (`tests`/`examples`/`benches` dirs, a Tauri app's
`commands`/`handlers` surface) and one layer per `[[bin]]` target read
straight from the manifest — rather than the Flask/Django-shaped
directory-naming guesses, which don't apply to Rust code at all. C has no
single dominant manifest format the way Cargo.toml is for Rust (CMake,
Makefiles, Meson, Bazel, and Autotools all coexist), so this is scoped to
`CMakeLists.txt` only: every `add_library`/`add_executable` call with a
literal target name becomes its own `modules` rule (matched by the source
files' basenames), resolved straight from the manifest rather than guessed
from directory naming — a named CMake target is a sibling deliverable
(the `modules` axis, same reasoning as Rust crate names above), not a
recurring architectural role. A target built from a variable name (e.g.
`add_library(${LIB_NAME} ...)`, common in larger CMake projects) can't be
resolved this way and is skipped — a directory-naming-based layer guess
may still catch it. Same caution applies to individual source arguments:
a `$<...>` generator expression is dropped on its own (usually one extra
entry alongside otherwise-literal sources), but a bare `${...}` variable
reference aborts the whole target instead of just that argument — real
CMake very commonly builds an entire source list into one variable first,
and keeping whatever incidental literal token happens to sit next to it
would report a confidently-wrong, badly incomplete file list rather than
correctly detecting nothing. `layergrep calibrate-thresholds` reports the real
match-count distribution for the
`literal_noise_threshold`/`import_noise_threshold` fields on an already-
indexed project, instead of leaving you to guess whether the defaults (tuned
on one ~8000-chunk corpus) fit yours.

## Development

```bash
pip install -e ".[dev]"
pytest              # full suite
pytest -m "not slow"  # skip tests that load the real embedding model / spawn subprocesses
```

Optional local git hooks (same ruff/mypy/bandit checks CI runs, catching issues before a
commit instead of after a push):

```bash
pre-commit install                        # ruff/mypy/bandit before every commit
pre-commit install --hook-type pre-push   # full (non-"slow") test suite before every push
```

They run via whatever's already installed in your activated dev venv (`language: system` in
`.pre-commit-config.yaml`), not a separate managed environment — so a local run always
matches a manual `ruff check .`/`mypy .`/`bandit ...` invocation exactly, with nothing to
keep in sync.

## License

MIT — see [LICENSE](LICENSE).
