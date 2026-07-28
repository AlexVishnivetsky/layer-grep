import logging
import os
import time

# Child of "layergrep" (mcp_server.py attaches the single FileHandler to that parent and
# lets this propagate up) - checkpoints inside load_model() are logged at each sub-step
# (import vs constructor vs encode) so a stuck call can be pinned down precisely.
logger = logging.getLogger("layergrep.model_config")

# Overridable so the embedding model can be swapped without touching code:
#   LAYERGREP_MODEL="BAAI/bge-m3" layergrep index
MODEL_NAME = os.environ.get("LAYERGREP_MODEL", "intfloat/multilingual-e5-small")

# (query_prefix, passage_prefix) per model family. bge-small-en-v1.5 is English-only and
# ranks poorly on non-English queries - not a good default for a codebase whose comments
# aren't in English throughout. multilingual-e5-small ranks correctly regardless of query
# language, with a tighter score spread than bge on English-only text.
_QWEN3_QUERY_INSTRUCT = (
    "Instruct: Given a question or task about a codebase, retrieve relevant code "
    "snippets and their descriptions that fulfill the request\nQuery: "
)

MODEL_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "BAAI/bge-small-en-v1.5": ("Represent this sentence for searching relevant passages: ", ""),
    "BAAI/bge-m3": ("", ""),
    "jinaai/jina-embeddings-v2-base-code": ("", ""),
    # Qwen3-Embedding: instruction on the query only, passages used as-is (same convention
    # as e5-mistral/GTE-Qwen instruct embeddings)
    "Qwen/Qwen3-Embedding-0.6B": (_QWEN3_QUERY_INSTRUCT, ""),
    "Qwen/Qwen3-Embedding-4B": (_QWEN3_QUERY_INSTRUCT, ""),
    "Qwen/Qwen3-Embedding-8B": (_QWEN3_QUERY_INSTRUCT, ""),
}

EMBEDDING_DIMS: dict[str, int] = {
    "intfloat/multilingual-e5-small": 384,
    "intfloat/multilingual-e5-base": 768,
    "intfloat/multilingual-e5-large": 1024,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-m3": 1024,
    "jinaai/jina-embeddings-v2-base-code": 768,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2560,
    "Qwen/Qwen3-Embedding-8B": 4096,
}

# Models this size are only practical with GPU acceleration - on CPU they still work
# (load_model() never refuses to run something), just slowly. Used only to decide whether
# to print a heads-up when no CUDA device is available.
GPU_RECOMMENDED_MODELS: frozenset[str] = frozenset({
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
})


def get_prefixes(model_name: str) -> tuple[str, str]:
    return MODEL_PREFIXES.get(model_name, ("", ""))


def get_embedding_dim(model_name: str) -> int:
    if model_name in EMBEDDING_DIMS:
        return EMBEDDING_DIMS[model_name]
    return load_model(model_name).get_sentence_embedding_dimension()


_MODEL_CACHE: dict[str, object] = {}


class ModelNotCachedError(Exception):
    pass


def load_model(model_name: str, allow_download: bool = False):
    """This is meant to run fully offline (no API, CPU-only, per the project's constraints) -
    two distinct modes, not one path that quietly does both:

    - allow_download=False (the default - used by indexer.py/search.py/mcp_server.py, i.e.
      every "use the tool" path): strictly offline, HF_HUB_OFFLINE=1, no exceptions. If the
      model isn't downloaded yet this raises ModelNotCachedError immediately instead of ever
      touching the network - no retry-with-backoff loop that can silently stall for minutes
      waiting on a socket. "Search never phones home" is a hard guarantee here, not just the
      common case.
    - allow_download=True (used by embed_check.py, i.e. the deliberate "set up/try a new
      model" path): behaves like a normal first-time sentence-transformers load, network
      allowed, so a genuinely new model can be fetched.

    Also caches by model_name in-process either way - load once per process, reuse after
    that (this used to reconstruct the model from scratch on every single call, which alone
    cost ~5s per search even when nothing was actually wrong).
    """
    if model_name in _MODEL_CACHE:
        logger.info(f"load_model({model_name!r}): in-process cache hit")
        return _MODEL_CACHE[model_name]

    import os
    import sys

    t0 = time.monotonic()
    logger.info(f"load_model({model_name!r}): cache miss, allow_download={allow_download}, pid={os.getpid()}")

    # Must be set *before* importing sentence_transformers/huggingface_hub, not after:
    # huggingface_hub freezes HF_HUB_OFFLINE into a module-level constant at first import
    # time (huggingface_hub/constants.py) and reads that frozen snapshot afterward, not
    # os.environ live - setting the env var post-import is a silent no-op.
    prev_offline = os.environ.get("HF_HUB_OFFLINE")
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"

    logger.info(f"load_model({model_name!r}): importing torch...")
    import torch
    logger.info(f"load_model({model_name!r}): torch imported after {time.monotonic() - t0:.3f}s, "
                f"cuda_available={torch.cuda.is_available()}")

    logger.info(f"load_model({model_name!r}): importing sentence_transformers...")
    from sentence_transformers import SentenceTransformer
    logger.info(f"load_model({model_name!r}): sentence_transformers imported after "
                f"{time.monotonic() - t0:.3f}s")

    import huggingface_hub.constants as _hfh_constants
    logger.info(f"load_model({model_name!r}): huggingface_hub.constants.HF_HUB_OFFLINE="
                f"{_hfh_constants.HF_HUB_OFFLINE}")

    if model_name in GPU_RECOMMENDED_MODELS and not torch.cuda.is_available():
        print(
            f"warning: {model_name!r} is a large model best run on GPU, but no CUDA device "
            f"is available here - falling back to CPU, this will be significantly slower.",
            file=sys.stderr,
        )

    try:
        logger.info(f"load_model({model_name!r}): constructing SentenceTransformer()...")
        t_ctor = time.monotonic()
        model = SentenceTransformer(model_name)
        logger.info(f"load_model({model_name!r}): SentenceTransformer() constructor returned "
                    f"after {time.monotonic() - t_ctor:.3f}s (total {time.monotonic() - t0:.3f}s)")
    except Exception as e:
        logger.info(f"load_model({model_name!r}): SentenceTransformer() raised {type(e).__name__} "
                    f"after {time.monotonic() - t_ctor:.3f}s: {e}")
        if prev_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_offline

        if not allow_download:
            raise ModelNotCachedError(
                f"{model_name!r} isn't downloaded yet, and this code path never touches "
                f"the network (allow_download=False). Run `layergrep install-model` (CLI) "
                f"or the install_model MCP tool to fetch it, then retry - or embed_check.py "
                f"if you're trying a model other than this project's configured default."
            ) from e

        logger.info(f"load_model({model_name!r}): retrying SentenceTransformer() with download allowed...")
        model = SentenceTransformer(model_name)  # deliberate first-time download
        logger.info(f"load_model({model_name!r}): download retry succeeded after "
                    f"{time.monotonic() - t0:.3f}s total")

    logger.info(f"load_model({model_name!r}): caching and returning, {time.monotonic() - t0:.3f}s total")
    _MODEL_CACHE[model_name] = model
    return model
