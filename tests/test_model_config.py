from __future__ import annotations

import pytest

import model_config


def test_get_prefixes_known_model():
    assert model_config.get_prefixes("intfloat/multilingual-e5-small") == ("query: ", "passage: ")


def test_get_prefixes_unknown_model_returns_empty_strings():
    assert model_config.get_prefixes("some/unregistered-model") == ("", "")


def test_get_embedding_dim_known_model_is_pure_lookup():
    assert model_config.get_embedding_dim("intfloat/multilingual-e5-small") == 384
    assert model_config.get_embedding_dim("BAAI/bge-m3") == 1024


def test_load_model_uncached_unknown_model_raises_without_network():
    # allow_download=False on a model that was never downloaded must fail fast with
    # ModelNotCachedError, not hang/retry - this is the exact guarantee load_model's
    # docstring exists to make.
    assert "totally/not-a-real-model" not in model_config._MODEL_CACHE
    with pytest.raises(model_config.ModelNotCachedError):
        model_config.load_model("totally/not-a-real-model", allow_download=False)


@pytest.mark.slow
def test_load_model_real_model_caches_and_encodes():
    model = model_config.load_model(model_config.MODEL_NAME)
    assert model_config.MODEL_NAME in model_config._MODEL_CACHE
    second = model_config.load_model(model_config.MODEL_NAME)
    assert second is model  # in-process cache hit, not reconstructed

    query_prefix, _passage_prefix = model_config.get_prefixes(model_config.MODEL_NAME)
    embedding = model.encode([query_prefix + "test query"], normalize_embeddings=True)[0]
    assert embedding.shape[0] == model_config.get_embedding_dim(model_config.MODEL_NAME)
