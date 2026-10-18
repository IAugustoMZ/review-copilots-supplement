from __future__ import annotations

from pathlib import Path

import numpy as np

from review_pipeline.src.validation import embeddings

# The autouse _fake_embedding_backend fixture in conftest.py already replaces
# embeddings._get_model with a deterministic fake and points
# DEFAULT_EMBEDDING_CACHE_PATH at a scratch file -- these tests exercise
# embed_texts()'s own caching logic (hash lookup, persistence, ordering)
# against that fake, never against real SPECTER.


def _counting_model(monkeypatch, calls: list[list[str]]) -> None:
    real_get_model = embeddings._get_model

    class _Counting:
        def encode(self, texts, **kwargs):
            calls.append(list(texts))
            return real_get_model().encode(texts, **kwargs)

    monkeypatch.setattr(embeddings, "_get_model", lambda: _Counting())


def test_embed_texts_returns_one_row_per_text_in_order() -> None:
    texts = ["a chemical process copilot", "a folk music survey", "a chemical process copilot"]

    vectors = embeddings.embed_texts(texts)

    assert vectors.shape[0] == 3
    assert vectors.shape[1] > 0
    # Same text -> same vector, whether repeated within one call or not.
    assert np.array_equal(vectors[0], vectors[2])
    assert not np.array_equal(vectors[0], vectors[1])


def test_embed_texts_caches_to_disk_and_reuses_on_a_fresh_call(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "cache.npz"
    calls: list[list[str]] = []
    _counting_model(monkeypatch, calls)

    first = embeddings.embed_texts(["text one", "text two"], cache_path=cache_path)
    assert calls == [["text one", "text two"]]

    second = embeddings.embed_texts(["text one", "text two"], cache_path=cache_path)
    assert len(calls) == 1  # no new encode() call -- both texts were already cached
    assert np.array_equal(first, second)


def test_embed_texts_only_encodes_cache_misses(tmp_path: Path, monkeypatch) -> None:
    cache_path = tmp_path / "cache.npz"
    calls: list[list[str]] = []
    _counting_model(monkeypatch, calls)

    embeddings.embed_texts(["alpha", "beta"], cache_path=cache_path)
    embeddings.embed_texts(["beta", "gamma"], cache_path=cache_path)

    assert calls == [["alpha", "beta"], ["gamma"]]  # "beta" was already cached from the first call


def test_embed_texts_persists_across_process_boundary_via_cache_file(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    first = embeddings.embed_texts(["persisted text"], cache_path=cache_path)

    embeddings._model = None  # simulate a fresh process: no in-memory model, only the on-disk cache
    second = embeddings.embed_texts(["persisted text"], cache_path=cache_path)

    assert np.array_equal(first, second)
    assert cache_path.exists()


def test_embedding_extractor_transform_matches_embed_texts(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.npz"
    extractor = embeddings.EmbeddingExtractor(cache_path=cache_path)

    texts = ["one text", "another text"]
    assert np.array_equal(extractor.transform(texts), embeddings.embed_texts(texts, cache_path=cache_path))


def test_embed_texts_default_cache_path_is_resolved_dynamically(tmp_path: Path, monkeypatch) -> None:
    """embed_texts()/EmbeddingExtractor() must look up
    DEFAULT_EMBEDDING_CACHE_PATH inside the function body rather than binding
    it as a parameter default at def time -- a def-time-bound default
    wouldn't see a later monkeypatch.setattr of the module constant, which is
    exactly what conftest.py's autouse fixture relies on to keep every test
    off the real project's cache file."""
    redirected = tmp_path / "redirected.npz"
    monkeypatch.setattr(embeddings, "DEFAULT_EMBEDDING_CACHE_PATH", redirected)

    embeddings.embed_texts(["some text"])  # no explicit cache_path -- must use the (patched) default

    assert redirected.exists()
