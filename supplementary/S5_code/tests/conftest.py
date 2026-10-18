from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from review_pipeline.src.validation import embeddings

FAKE_EMBEDDING_DIM = 16


class _FakeEmbeddingModel:
    """Deterministic stand-in for the real SPECTER SentenceTransformer:
    same text always maps to the same vector, no network access, no model
    download, and no meaningful GPU/CPU cost -- what real encode() promises
    for our purposes, minus the ~440MB download and the runtime."""

    def encode(self, texts: list[str], normalize_embeddings: bool = True, show_progress_bar: bool = False):
        vectors = []
        for text in texts:
            seed = int(embeddings._hash_text(text)[:8], 16)
            vectors.append(np.random.RandomState(seed).rand(FAKE_EMBEDDING_DIM).astype("float32"))
        return np.stack(vectors)


@pytest.fixture(autouse=True)
def _fake_embedding_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The suggestion-model comparison (ml_assist.MODEL_SPECS) includes 4
    SPECTER-embedding-backed model types (src/validation/embeddings.py).
    Real SPECTER is a ~440MB download and needs network access -- unusable
    and undesirable in a test suite. Every test instead gets a tiny
    deterministic fake encoder and its own scratch cache file, applied
    globally (autouse) since any test exercising ml_assist.train_all /
    ml_model_metrics.backfill_batches / `classify assisted` ends up calling
    into this module. embed_texts()'s own caching logic (hash lookup,
    persistence) still runs for real against the fake model -- only the
    actual encode() call is swapped out."""
    monkeypatch.setattr(embeddings, "_model", None)
    monkeypatch.setattr(embeddings, "_get_model", lambda: _FakeEmbeddingModel())
    monkeypatch.setattr(embeddings, "DEFAULT_EMBEDDING_CACHE_PATH", tmp_path / "embedding_cache.npz")
