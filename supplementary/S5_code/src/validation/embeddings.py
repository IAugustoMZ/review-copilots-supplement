"""SPECTER embedding backend for the embedding-based suggestion models in
ml_assist.py (embedding_logistic_regression, embedding_decision_tree,
embedding_random_forest, embedding_gradient_boosting).

Model choice: sentence-transformers/allenai-specter, not generic BERT. SPECTER
is a SciBERT-family encoder fine-tuned via a citation-graph triplet objective
on scientific papers' title+abstract text -- i.e. trained so that
citation-related (bibliometrically related) papers land close together in
embedding space. That's a direct match for classifying bibliographic records,
and it's a drop-in sentence-transformers model (single .encode() call, no
extra adapter library, unlike SPECTER2). Runs on GPU automatically when one is
available (torch.cuda.is_available()), else CPU.

Encoding a transformer over every record is far more expensive than TF-IDF, so
results are cached to disk keyed by a hash of the exact text encoded --
ml_model_metrics.backfill_batches() re-embeds heavily overlapping "prior sets"
for every historical batch, and this cache is what keeps that fast: each
distinct title+abstract string is ever sent through the model once, ever.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from review_pipeline.src.common.config import DATA_DIR

EMBEDDING_MODEL_NAME = "sentence-transformers/allenai-specter"
EMBEDDING_DIM = 768

DEFAULT_EMBEDDING_CACHE_PATH = DATA_DIR / "embedding_cache" / "specter.npz"

_model = None  # lazy singleton -- loading/downloading SPECTER is expensive, do it at most once per process


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        if device == "cuda":
            # torch.cuda.is_available() only checks that a CUDA driver is
            # present, not that this torch build actually ships kernels for
            # this GPU's compute capability -- an older/unsupported card
            # (seen in practice: a GeForce MX110, CC 5.0, against a torch
            # build targeting CC>=7.5) reports "available" but throws
            # `CUDA error: no kernel image is available for execution on the
            # device` the moment a real tensor op runs. A one-token smoke
            # test surfaces that immediately, so it's caught here and falls
            # back to CPU, instead of every real encode() call failing and
            # train_all() silently (and wrongly) treating the whole
            # embedding backend as unavailable.
            try:
                model.encode(["gpu smoke test"], show_progress_bar=False)
            except Exception:  # noqa: BLE001 -- any failure here means "this GPU doesn't actually work," fall back
                model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        _model = model
    return _model


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}
    with np.load(cache_path) as data:
        hashes = data["hashes"]
        vectors = data["vectors"]
    return {h: vectors[i] for i, h in enumerate(hashes)}


def _save_cache(cache: dict[str, np.ndarray], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    hashes = np.array(list(cache.keys()))
    vectors = np.stack(list(cache.values())) if cache else np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    np.savez(cache_path, hashes=hashes, vectors=vectors)


def embed_texts(texts: list[str], cache_path: Path | None = None) -> np.ndarray:
    """Returns one row per entry in `texts`, in the same order, as
    L2-normalized SPECTER embeddings. Already-cached texts are read straight
    from `cache_path`; only cache misses are actually sent through the model,
    and the cache is updated with them before returning.

    cache_path defaults to DEFAULT_EMBEDDING_CACHE_PATH, looked up here
    (rather than as the parameter's default value) so tests can
    monkeypatch.setattr the module-level constant and have it actually take
    effect -- a default bound at def time wouldn't see that patch."""
    if cache_path is None:
        cache_path = DEFAULT_EMBEDDING_CACHE_PATH
    cache = _load_cache(cache_path)
    hashes = [_hash_text(t) for t in texts]
    missing_idx = [i for i, h in enumerate(hashes) if h not in cache]

    if missing_idx:
        model = _get_model()
        missing_texts = [texts[i] for i in missing_idx]
        encoded = model.encode(missing_texts, normalize_embeddings=True, show_progress_bar=False)
        for i, vector in zip(missing_idx, encoded, strict=True):
            cache[hashes[i]] = np.asarray(vector, dtype=np.float32)
        _save_cache(cache, cache_path)

    return np.stack([cache[h] for h in hashes])


class EmbeddingExtractor:
    """Mimics the subset of TfidfVectorizer's interface ml_assist.TrainedModel
    relies on (a fitted `.transform(texts) -> matrix`), so the same
    TrainedModel.predict() code path works unchanged for embedding-backed
    models -- no feature-kind branching needed at prediction time."""

    def __init__(self, cache_path: Path | None = None) -> None:
        self.cache_path = cache_path

    def transform(self, texts: list[str]) -> np.ndarray:
        return embed_texts(texts, cache_path=self.cache_path)
