"""Swappable embedding backends + a tiny vector index.

Backends
--------
``sentence-transformers``  local, default (``all-MiniLM-L6-v2``);
``azure``                  Azure OpenAI ``text-embedding-3-*`` through the same
                           client (and the same private CA) as the chat
                           backend; ``azure_dimensions`` truncates the vector.
``hashing``                deterministic, dependency-free hashed bag-of-ngrams.
                           Not competitive, but it makes the *entire* pipeline
                           runnable and testable with no model download -- the
                           test suite uses it.

The index is a plain numpy inner-product search by default (exact, no
dependency).  FAISS is used when ``index.backend: faiss`` and the package is
installed; both expose the same ``add``/``search`` API.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from fgl.config import EmbeddingConfig, IndexConfig


# --------------------------------------------------------------------------- #
# Embedders                                                                    #
# --------------------------------------------------------------------------- #


class Embedder:
    """Interface: ``encode(list[str]) -> np.ndarray`` of shape (n, dim)."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class HashingEmbedder(Embedder):
    """Deterministic hashed character-ngram embedder (no external deps).

    Cosine similarity between related sentences is meaningful enough for smoke
    tests and unit tests, and it is bit-for-bit reproducible across machines.
    """

    def __init__(self, dim: int = 384, normalize: bool = True) -> None:
        self.dim = dim
        self.normalize = normalize

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _features(text):
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        if self.normalize:
            out = _l2(out)
        return out


def _features(text: str) -> Iterable[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    yield from words
    for a, b in zip(words, words[1:]):
        yield f"{a}_{b}"
    for w in words:
        for k in range(len(w) - 2):
            yield f"#{w[k:k+3]}"


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, normalize: bool = True, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_embedding_dimension())
        self.normalize = normalize
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)


class AzureEmbedder(Embedder):
    """Azure OpenAI embeddings, through the same client the chat backend uses.

    Four things this has to get right, and the previous version got none of
    them, which is why ``provider: azure`` had never actually run against the
    gateway:

    **The gateway.** Credentials, the private CA bundle and the
    ``base_url``-vs-``azure_endpoint`` distinction all come from
    :func:`fgl.llm.azure.build_azure_client`. The old implementation read
    ``os.environ`` directly and so ignored ``FGL_CA_BUNDLE`` and any endpoint
    that carries a routing path.

    **The token limit.** ``text-embedding-3-*`` takes 8192 tokens per input.
    An episode is 2-6 turns and stays far inside that, but a *batch* of 64 of
    them does not always stay inside the per-request total a gateway will
    accept. Batches are therefore cut by an estimated token budget as well as
    by a count, and an over-long single input is truncated rather than sent to
    fail.

    **The dimension.** Read from the first response, never assumed. The old
    code hard-coded 1536, which is silently wrong for
    ``text-embedding-3-large`` (3072) and would have mis-shaped the index.

    **Retries.** Embedding a whole corpus is thousands of calls; a single 429
    must not abort an ingest. Same backoff classification as the chat client.

    ``dimensions`` (Matryoshka truncation, supported by the ``-3-`` family) is
    passed through when set. It is the one knob that trades index memory for
    quality: 3072 -> 1024 cuts the vector store 3x.
    """

    #: chars per token, deliberately pessimistic so the estimate over-counts
    CHARS_PER_TOKEN = 3.5

    def __init__(
        self,
        deployment: str,
        normalize: bool = True,
        batch_size: int = 64,
        dimensions: int | None = None,
        max_input_tokens: int = 8192,
        max_batch_tokens: int = 60_000,
        max_retries: int = 6,
    ):
        from fgl.llm.azure import build_azure_client

        self._client, _ = build_azure_client()
        self._deployment = deployment
        self.normalize = normalize
        self.batch_size = max(1, batch_size)
        self.dimensions = dimensions
        self.max_input_tokens = max_input_tokens
        self.max_batch_tokens = max_batch_tokens
        self.max_retries = max(1, max_retries)
        #: provisional; overwritten by the first real response
        self.dim = dimensions or 1536
        self._dim_known = False

    # ------------------------------------------------------------ batching --
    def _prepare(self, text: str) -> str:
        # The API rejects an empty input outright, and a whitespace-only one is
        # the cheapest legal stand-in that keeps positions aligned.
        t = text if text and text.strip() else " "
        cap = int(self.max_input_tokens * self.CHARS_PER_TOKEN)
        return t[:cap]

    def _batches(self, texts: Sequence[str]) -> Iterable[list[str]]:
        budget = int(self.max_batch_tokens * self.CHARS_PER_TOKEN)
        batch: list[str] = []
        size = 0
        for raw in texts:
            t = self._prepare(raw)
            # A single item over budget still goes out alone: it was already
            # truncated to the per-input cap, so the server can take it.
            if batch and (len(batch) >= self.batch_size or size + len(t) > budget):
                yield batch
                batch, size = [], 0
            batch.append(t)
            size += len(t)
        if batch:
            yield batch

    # ---------------------------------------------------------------- call --
    def _embed(self, batch: list[str]) -> np.ndarray:
        import random
        import time

        from fgl.llm.azure import _is_retryable, _retry_after_seconds

        kwargs: dict = {"model": self._deployment, "input": batch}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.embeddings.create(**kwargs)
                # order is not guaranteed by contract; the index field is
                return np.asarray(
                    [d.embedding for d in sorted(resp.data, key=lambda d: d.index)],
                    dtype=np.float32,
                )
            except Exception as exc:  # noqa: BLE001 - classified below
                last = exc
                # A gateway that does not implement Matryoshka rejects
                # `dimensions`; drop it once and keep the full width rather
                # than failing the whole ingest over an optional parameter.
                if self.dimensions and "dimensions" in str(exc):
                    kwargs.pop("dimensions", None)
                    self.dimensions = None
                    continue
                if not _is_retryable(exc) or attempt == self.max_retries - 1:
                    break
                delay = min(60.0, 2.0**attempt) * (0.5 + random.random())
                time.sleep(max(delay, _retry_after_seconds(exc)))
        raise RuntimeError(
            f"embeddings do Azure falharam (deployment={self._deployment!r}, "
            f"batch de {len(batch)} textos): {last}"
        ) from last

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for batch in self._batches(texts):
            chunks.append(self._embed(batch))
        if not chunks:
            return np.zeros((0, self.dim), np.float32)
        out = np.vstack(chunks)
        if not self._dim_known:
            self.dim = int(out.shape[1])
            self._dim_known = True
        return _l2(out) if self.normalize else out


class CachedEmbedder(Embedder):
    """Memoises ``encode`` on disk, keyed by ``sha1(backend|model|text)``."""

    def __init__(self, inner: Embedder, cache_dir: str | Path, tag: str) -> None:
        self.inner = inner
        self.dim = inner.dim
        self.tag = tag
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._path = self.dir / f"{_slug(tag)}.npz"
        self._mem: dict[str, np.ndarray] = {}
        if self._path.exists():
            try:
                with np.load(self._path) as data:
                    self._mem = {k: data[k] for k in data.files}
            except Exception:
                self._mem = {}
        self._dirty = False

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        keys = [hashlib.sha1(f"{self.tag}\x1f{t}".encode()).hexdigest() for t in texts]
        missing = [(i, t) for i, (k, t) in enumerate(zip(keys, texts)) if k not in self._mem]
        if missing:
            vecs = self.inner.encode([t for _, t in missing])
            for (i, _), vec in zip(missing, vecs):
                self._mem[keys[i]] = vec
            self._dirty = True
        return np.vstack([self._mem[k] for k in keys]) if keys else np.zeros(
            (0, self.dim), np.float32
        )

    def flush(self) -> None:
        if self._dirty:
            np.savez_compressed(self._path, **self._mem)
            self._dirty = False


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.provider == "hashing":
        inner: Embedder = HashingEmbedder(cfg.dim, cfg.normalize)
        tag = f"hashing-{cfg.dim}"
    elif cfg.provider == "sentence-transformers":
        inner = SentenceTransformerEmbedder(cfg.model, cfg.normalize, cfg.batch_size)
        tag = f"st-{cfg.model}"
    elif cfg.provider == "azure":
        inner = AzureEmbedder(
            cfg.azure_deployment,
            cfg.normalize,
            cfg.batch_size,
            dimensions=cfg.azure_dimensions,
            max_input_tokens=cfg.azure_max_input_tokens,
            max_batch_tokens=cfg.azure_max_batch_tokens,
        )
        # The width is part of the cache key, not a detail: vectors written at
        # 3072 and read back as 1024 would concatenate into a silently
        # corrupt index instead of a cache miss.
        width = cfg.azure_dimensions or "full"
        tag = f"azure-{cfg.azure_deployment}-d{width}"
    else:
        raise ValueError(f"unknown embedding provider {cfg.provider!r}")
    return CachedEmbedder(inner, cfg.cache_dir, tag)


# --------------------------------------------------------------------------- #
# Vector index                                                                 #
# --------------------------------------------------------------------------- #


class VectorIndex:
    """Exact top-k search over unit vectors (cosine == inner product)."""

    def __init__(self, dim: int, backend: str = "numpy") -> None:
        self.dim = dim
        self.backend = backend
        self.ids: list[str] = []
        self._mat: np.ndarray | None = None
        self._faiss = None
        if backend == "faiss":
            try:
                import faiss  # noqa: F401

                self._faiss = faiss.IndexFlatIP(dim)
            except Exception:
                self.backend = "numpy"  # graceful degradation, documented

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vectors = _l2(np.asarray(vectors, dtype=np.float32))
        self.ids.extend(ids)
        if self._faiss is not None:
            self._faiss.add(vectors)
        else:
            self._mat = vectors if self._mat is None else np.vstack([self._mat, vectors])

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = _l2(np.asarray(query, dtype=np.float32).reshape(1, -1))
        k = min(k, len(self.ids))
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            return [(self.ids[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]
        scores = (self._mat @ q[0]).astype(np.float32)  # type: ignore[operator]
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.ids[int(i)], float(scores[int(i)])) for i in top]

    def __len__(self) -> int:
        return len(self.ids)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mat = self._mat
        if mat is None and self._faiss is not None:  # pragma: no cover
            import faiss

            mat = self._faiss.reconstruct_n(0, self._faiss.ntotal)
        np.savez_compressed(
            p.with_suffix(".npz"), vectors=mat if mat is not None else np.zeros((0, self.dim))
        )
        p.with_suffix(".json").write_text(
            json.dumps({"ids": self.ids, "dim": self.dim, "backend": self.backend}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        p = Path(path)
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        idx = cls(meta["dim"], meta.get("backend", "numpy"))
        with np.load(p.with_suffix(".npz")) as data:
            vectors = data["vectors"]
        idx.add(meta["ids"], vectors)
        return idx


def build_index(cfg: IndexConfig, dim: int) -> VectorIndex:
    if cfg.metric != "cosine":
        raise ValueError("only cosine similarity is implemented")
    return VectorIndex(dim, cfg.backend)


# --------------------------------------------------------------------------- #
# utils                                                                        #
# --------------------------------------------------------------------------- #


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_l2(a.reshape(1, -1))[0], _l2(b.reshape(1, -1))[0]))


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)
