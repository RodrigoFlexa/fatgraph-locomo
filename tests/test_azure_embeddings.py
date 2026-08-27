"""The Azure embedding backend.

The previous implementation could not have worked against the gateway it was
written for: it built its own ``AzureOpenAI`` from ``os.environ``, so it ignored
``FGL_CA_BUNDLE`` and any endpoint carrying a routing path, and it hard-coded
``dim = 1536`` -- silently wrong for ``text-embedding-3-large`` (3072). These
tests pin the four things that had to change.
"""

from __future__ import annotations

import pytest

from fgl.config import Config
from fgl.retrieval.embeddings import AzureEmbedder, build_embedder

# --------------------------------------------------------------------------- #
# A fake embeddings endpoint                                                   #
# --------------------------------------------------------------------------- #


class _FakeEmbeddings:
    def __init__(self, width=3072, script=()):
        self.width = width
        self.seen: list[dict] = []
        self.script = list(script)

    def create(self, **kwargs):
        self.seen.append(kwargs)
        if self.script:
            exc = self.script.pop(0)
            if exc is not None:
                raise exc
        n = len(kwargs["input"])
        width = kwargs.get("dimensions") or self.width
        data = [
            type("D", (), {"index": i, "embedding": [float(i + 1)] * width})()
            for i in range(n)
        ]
        # deliberately out of order: the client must sort by `index`
        return type("R", (), {"data": list(reversed(data))})()


def _embedder(width=3072, script=(), **kwargs):
    e = AzureEmbedder.__new__(AzureEmbedder)
    e._client = type("C", (), {"embeddings": _FakeEmbeddings(width, script)})()
    e._deployment = kwargs.pop("deployment", "embedding-3-large-global")
    e.normalize = kwargs.pop("normalize", True)
    e.batch_size = kwargs.pop("batch_size", 64)
    e.dimensions = kwargs.pop("dimensions", None)
    e.max_input_tokens = kwargs.pop("max_input_tokens", 8192)
    e.max_batch_tokens = kwargs.pop("max_batch_tokens", 60_000)
    e.max_retries = kwargs.pop("max_retries", 6)
    e.dim = e.dimensions or 1536
    e._dim_known = False
    assert not kwargs
    return e


# --------------------------------------------------------------------------- #
# Dimension                                                                    #
# --------------------------------------------------------------------------- #


def test_dimension_is_read_from_the_response_not_assumed():
    e = _embedder(width=3072)
    out = e.encode(["a", "b"])
    assert out.shape == (2, 3072)
    assert e.dim == 3072, "hard-coding 1536 would mis-shape the whole index"


def test_dimensions_parameter_is_passed_through():
    e = _embedder(dimensions=1024)
    out = e.encode(["a"])
    assert e._client.embeddings.seen[0]["dimensions"] == 1024
    assert out.shape == (1, 1024)


def test_a_gateway_that_rejects_dimensions_falls_back_to_full_width():
    """An optional parameter must not be able to fail an entire ingest."""

    class BadRequest(Exception):
        status_code = 400

        def __str__(self):
            return "Unrecognized request argument supplied: dimensions"

    e = _embedder(width=3072, dimensions=1024, script=[BadRequest()])
    out = e.encode(["a"])
    assert out.shape == (1, 3072)
    assert e.dimensions is None
    assert "dimensions" not in e._client.embeddings.seen[1]


# --------------------------------------------------------------------------- #
# Batching                                                                     #
# --------------------------------------------------------------------------- #


def test_batches_respect_the_count():
    e = _embedder(batch_size=2)
    e.encode(["a", "b", "c", "d", "e"])
    sizes = [len(c["input"]) for c in e._client.embeddings.seen]
    assert sizes == [2, 2, 1]


def test_batches_also_respect_a_token_budget():
    """batch_size alone does not respect the per-request total a gateway takes."""
    e = _embedder(batch_size=64, max_batch_tokens=100)  # ~350 chars
    e.encode(["x" * 200] * 4)
    sizes = [len(c["input"]) for c in e._client.embeddings.seen]
    assert len(sizes) > 1, "four 200-char inputs must not go in one request"
    assert all(s >= 1 for s in sizes)
    assert sum(sizes) == 4, "every input is still sent exactly once"


def test_an_over_long_input_is_truncated_rather_than_sent_to_fail():
    e = _embedder(max_input_tokens=10)  # 35 chars
    e.encode(["y" * 5000])
    sent = e._client.embeddings.seen[0]["input"][0]
    assert len(sent) == 35


def test_empty_text_becomes_a_space():
    e = _embedder()
    e.encode(["", "   "])
    assert e._client.embeddings.seen[0]["input"] == [" ", " "]


def test_order_is_restored_from_the_index_field():
    e = _embedder(width=4, normalize=False)
    out = e.encode(["a", "b", "c"])
    assert [row[0] for row in out] == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# Retries                                                                      #
# --------------------------------------------------------------------------- #


def test_a_rate_limit_is_retried(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    class RateLimit(Exception):
        status_code = 429

    e = _embedder(script=[RateLimit(), None])
    out = e.encode(["a"])
    assert out.shape[0] == 1
    assert len(e._client.embeddings.seen) == 2


def test_a_permanent_failure_names_the_deployment(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)

    class Boom(Exception):
        status_code = 401

    e = _embedder(script=[Boom()])
    with pytest.raises(RuntimeError, match="embedding-3-large-global"):
        e.encode(["a"])


# --------------------------------------------------------------------------- #
# Wiring                                                                       #
# --------------------------------------------------------------------------- #


def test_the_cache_tag_carries_the_width(tmp_path):
    """3072-wide vectors read back as 1024 would corrupt, not miss."""
    cfg = Config.load("G1").embeddings
    cfg.provider = "hashing"  # any provider: we only inspect the tag rule
    full = Config.load("G1").embeddings
    full.provider = "azure"
    full.cache_dir = str(tmp_path)
    full.azure_dimensions = None
    narrow = Config.load("G1").embeddings
    narrow.provider = "azure"
    narrow.cache_dir = str(tmp_path)
    narrow.azure_dimensions = 1024

    import fgl.retrieval.embeddings as mod

    made: list[str] = []
    mod_AzureEmbedder = mod.AzureEmbedder
    try:
        mod.AzureEmbedder = lambda *a, **k: _embedder()
        for c in (full, narrow):
            made.append(build_embedder(c).tag)
    finally:
        mod.AzureEmbedder = mod_AzureEmbedder

    assert made[0] != made[1]
    assert made[0].endswith("-dfull")
    assert made[1].endswith("-d1024")


def test_the_embedder_uses_the_shared_gateway_client(monkeypatch):
    """Regression: it used to read os.environ and ignore FGL_CA_BUNDLE."""
    calls: list[int] = []

    def fake_build():
        calls.append(1)
        return type("C", (), {"embeddings": _FakeEmbeddings()})(), None

    monkeypatch.setattr("fgl.llm.azure.build_azure_client", fake_build)
    AzureEmbedder("embedding-3-large-global")
    assert calls == [1], "the embedder must not build its own client"


def test_default_deployment_is_the_large_model():
    assert Config.load("G1").embeddings.azure_deployment == "embedding-3-large-global"
