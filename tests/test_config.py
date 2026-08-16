"""Condition resolution, `--set` overrides, validation and provenance."""

from __future__ import annotations

import pytest

from fgl.config import Config, ConfigError, list_conditions, resolve_condition


# --------------------------------------------------------------------------- #
# Discovery                                                                    #
# --------------------------------------------------------------------------- #


def test_every_shipped_condition_loads_and_validates():
    found = {cond for _, cond, _ in list_conditions()}
    assert {
        "B1-full-context", "B2-rag-turns", "B3-rag-facts",
        "G1-fatgraph-min", "G2-fatgraph-cur", "G3-fatgraph-agent",
    } <= found
    for _, cond, path in list_conditions():
        Config.from_yaml(path).validate()


@pytest.mark.parametrize(
    "alias", ["G1", "g1", "G1-fatgraph-min", "G1_fatgraph_min", "g1_fatgraph"]
)
def test_conditions_resolve_from_id_stem_or_prefix(alias):
    assert resolve_condition(alias).stem == "G1_fatgraph_min"


def test_unknown_condition_lists_the_alternatives():
    with pytest.raises(ConfigError, match="G1-fatgraph-min"):
        resolve_condition("does-not-exist")


def test_ambiguous_prefix_is_rejected():
    with pytest.raises(ConfigError, match="ambiguous"):
        resolve_condition("G")


# --------------------------------------------------------------------------- #
# Inheritance                                                                  #
# --------------------------------------------------------------------------- #


def test_condition_inherits_base_and_overrides_only_what_it_declares():
    cfg = Config.load("G2")
    assert cfg.condition == "G2-fatgraph-cur"
    assert cfg.curation.curation is True          # from the condition file
    assert cfg.curation.consolidation is True     # from the condition file
    assert cfg.retrieval.budget_tokens == 2000    # inherited from base.yaml
    assert cfg.source and cfg.source.endswith("G2_fatgraph_cur.yaml")


def test_the_three_g_conditions_differ_only_where_intended():
    g1, g2, g3 = Config.load("G1"), Config.load("G2"), Config.load("G3")
    assert set(g1.diff(g2)) == {"condition", "curation.curation", "curation.consolidation"}
    assert set(g2.diff(g3)) == {"condition", "ingest.sigma_policy"}


def test_b3_and_g1_share_the_extraction_cache_path():
    """The B3-vs-G1 ablation is only valid if the fact cache is identical."""
    assert Config.load("B3").paths.facts_cache == Config.load("G1").paths.facts_cache


# --------------------------------------------------------------------------- #
# --set overrides                                                              #
# --------------------------------------------------------------------------- #


def test_overrides_are_typed_from_the_dataclass():
    cfg = Config.load(
        "G1",
        overrides=[
            "retrieval.top_m_anchors=8",
            "retrieval.budget_tokens=4000",
            "retrieval.level2_boost=0.25",
            "curation.curation=true",
            "llm.deployment=gpt-4o",
            "retrieval.recall_ks=1,3,5",
        ],
    )
    assert cfg.retrieval.top_m_anchors == 8 and isinstance(cfg.retrieval.top_m_anchors, int)
    assert cfg.retrieval.budget_tokens == 4000
    assert cfg.retrieval.level2_boost == pytest.approx(0.25)
    assert cfg.curation.curation is True
    assert cfg.llm.deployment == "gpt-4o"
    assert cfg.retrieval.recall_ks == (1, 3, 5)


@pytest.mark.parametrize("flag", ["false", "0", "no", "off"])
def test_boolean_overrides_accept_the_usual_spellings(flag):
    assert Config.load("G2", overrides=[f"curation.curation={flag}"]).curation.curation is False


def test_a_typo_in_an_override_is_a_hard_error():
    with pytest.raises(ConfigError, match="unknown config key"):
        Config.load("G1", overrides=["retrieval.top_m_anchor=8"])
    with pytest.raises(ConfigError, match="unknown config section"):
        Config.load("G1", overrides=["retreival.top_m_anchors=8"])
    with pytest.raises(ConfigError, match="malformed override"):
        Config.load("G1", overrides=["retrieval.top_m_anchors"])
    with pytest.raises(ConfigError, match="boolean"):
        Config.load("G1", overrides=["curation.curation=maybe"])


@pytest.mark.parametrize(
    "override,message",
    [
        ("retrieval.top_m_anchors=zero", "integer"),
        ("retrieval.level2_boost=lots", "number"),
        ("retrieval.recall_ks=5,ten", "list of integers"),
    ],
)
def test_a_mistyped_value_is_a_config_error_not_a_crash(override, message):
    """It must exit as a configuration problem, never as an uncaught ValueError."""
    with pytest.raises(ConfigError, match=message):
        Config.load("G1", overrides=[override])


def test_an_unknown_yaml_key_is_a_hard_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("condition: X\nretrieval:\n  top_m_anchor: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown keys in config.retrieval"):
        Config.from_yaml(bad)


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "override",
    [
        "ingest.sigma_policy=sigma-random",
        "llm.provider=anthropic",
        "embeddings.provider=word2vec",
        "retrieval.top_m_anchors=0",
        "curation.min_face_len=1",
        "entities.match_threshold=0.5",  # below llm_threshold
    ],
)
def test_validation_rejects_impossible_settings(override):
    with pytest.raises(ConfigError):
        Config.load("G1", overrides=[override])


# --------------------------------------------------------------------------- #
# Serialisation                                                                #
# --------------------------------------------------------------------------- #


def test_round_trip_through_yaml_is_lossless():
    cfg = Config.load("G3", overrides=["retrieval.top_m_anchors=7"])
    back = Config.from_dict(__import__("yaml").safe_load(cfg.to_yaml()))
    assert back.flat() == cfg.flat()


def test_flat_keys_are_exactly_what_set_accepts():
    cfg = Config()
    for key in cfg.flat():
        if key in ("condition", "seed"):
            continue
        cfg.get(key)  # must not raise


def test_requires_azure_is_false_offline():
    assert Config.load("test_offline").requires_azure() is False
    assert Config.load("G1").requires_azure() is True
