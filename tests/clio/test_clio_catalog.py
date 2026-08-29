"""Sigma loading and validation (spec section 4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fgl.clio.catalog import CatalogError, load_catalog
from fgl.clio.config import ClioConfig

CATALOG_PATH = ClioConfig.default().catalog_path


def test_personal_dialogue_catalog_loads():
    cat = load_catalog(CATALOG_PATH)
    assert "works_at" in cat
    assert "lives_in" in cat
    assert "Person" in cat.types


def test_works_at_declares_its_dependents():
    cat = load_catalog(CATALOG_PATH)
    spec = cat["works_at"]
    assert spec.cardinality == "functional"
    assert spec.volatility == "slow"
    assert set(spec.dependents) == {"managed_by", "works_with", "has_role"}


def test_invertible_relation_requires_inverse_name():
    cat = load_catalog(CATALOG_PATH)
    assert cat["works_at"].invertible
    assert cat.inverse_of("works_at") == "employs"
    assert cat.inverse_of("born_in") is None  # not invertible
    assert cat.inverse_of("lives_in") == "resided_by"


def test_unknown_relation_raises():
    cat = load_catalog(CATALOG_PATH)
    with pytest.raises(CatalogError):
        cat["does_not_exist"]


def test_filter_by_types():
    cat = load_catalog(CATALOG_PATH)
    person_only = cat.filter_by_types({"Person"})
    assert all(r.signature[0] == "Person" for r in person_only)
    assert any(r.name == "works_at" for r in person_only)


def _write(tmp_path: Path, doc: dict) -> Path:
    p = tmp_path / "catalog.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_cycle_in_dependents_is_rejected(tmp_path):
    doc = {
        "types": ["Person", "Organization"],
        "relations": [
            {
                "name": "a",
                "signature": ["Person", "Organization"],
                "cardinality": "functional",
                "volatility": "slow",
                "dependents": ["b"],
            },
            {
                "name": "b",
                "signature": ["Person", "Organization"],
                "cardinality": "functional",
                "volatility": "slow",
                "dependents": ["a"],
            },
        ],
    }
    with pytest.raises(CatalogError, match="cycle"):
        load_catalog(_write(tmp_path, doc))


def test_unknown_type_in_signature_is_rejected(tmp_path):
    doc = {
        "types": ["Person"],
        "relations": [
            {
                "name": "a",
                "signature": ["Person", "Ghost"],
                "cardinality": "multi",
                "volatility": "static",
            },
        ],
    }
    with pytest.raises(CatalogError, match="Ghost"):
        load_catalog(_write(tmp_path, doc))


def test_unknown_dependent_is_rejected(tmp_path):
    doc = {
        "types": ["Person", "Organization"],
        "relations": [
            {
                "name": "a",
                "signature": ["Person", "Organization"],
                "cardinality": "functional",
                "volatility": "slow",
                "dependents": ["ghost_relation"],
            },
        ],
    }
    with pytest.raises(CatalogError, match="ghost_relation"):
        load_catalog(_write(tmp_path, doc))


def test_invertible_without_inverse_name_is_rejected(tmp_path):
    doc = {
        "types": ["Person", "Organization"],
        "relations": [
            {
                "name": "a",
                "signature": ["Person", "Organization"],
                "cardinality": "functional",
                "volatility": "slow",
                "invertible": True,
            },
        ],
    }
    with pytest.raises(CatalogError):
        load_catalog(_write(tmp_path, doc))


def test_duplicate_relation_name_is_rejected(tmp_path):
    doc = {
        "types": ["Person", "Organization"],
        "relations": [
            {
                "name": "a",
                "signature": ["Person", "Organization"],
                "cardinality": "multi",
                "volatility": "static",
            },
            {
                "name": "a",
                "signature": ["Person", "Organization"],
                "cardinality": "multi",
                "volatility": "static",
            },
        ],
    }
    with pytest.raises(CatalogError, match="twice"):
        load_catalog(_write(tmp_path, doc))
