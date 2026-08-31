"""Loads and validates Sigma, the relation catalog (spec section 4).

The catalog is a versioned YAML file, not something the LLM ever edits or
sees beyond a type-filtered read (spec 6.2c). Loading it validates two
things that would otherwise fail silently much later: every relation's
``dependents`` form a DAG (a cycle would loop forever in phase 5, spec 7.6),
and every type referenced by a signature is declared.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from fgl.clio.catalog.spec import RelationSpec


class CatalogError(ValueError):
    pass


class Catalog:
    """Sigma: relation name -> :class:`RelationSpec`, plus the declared types.

    Inverse labels (``employs`` for ``works_at``, spec 3.3) are known here
    but never get their own :class:`RelationSpec` or their own written
    edges -- see :func:`fgl.clio.graph.queries.out_edges` for why physically
    materialising a mirror edge per spec's literal wording was replaced with
    a query-time reversal: two independently-mutated copies of one fact can
    drift (a fold, an unfold, or a cardinality supersession on one side with
    nothing updating the other), and a query-time view cannot drift because
    there is nothing to keep in sync. A self-inverse relation (``inverse_name
    == name``, e.g. ``friend_of``) needs no separate bookkeeping at all: the
    same label read from the object's side already means the same thing.
    """

    def __init__(
        self,
        types: list[str],
        relations: dict[str, RelationSpec],
        type_classes: list[list[str]] | None = None,
    ):
        self.types = tuple(types)
        self._relations = relations
        #: type -> the set of types interchangeable with it. A type in no
        #: declared class is a class of its own.
        self._type_class: dict[str, frozenset[str]] = {}
        for group in type_classes or []:
            frozen = frozenset(group)
            for t in group:
                self._type_class[t] = frozen
        #: inverse label -> forward relation name, excluding self-inverse
        #: relations (inverse_name == name), which need no separate entry.
        self._inverse_to_forward: dict[str, str] = {
            spec.inverse_name: name
            for name, spec in relations.items()
            if spec.invertible and spec.inverse_name != name
        }
        self._surface_to_relation: dict[str, str] = {}
        for name, spec in relations.items():
            for alias in (name, *spec.aliases_surface):
                key = alias.strip().casefold().replace(" ", "_")
                previous = self._surface_to_relation.setdefault(key, name)
                if previous != name:
                    raise CatalogError(
                        f"surface alias {alias!r} belongs to both {previous!r} and {name!r}"
                    )

    def __contains__(self, name: str) -> bool:
        return name in self._relations

    def __getitem__(self, name: str) -> RelationSpec:
        try:
            return self._relations[name]
        except KeyError:
            raise CatalogError(f"unknown relation {name!r}") from None

    def get(self, name: str) -> RelationSpec | None:
        return self._relations.get(name)

    def __iter__(self) -> Iterable[str]:
        return iter(self._relations)

    def names(self) -> list[str]:
        return list(self._relations)

    def filter_by_types(self, types_present: set[str]) -> list[RelationSpec]:
        """Relations whose subject type is among ``types_present``.

        Used to shrink the catalog handed to the extractor (spec 6.2c) to
        what could plausibly apply to the entities already anchored in a
        turn. Not used by consolidation itself.
        """
        return [r for r in self._relations.values() if r.signature[0] in types_present]

    def inverse_of(self, name: str) -> str | None:
        spec = self[name]
        return spec.inverse_name if spec.invertible else None

    def is_inverse_label(self, label: str) -> bool:
        """True for a genuine inverse label (``employs``), false for a
        forward relation -- including a self-inverse one (``friend_of``),
        which IS a forward relation and is looked up as one."""
        return label in self._inverse_to_forward

    def forward_of(self, inverse_label: str) -> str:
        try:
            return self._inverse_to_forward[inverse_label]
        except KeyError:
            raise CatalogError(f"{inverse_label!r} is not a known inverse label") from None

    def type_class(self, type_: str) -> frozenset[str]:
        """Every type interchangeable with ``type_``, including itself.

        Phase 1 assigns a brand-new entity the type of the relation's own
        signature slot, so one thing named once can be reached under two
        different types (``practices`` -> Activity, ``attended`` ->
        Event). Without a declared class, that mints two vertices and fold
        can never merge them: ``identity_score`` returns 0.0 on a type
        mismatch before it looks at anything else.
        """
        return self._type_class.get(type_, frozenset({type_}))

    def types_compatible(self, a: str, b: str) -> bool:
        """True when two types name the same kind of thing (spec's own
        signature check, widened by the catalog's ``type_classes``)."""
        return a == b or b in self.type_class(a)

    def is_known(self, label: str) -> bool:
        """A label the access algebra's ``follow`` can walk: a relation
        name, or one of its inverses."""
        return label in self._relations or label in self._inverse_to_forward

    def canonical_relation(self, surface: str | None) -> str | None:
        """Map a declared surface/proposal alias to its canonical relation."""
        if not surface:
            return None
        key = surface.strip().casefold().replace(" ", "_")
        return self._surface_to_relation.get(key)


def _build_relation(name: str, raw: dict) -> RelationSpec:
    signature = raw.get("signature")
    if not signature or len(signature) != 2:
        raise CatalogError(f"relation {name!r}: signature must be [Subject, Object]")
    cardinality = raw.get("cardinality")
    if cardinality not in ("functional", "multi"):
        raise CatalogError(f"relation {name!r}: cardinality must be functional|multi")
    volatility = raw.get("volatility")
    if volatility not in ("static", "slow", "fast"):
        raise CatalogError(f"relation {name!r}: volatility must be static|slow|fast")
    invertible = bool(raw.get("invertible", False))
    inverse_name = raw.get("inverse_name")
    if invertible and not inverse_name:
        raise CatalogError(f"relation {name!r}: invertible=true needs inverse_name")
    return RelationSpec(
        name=name,
        signature=(signature[0], signature[1]),
        cardinality=cardinality,
        volatility=volatility,
        invertible=invertible,
        inverse_name=inverse_name,
        default_duration_days=raw.get("default_duration"),
        closes_on_new=bool(raw.get("closes_on_new", True)),
        dependents=tuple(raw.get("dependents") or ()),
        aliases_surface=tuple(raw.get("aliases_surface") or ()),
    )


def _check_dag(relations: dict[str, RelationSpec]) -> None:
    """dependents must form a DAG; a cycle would spin phase 5 forever."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {name: WHITE for name in relations}

    def visit(name: str, path: list[str]) -> None:
        color[name] = GRAY
        for dep in relations[name].dependents:
            if dep not in relations:
                raise CatalogError(f"relation {name!r} depends on unknown relation {dep!r}")
            if color[dep] == GRAY:
                cycle = " -> ".join([*path, name, dep])
                raise CatalogError(f"cycle in Sigma.dependents: {cycle}")
            if color[dep] == WHITE:
                visit(dep, [*path, name])
        color[name] = BLACK

    for name in relations:
        if color[name] == WHITE:
            visit(name, [])


def load_catalog(path: str | Path) -> Catalog:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    types = raw.get("types") or []
    if not types:
        raise CatalogError(f"{p}: no 'types' declared")

    raw_relations = raw.get("relations") or []
    relations: dict[str, RelationSpec] = {}
    for entry in raw_relations:
        name = entry.get("name")
        if not name:
            raise CatalogError(f"{p}: relation entry with no 'name'")
        if name in relations:
            raise CatalogError(f"{p}: relation {name!r} declared twice")
        relations[name] = _build_relation(name, entry)

    for spec in relations.values():
        for t in spec.signature:
            if t not in types:
                raise CatalogError(
                    f"relation {spec.name!r}: type {t!r} not in declared types {types}"
                )

    type_classes = raw.get("type_classes") or []
    for group in type_classes:
        if len(group) < 2:
            raise CatalogError(f"{p}: type_class {group!r} needs at least two types")
        for t in group:
            if t not in types:
                raise CatalogError(f"{p}: type_class {group!r} names undeclared type {t!r}")
    seen_in_class: set[str] = set()
    for group in type_classes:
        overlap = seen_in_class & set(group)
        if overlap:
            raise CatalogError(
                f"{p}: type(s) {sorted(overlap)} appear in more than one type_class; "
                "classes must partition the types they cover"
            )
        seen_in_class |= set(group)

    _check_dag(relations)
    return Catalog(types, relations, type_classes)
