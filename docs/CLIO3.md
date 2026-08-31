# CLIO3

CLIO3 is the open-schema alternative to CLIO2. It keeps the append-only raw
episode log and evidence contract, but replaces the fixed personal-dialogue
relation catalog on its write/read path with an event-centred hypergraph.

## Design rule

The language model proposes semantic organization; deterministic code owns
identity, persistence, temporal projection, graph mutation, resource limits,
and evidence verification.

## Write path

For each episode, one extraction call receives nearby entities and records. It
may reuse an entity, introduce a typed entity, and emit event/state/preference/
plan/fact records with open record types, participant roles, attributes,
temporal expressions, lifecycle operations, and links to prior records.

Evidence must be a verbatim substring of the current episode. Records are
append-oriented. Updates and retractions retain the superseded record and its
provenance.

## Graph

Entities are vertices. A memory record is a role-labelled hyperedge connecting
any number of entities; record-to-record links encode open relations such as a
causal, follow-up, or identity connection. The graph is not restricted to the
CLIO2 catalog and does not contain benchmark names or question templates.

## Read path

An LLM compiles the question into a small domain-independent query containing
an operation, answer type, focal entities, open concepts, temporal constraints,
and a bounded hop budget. Retrieval uses reciprocal-rank fusion of dense and
lexical rankings, then expands through shared participants and explicit record
links. Raw episodes remain a parallel recall channel.

The answer model selects values and raw episode ids. Code removes invented
support, projects typed dates, and abstains when no supplied episode supports
the answer.

## Benchmark

Run one conversation:

```bash
fgl clio bench --reader clio3 -n 1 --name CLIO3_CONV0 --no-cache
```

Run the complete benchmark:

```bash
fgl clio bench --reader clio3 --name CLIO3_FULL --no-cache
```
