# CLIO2: compiled long-term memory

CLIO2 treats long-memory question answering as query compilation over an
evidence ledger. It does not treat the graph as a space in which an agent should
wander until it happens to encounter an answer.

## Design thesis

The immutable episode log is the record of what was said. Bitemporal graph facts
are normalized claims derived from that log. Events and indexes are rebuildable
views. A question is compiled into a typed query, executed deterministically,
and rendered only from selected evidence.

```text
immutable episodes
       +
bitemporal propositions
       |
       v
semantic ledger ----> event view + hybrid fact index
       |                         |
       +----------+--------------+
                  v
          typed query plan
                  v
      deterministic query algebra
       (filter / join / intersect /
        count / latest / temporal)
                  v
        evidence-bounded answer
                  v
              verifier
```

The graph remains valuable for identity, relations, consolidation, provenance,
and temporal validity. It is a derived representation, not the control flow of
the reader.

## Invariants

1. The episode log and stored propositions are never rewritten by CLIO2.
2. The semantic ledger is rebuildable; invalidating it cannot lose memory.
3. Planning interprets language but never computes an answer.
4. Counting and intersection are code-level algebra, not language-model guesses.
5. Every returned value is checked against selected episode evidence.
6. First-person claims are attributed to their speaker at read time. This repairs
   extraction mistakes and collective subjects without changing source data.
7. Raw episode retrieval is a bounded recall backstop, not permission to ignore
   the structured ledger.

## Components

- `src/fgl/clio2/model.py`: typed intermediate representation and answer contract.
- `src/fgl/clio2/ledger.py`: fact/event materialization and hybrid index.
- `src/fgl/clio2/planner.py`: deterministic and model-assisted query compiler.
- `src/fgl/clio2/executor.py`: structural filters and deterministic algebra.
- `src/fgl/clio2/answer.py`: bounded answer generation and verification.
- `src/fgl/clio2/engine.py`: runtime cache and compatibility trace.

Schema compatibility is deliberate. A current `created` intent may be stored in
an older memory as `owns(artwork)` plus `practices(painting)`. CLIO2 expands the
compatible relations and joins the event roles within one episode; it does not
equate arbitrary owned objects with created objects.

## Running it

CLIO2 is the default reader of the CLIO benchmark command:

```bash
fgl clio bench -n 1 --name CLIO2_CONV0
```

For an explicit comparison with the original movement agent:

```bash
fgl clio bench -n 1 --reader agent --name CLIO_AGENT_CONV0
fgl clio bench -n 1 --reader clio2 --name CLIO2_CONV0
```

To evaluate a reader change against an existing snapshot without ingesting the
419 turns again:

```bash
fgl clio bench -n 1 --reader clio2 \
  --memory results/CLIO2_CONV0/memory_conv-26.json \
  --name CLIO2_TEMPORAL_REPLAY
```

Programmatically, `clio.ask2(question)` always selects CLIO2. The regular
`clio.ask(question)` follows `ClioConfig.reader`.

## Evaluation discipline

Overall F1 is not enough. Compare per category, especially multi-hop and
single-hop, and inspect evidence recall separately from answer F1. CLIO2 traces
have three stable phases (`compile`, `execute`, `verify`), so a failure can be
assigned to interpretation, ledger recall/algebra, or answer verification.
