"""Passo 1: top-k vs cobertura vs MMR, mesmo grafo, mesmo orçamento.

Um grafo e um retriever por conversa; os braços são apenas um atributo de
config trocado entre chamadas de `retrieve`, então a unica coisa que difere
entre eles e a POLITICA DE SELECAO. Registra por pergunta para permitir cortes
por numero de pecas de evidencia e por dispersao entre sessoes -- que e onde a
hipotese vive.
"""
import json, sys, time
from collections import defaultdict

from fgl.config import Config
from fgl.data.locomo import load_conversations
from fgl.paths import Paths
from fgl.pipeline import Runner
from fgl.llm import build_llm
from fgl.retrieval.slots import SlotRetriever
from fgl.evaluation.scorer import evidence_recall

ARMS = [
    ("topk",         dict(selection="topk")),
    ("mmr-0.7",      dict(selection="mmr", mmr_lambda=0.7)),
    ("cover-0.7",    dict(selection="coverage", coverage_decay=0.7)),
    ("cover-0.5",    dict(selection="coverage", coverage_decay=0.5)),
    ("cover-0.3",    dict(selection="coverage", coverage_decay=0.3)),
    ("cover-0.0",    dict(selection="coverage", coverage_decay=0.0)),
    ("cover-0.5-nopair", dict(selection="coverage", coverage_decay=0.5,
                             coverage_keep_pairs=False)),
]

n_convs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
paths = Paths.build()
convs = load_conversations(paths.locomo_file)[:n_convs]
cfg = Config.load("L2d")
cfg.llm.provider = "fake"; cfg.llm.cache_enabled = False
runner = Runner(cfg, llm=build_llm(cfg.llm))

rows = []
t0 = time.time()
for ci, conv in enumerate(convs):
    graph, _ = runner._ingest(conv)
    retr = SlotRetriever(graph, runner.embedder, cfg,
                         {s.id: s.date_time_raw for s in conv.sessions},
                         question_corpus=[q.prompt_question() for q in conv.questions])
    for q in conv.questions:
        ev = list(q.evidence or [])
        base = dict(conv=conv.sample_id, cat=q.category_name,
                    n_ev=len(ev), n_sess=len({t.split(':')[0] for t in ev}))
        for name, knobs in ARMS:
            for k, v in knobs.items():
                setattr(cfg.slots, k, v)
            cfg.slots.coverage_keep_pairs = knobs.get("coverage_keep_pairs", True)
            r = retr.retrieve(q.prompt_question())
            base[name] = dict(recall=evidence_recall(ev, r.turn_ids),
                              tokens=r.tokens_used, units=len(r.facts))
        rows.append(base)
    print(f"[{ci+1}/{len(convs)}] {conv.sample_id}  {len(rows)} perguntas  "
          f"{time.time()-t0:.0f}s", flush=True)

json.dump(rows, open("/tmp/coverage_rows.json", "w"))
print("gravado /tmp/coverage_rows.json", len(rows))
