# Resultados — memória fatgraph no LoCoMo

F1 é a métrica oficial do LoCoMo (token-level, `task_eval/evaluation.py`), reportada para **todas** as categorias, inclusive adversarial. Nada foi filtrado nem subamostrado.


## F1 por categoria

| condition | multi-hop | temporal | open-domain | single-hop | adversarial | macro | micro |
|---|---|---|---|---|---|---|---|
| L1-bipartite | 0.307 | 0.533 | 0.204 | 0.558 | 0.666 | 0.454 | 0.526 |
| L2-slots | 0.376 | 0.532 | 0.249 | 0.600 | 0.608 | 0.473 | 0.542 |
| L2d-derived | 0.430 | 0.509 | 0.281 | 0.607 | 0.242 | 0.414 | 0.469 |
| L5-conjunction | 0.432 | 0.507 | 0.263 | 0.600 | 0.231 | 0.407 | 0.462 |
| L6-bridges | 0.431 | 0.508 | 0.263 | 0.601 | 0.231 | 0.407 | 0.462 |

## Métrica: sobreposição de tokens vs juiz LLM

_(rode `fgl judge` para pontuar as predições com o juiz LLM)_

## Comparações-chave

_(rode as duas condições de cada par para ver os deltas)_

## Identidade dos grafos (as ablations isolam o que dizem isolar?)

_(rode a G1 para comparar as impressões digitais dos grafos)_

## Recall da recuperação (evidências anotadas)

| condition | recall@10 | recall@5 | recall_context |
|---|---|---|---|
| L1-bipartite | 0.271 | 0.211 | 0.614 |
| L2-slots | 0.214 | 0.209 | 0.770 |
| L2d-derived | 0.211 | 0.208 | 0.776 |
| L5-conjunction | 0.211 | 0.208 | 0.770 |
| L6-bridges | 0.211 | 0.208 | 0.770 |

## Expansão por sigma (auditoria)

`uso` = fração de perguntas em que a órbita de sigma contribuiu com pelo menos um fato. `evidência só via σ` = fração em que sigma alcançou um turno de evidência que nenhuma face alcançou — a contribuição marginal do salto, não só sua atividade.

_(nenhuma condição rodou com `retrieval.sigma_expand` — as colunas de auditoria estão ausentes/zeradas, como esperado para G1–G3)_

## Recuperação por cobertura de entidades (auditoria)

_(nenhuma condição rodou com `retrieval.face_coverage`)_

## Estatísticas do grafo

| condition | V | E | F | C | genus | max face | bigons | leaf faces | collapses | consolid. | incongr. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1-bipartite | 15193 | 21627 | 494 | 405 | 3375 | 4340 | 5 | 248 | 0 | 0 | 0 |
| L2-slots | 20978 | 68710 | 108 | 10 | 23822 | 11166 | 7 | 0 | 0 | 0 | 0 |
| L2d-derived | 21638 | 73789 | 97 | 10 | 26037 | 15824 | 7 | 0 | 0 | 0 | 0 |
| L5-conjunction | 21638 | 73789 | 97 | 10 | 26037 | 15824 | 7 | 0 | 0 | 0 | 0 |
| L6-bridges | 21640 | 73791 | 107 | 10 | 26032 | 14442 | 7 | 0 | 0 | 0 | 0 |

## Custo em tokens de LLM

| condition | calls | cached | tokens ingest | tokens QA | total | wall |
|---|---|---|---|---|---|---|
| L1-bipartite | 1986 | 100 | 0 | 6,386,599 | 6,386,599 | 6347s |
| L2-slots | 1986 | 102 | 0 | 6,794,302 | 6,794,302 | 6742s |
| L2d-derived | 1986 | 12 | 0 | 6,450,319 | 6,450,319 | 2963s |
| L5-conjunction | 1986 | 326 | 0 | 6,644,309 | 6,644,309 | 2366s |
| L6-bridges | 1986 | 1869 | 0 | 6,644,579 | 6,644,579 | 342s |
