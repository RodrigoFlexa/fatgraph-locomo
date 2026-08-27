# Resultados — memória fatgraph no LoCoMo

F1 é a métrica oficial do LoCoMo (token-level, `task_eval/evaluation.py`), reportada para **todas** as categorias, inclusive adversarial. Nada foi filtrado nem subamostrado.


## F1 por categoria

| condition | multi-hop | temporal | open-domain | single-hop | adversarial | macro | micro |
|---|---|---|---|---|---|---|---|
| L1-bipartite | 0.305 | 0.526 | 0.210 | 0.551 | 0.695 | 0.457 | 0.528 |
| L2-slots | 0.309 | 0.524 | 0.258 | 0.598 | 0.648 | 0.467 | 0.540 |

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
| L2-slots | 0.214 | 0.209 | 0.773 |

## Expansão por sigma (auditoria)

`uso` = fração de perguntas em que a órbita de sigma contribuiu com pelo menos um fato. `evidência só via σ` = fração em que sigma alcançou um turno de evidência que nenhuma face alcançou — a contribuição marginal do salto, não só sua atividade.

_(nenhuma condição rodou com `retrieval.sigma_expand` — as colunas de auditoria estão ausentes/zeradas, como esperado para G1–G3)_

## Recuperação por cobertura de entidades (auditoria)

_(nenhuma condição rodou com `retrieval.face_coverage`)_

## Estatísticas do grafo

| condition | V | E | F | C | genus | max face | bigons | leaf faces | collapses | consolid. | incongr. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1-bipartite | 15193 | 21627 | 494 | 405 | 3375 | 4340 | 5 | 248 | 0 | 0 | 0 |
| L2-slots | 20978 | 68710 | 104 | 10 | 23824 | 13880 | 7 | 0 | 0 | 0 | 0 |

## Custo em tokens de LLM

| condition | calls | cached | tokens ingest | tokens QA | total | wall |
|---|---|---|---|---|---|---|
| L1-bipartite | 1986 | 86 | 0 | 6,148,907 | 6,148,907 | 5759s |
| L2-slots | 1986 | 12 | 0 | 6,546,653 | 6,546,653 | 6506s |
