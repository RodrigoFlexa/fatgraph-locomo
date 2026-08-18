# Resultados — memória fatgraph no LoCoMo

F1 é a métrica oficial do LoCoMo (token-level, `task_eval/evaluation.py`), reportada para **todas** as categorias, inclusive adversarial. Nada foi filtrado nem subamostrado.


> ## ⚠️ Corrida suspeita — não interprete estes números
>
> **G10-face-units**
>   - o grafo é quase uma ESTRELA (81% dos vértices têm grau 1, 45% das meias-arestas estão nos dois maiores vértices, 51.1 faces não triviais em média) — G4/G5/G6 sobre estes grafos vão reproduzir a G1. A causa está no INGEST, não na recuperação: os fatos estão sendo ancorados em quem falou, não no que foi dito. Confira a extração e a resolução de entidades antes de interpretar qualquer número.
>
> Diagnostique com `fgl doctor`.

## F1 por categoria

| condition | multi-hop | temporal | open-domain | single-hop | adversarial | macro | micro |
|---|---|---|---|---|---|---|---|
| B3-rag-facts | 0.232 | 0.431 | 0.221 | 0.384 | 0.854 | 0.424 | 0.468 |
| G10-face-units | 0.174 | 0.414 | 0.185 | 0.348 | 0.877 | 0.400 | 0.445 |

## Métrica: sobreposição de tokens vs juiz LLM

_(rode `fgl judge` para pontuar as predições com o juiz LLM)_

## Comparações-chave

| comparação | isola | F1 base | F1 novo | Δ | Δ% |
|---|---|---|---|---|---|
| G10-face-units − B3-rag-facts | FACE COMO UNIDADE vs k-NN puro (o alvo) | 0.468 | 0.445 | -0.023 | -4.9% |

## Identidade dos grafos (as ablations isolam o que dizem isolar?)

_(rode a G1 para comparar as impressões digitais dos grafos)_

## Recall da recuperação (evidências anotadas)

| condition | recall@10 | recall@5 | recall_context |
|---|---|---|---|
| B3-rag-facts | 0.562 | 0.474 | 0.571 |
| G10-face-units | 0.570 | 0.480 | 0.460 |

## Expansão por sigma (auditoria)

`uso` = fração de perguntas em que a órbita de sigma contribuiu com pelo menos um fato. `evidência só via σ` = fração em que sigma alcançou um turno de evidência que nenhuma face alcançou — a contribuição marginal do salto, não só sua atividade.

_(nenhuma condição rodou com `retrieval.sigma_expand` — as colunas de auditoria estão ausentes/zeradas, como esperado para G1–G3)_

## Recuperação por cobertura de entidades (auditoria)

_(nenhuma condição rodou com `retrieval.face_coverage`)_

## Estatísticas do grafo

| condition | V | E | F | C | genus | max face | bigons | leaf faces | collapses | consolid. | incongr. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G10-face-units | 2982 | 3711 | 511 | 101 | 210 | 141 | 122 | 69 | 0 | 0 | 11 |

## Custo em tokens de LLM

| condition | calls | cached | tokens ingest | tokens QA | total | wall |
|---|---|---|---|---|---|---|
| B3-rag-facts | 1986 | 1890 | 0 | 1,194,190 | 1,194,190 | 250s |
| G10-face-units | 4338 | 2364 | 780,864 | 2,274,288 | 3,055,152 | 4702s |
