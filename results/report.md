# Resultados — memória fatgraph no LoCoMo

F1 é a métrica oficial do LoCoMo (token-level, `task_eval/evaluation.py`), reportada para **todas** as categorias, inclusive adversarial. Nada foi filtrado nem subamostrado.

## F1 por categoria

| condition | multi-hop | temporal | open-domain | single-hop | adversarial | macro | micro |
|---|---|---|---|---|---|---|---|
| B1-full-context | 0.005 | 0.014 | 0.023 | 0.010 | 1.000 | 0.210 | 0.233 |
| B2-rag-turns | 0.005 | 0.014 | 0.023 | 0.010 | 1.000 | 0.210 | 0.233 |
| B3-rag-facts | 0.005 | 0.014 | 0.023 | 0.010 | 1.000 | 0.210 | 0.233 |
| G1-fatgraph-min | 0.005 | 0.014 | 0.023 | 0.010 | 1.000 | 0.210 | 0.233 |
| G2-fatgraph-cur | 0.005 | 0.014 | 0.023 | 0.010 | 1.000 | 0.210 | 0.233 |
| G3-fatgraph-agent | 0.005 | 0.014 | 0.023 | 0.010 | 1.000 | 0.210 | 0.233 |

## Comparações-chave

| comparação | isola | F1 base | F1 novo | Δ | Δ% |
|---|---|---|---|---|---|
| G1-fatgraph-min − B3-rag-facts | valor das faces (mesmos fatos) | 0.233 | 0.233 | +0.000 | +0.0% |
| G2-fatgraph-cur − G1-fatgraph-min | valor da curadoria + consolidação | 0.233 | 0.233 | +0.000 | +0.0% |
| G3-fatgraph-agent − G2-fatgraph-cur | valor do sigma-agent | 0.233 | 0.233 | +0.000 | +0.0% |

## Recall da recuperação (evidências anotadas)

| condition | recall@10 | recall@5 | recall_context |
|---|---|---|---|
| B1-full-context | 0.036 | 0.022 | 0.992 |
| B2-rag-turns | 0.384 | 0.297 | 0.384 |
| B3-rag-facts | 0.131 | 0.119 | 0.133 |
| G1-fatgraph-min | 0.133 | 0.118 | 0.132 |
| G2-fatgraph-cur | 0.133 | 0.118 | 0.133 |
| G3-fatgraph-agent | 0.132 | 0.118 | 0.129 |

## Estatísticas do grafo

| condition | V | E | F | C | genus | max face | bigons | leaf faces | collapses | consolid. | incongr. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1-fatgraph-min | 590 | 682 | 82 | 39 | 44 | 159 | 19 | 16 | 0 | 0 | 0 |
| G2-fatgraph-cur | 590 | 692 | 86 | 39 | 47 | 159 | 20 | 16 | 3 | 13 | 0 |
| G3-fatgraph-agent | 593 | 694 | 93 | 40 | 44 | 219 | 31 | 16 | 3 | 15 | 0 |

## Custo em tokens de LLM

| condition | calls | cached | tokens ingest | tokens QA | total | wall |
|---|---|---|---|---|---|---|
| B1-full-context | 1986 | 12 | 0 | 45,380,322 | 45,380,322 | 3754s |
| B2-rag-turns | 1986 | 12 | 0 | 1,383,123 | 1,383,123 | 2877s |
| B3-rag-facts | 1986 | 12 | 0 | 1,048,840 | 1,048,840 | 2777s |
| G1-fatgraph-min | 2288 | 211 | 131,452 | 2,413,671 | 2,545,123 | 3470s |
| G2-fatgraph-cur | 2795 | 1504 | 631,343 | 2,457,385 | 3,088,728 | 2133s |
| G3-fatgraph-agent | 3386 | 685 | 1,308,519 | 2,400,556 | 3,709,075 | 5855s |
