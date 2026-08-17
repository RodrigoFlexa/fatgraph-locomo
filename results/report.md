# Resultados — memória fatgraph no LoCoMo

F1 é a métrica oficial do LoCoMo (token-level, `task_eval/evaluation.py`), reportada para **todas** as categorias, inclusive adversarial. Nada foi filtrado nem subamostrado.


> ## ⚠️ Corrida suspeita — não interprete estes números
>
> **G4-fatgraph-sigma**
>   - o grafo é quase uma ESTRELA (81% dos vértices têm grau 1, 45% das meias-arestas estão nos dois maiores vértices, 21.3 faces não triviais em média) — sigma é redundante com phi e a cobertura por face não discrimina: esta condição vai reproduzir a G1 e o delta que ela deveria medir não existe neste grafo. A causa está no INGEST, não na recuperação: os fatos estão sendo ancorados em quem falou, não no que foi dito. Confira a extração e a resolução de entidades antes de interpretar qualquer número.
> **G5-fatgraph-coverage**
>   - o grafo é quase uma ESTRELA (81% dos vértices têm grau 1, 45% das meias-arestas estão nos dois maiores vértices, 21.3 faces não triviais em média) — sigma é redundante com phi e a cobertura por face não discrimina: esta condição vai reproduzir a G1 e o delta que ela deveria medir não existe neste grafo. A causa está no INGEST, não na recuperação: os fatos estão sendo ancorados em quem falou, não no que foi dito. Confira a extração e a resolução de entidades antes de interpretar qualquer número.
> **G6-fatgraph-join**
>   - o grafo é quase uma ESTRELA (81% dos vértices têm grau 1, 45% das meias-arestas estão nos dois maiores vértices, 21.3 faces não triviais em média) — sigma é redundante com phi e a cobertura por face não discrimina: esta condição vai reproduzir a G1 e o delta que ela deveria medir não existe neste grafo. A causa está no INGEST, não na recuperação: os fatos estão sendo ancorados em quem falou, não no que foi dito. Confira a extração e a resolução de entidades antes de interpretar qualquer número.
>   - apenas 16/1986 perguntas usaram a expansão por sigma: confira sigma_expand_k e sigma_budget_frac
>
> Diagnostique com `fgl doctor`.

## F1 por categoria

| condition | multi-hop | temporal | open-domain | single-hop | adversarial | macro | micro |
|---|---|---|---|---|---|---|---|
| B1-full-context | 0.392 | 0.426 | 0.069 | 0.653 | 0.630 | 0.434 | 0.546 |
| B2-rag-turns | 0.177 | 0.330 | 0.042 | 0.366 | 0.729 | 0.329 | 0.399 |
| B3-rag-facts | 0.232 | 0.431 | 0.082 | 0.384 | 0.854 | 0.397 | 0.461 |
| G1-fatgraph-min | 0.142 | 0.386 | 0.052 | 0.316 | 0.895 | 0.358 | 0.420 |
| G2-fatgraph-cur | 0.138 | 0.330 | 0.058 | 0.304 | 0.904 | 0.347 | 0.408 |
| G3-fatgraph-agent | 0.147 | 0.346 | 0.064 | 0.298 | 0.888 | 0.348 | 0.405 |
| G4-fatgraph-sigma | 0.190 | 0.428 | 0.062 | 0.355 | 0.872 | 0.381 | 0.445 |
| G5-fatgraph-coverage | 0.150 | 0.374 | 0.065 | 0.300 | 0.899 | 0.358 | 0.414 |
| G6-fatgraph-join | 0.167 | 0.393 | 0.071 | 0.343 | 0.908 | 0.376 | 0.440 |

## Comparações-chave

| comparação | isola | F1 base | F1 novo | Δ | Δ% |
|---|---|---|---|---|---|
| G1-fatgraph-min − B3-rag-facts | valor das faces (mesmos fatos) | 0.461 | 0.420 | -0.041 | -8.9% |
| G2-fatgraph-cur − G1-fatgraph-min | valor da curadoria + consolidação | 0.420 | 0.408 | -0.012 | -2.9% |
| G3-fatgraph-agent − G2-fatgraph-cur | valor do sigma-agent | 0.408 | 0.405 | -0.002 | -0.6% |
| G4-fatgraph-sigma − G1-fatgraph-min | valor da expansão por sigma (mesmo grafo) | 0.420 | 0.445 | +0.025 | +6.0% |
| G5-fatgraph-coverage − G1-fatgraph-min | valor da cobertura de entidades | 0.420 | 0.414 | -0.006 | -1.4% |
| G6-fatgraph-join − G4-fatgraph-sigma | cobertura ADICIONADA a sigma | 0.445 | 0.440 | -0.006 | -1.2% |
| G6-fatgraph-join − G5-fatgraph-coverage | sigma ADICIONADO à cobertura | 0.414 | 0.440 | +0.026 | +6.2% |

## Recall da recuperação (evidências anotadas)

| condition | recall@10 | recall@5 | recall_context | recall_context_anchors_only | recall_context_no_coverage | recall_context_no_sigma |
|---|---|---|---|---|---|---|
| B1-full-context | 0.036 | 0.022 | 0.992 | - | - | - |
| B2-rag-turns | 0.384 | 0.297 | 0.384 | - | - | - |
| B3-rag-facts | 0.562 | 0.474 | 0.571 | - | - | - |
| G1-fatgraph-min | 0.570 | 0.480 | 0.423 | - | - | - |
| G2-fatgraph-cur | 0.573 | 0.482 | 0.415 | - | - | - |
| G3-fatgraph-agent | 0.571 | 0.479 | 0.422 | - | - | - |
| G4-fatgraph-sigma | 0.570 | 0.480 | 0.492 | - | - | 0.415 |
| G5-fatgraph-coverage | 0.570 | 0.480 | 0.416 | - | 0.068 | - |
| G6-fatgraph-join | 0.570 | 0.480 | 0.481 | 0.065 | 0.065 | 0.481 |

## Expansão por sigma (auditoria)

`uso` = fração de perguntas em que a órbita de sigma contribuiu com pelo menos um fato. `evidência só via σ` = fração em que sigma alcançou um turno de evidência que nenhuma face alcançou — a contribuição marginal do salto, não só sua atividade.

| condition | uso | fatos σ | pontes | tokens σ | evidência só via σ | uso (multi-hop) | recall MH sem σ | recall MH com σ |
|---|---|---|---|---|---|---|---|---|
| G4-fatgraph-sigma | 0.972 | 4.22 | 1.45 | 48 | 0.109 | 0.975 | 0.281 | 0.391 |
| G6-fatgraph-join | 0.008 | 0.02 | 1.28 | 37 | 0.000 | 0.007 | 0.343 | 0.343 |

## Recuperação por cobertura de entidades (auditoria)

| condition | ligadas | entidades | cobertura máx | faces-ponte | uso | geodésica | evidência só via cobertura | recall MH sem cob. | recall MH com cob. |
|---|---|---|---|---|---|---|---|---|---|
| G5-fatgraph-coverage | 0.995 | 1.88 | 0.955 | 0.568 | 0.995 | 0.000 | 0.440 | 0.067 | 0.275 |
| G6-fatgraph-join | 0.995 | 1.88 | 0.955 | 0.568 | 0.995 | 0.000 | 0.513 | 0.068 | 0.343 |

## Estatísticas do grafo

| condition | V | E | F | C | genus | max face | bigons | leaf faces | collapses | consolid. | incongr. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G1-fatgraph-min | 2982 | 3711 | 213 | 101 | 359 | 719 | 44 | 69 | 0 | 0 | 11 |
| G2-fatgraph-cur | 2984 | 3786 | 198 | 102 | 404 | 661 | 37 | 71 | 37 | 112 | 11 |
| G3-fatgraph-agent | 2987 | 3781 | 206 | 105 | 399 | 784 | 44 | 73 | 47 | 117 | 11 |
| G4-fatgraph-sigma | 2982 | 3711 | 213 | 101 | 359 | 719 | 44 | 69 | 0 | 0 | 11 |
| G5-fatgraph-coverage | 2982 | 3711 | 213 | 101 | 359 | 719 | 44 | 69 | 0 | 0 | 11 |
| G6-fatgraph-join | 2982 | 3711 | 213 | 101 | 359 | 719 | 44 | 69 | 0 | 0 | 11 |

## Custo em tokens de LLM

| condition | calls | cached | tokens ingest | tokens QA | total | wall |
|---|---|---|---|---|---|---|
| B1-full-context | 1986 | 12 | 0 | 45,578,405 | 45,578,405 | 6339s |
| B2-rag-turns | 1986 | 12 | 0 | 1,535,641 | 1,535,641 | 4593s |
| B3-rag-facts | 1986 | 12 | 0 | 1,175,545 | 1,175,545 | 4515s |
| G1-fatgraph-min | 4185 | 211 | 729,653 | 2,749,929 | 3,479,582 | 8175s |
| G2-fatgraph-cur | 4535 | 2422 | 931,719 | 2,831,307 | 3,763,026 | 4903s |
| G3-fatgraph-agent | 8448 | 2107 | 4,552,711 | 2,822,229 | 7,374,940 | 18174s |
| G4-fatgraph-sigma | 1986 | 67 | 0 | 2,775,330 | 2,775,330 | 4472s |
| G5-fatgraph-coverage | 1986 | 663 | 0 | 2,755,004 | 2,755,004 | 3208s |
| G6-fatgraph-join | 1986 | 22 | 0 | 2,775,976 | 2,775,976 | 5213s |
