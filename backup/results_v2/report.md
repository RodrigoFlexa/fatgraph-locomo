# Resultados — memória fatgraph no LoCoMo

F1 é a métrica oficial do LoCoMo (token-level, `task_eval/evaluation.py`), reportada para **todas** as categorias, inclusive adversarial. Nada foi filtrado nem subamostrado.


> ## ⚠️ Corrida suspeita — não interprete estes números
>
> **G4-fatgraph-sigma**
>   - o grafo é quase uma ESTRELA (81% dos vértices têm grau 1, 45% das meias-arestas estão nos dois maiores vértices, 21.3 faces não triviais em média) — sigma é redundante com phi e a cobertura por face não discrimina: esta condição vai reproduzir a G1 e o delta que ela deveria medir não existe neste grafo. A causa está no INGEST, não na recuperação: os fatos estão sendo ancorados em quem falou, não no que foi dito. Confira a extração e a resolução de entidades antes de interpretar qualquer número.
> **G8-shuffled**
>   - o grafo é quase uma ESTRELA (81% dos vértices têm grau 1, 45% das meias-arestas estão nos dois maiores vértices, 21.3 faces não triviais em média) — sigma é redundante com phi e a cobertura por face não discrimina: esta condição vai reproduzir a G1 e o delta que ela deveria medir não existe neste grafo. A causa está no INGEST, não na recuperação: os fatos estão sendo ancorados em quem falou, não no que foi dito. Confira a extração e a resolução de entidades antes de interpretar qualquer número.
> **G9-genus**
>   - o grafo é quase uma ESTRELA (81% dos vértices têm grau 1, 45% das meias-arestas estão nos dois maiores vértices, 51.1 faces não triviais em média) — G4/G5/G6 sobre estes grafos vão reproduzir a G1. A causa está no INGEST, não na recuperação: os fatos estão sendo ancorados em quem falou, não no que foi dito. Confira a extração e a resolução de entidades antes de interpretar qualquer número.
>
> Diagnostique com `fgl doctor`.

## F1 por categoria

| condition | multi-hop | temporal | open-domain | single-hop | adversarial | macro | micro |
|---|---|---|---|---|---|---|---|
| G4-fatgraph-sigma | 0.190 | 0.428 | 0.208 | 0.355 | 0.872 | 0.411 | 0.452 |
| G8-shuffled | 0.195 | 0.393 | 0.170 | 0.343 | 0.888 | 0.398 | 0.444 |
| G9-genus | 0.196 | 0.421 | 0.194 | 0.365 | 0.857 | 0.406 | 0.452 |

## Comparações-chave

| comparação | isola | F1 base | F1 novo | Δ | Δ% |
|---|---|---|---|---|---|
| G8-shuffled − G4-fatgraph-sigma | A ORDEM IMPORTA? (mesmo conteúdo, permutado) | 0.452 | 0.444 | -0.008 | -1.8% |

## Identidade dos grafos (as ablations isolam o que dizem isolar?)

_(rode a G1 para comparar as impressões digitais dos grafos)_

## Recall da recuperação (evidências anotadas)

| condition | recall@10 | recall@5 | recall_context | recall_context_no_sigma |
|---|---|---|---|---|
| G4-fatgraph-sigma | 0.570 | 0.480 | 0.492 | 0.415 |
| G8-shuffled | 0.570 | 0.480 | 0.492 | 0.415 |
| G9-genus | 0.570 | 0.480 | 0.505 | - |

## Expansão por sigma (auditoria)

`uso` = fração de perguntas em que a órbita de sigma contribuiu com pelo menos um fato. `evidência só via σ` = fração em que sigma alcançou um turno de evidência que nenhuma face alcançou — a contribuição marginal do salto, não só sua atividade.

| condition | uso | fatos σ | pontes | tokens σ | evidência só via σ | uso (multi-hop) | recall MH sem σ | recall MH com σ |
|---|---|---|---|---|---|---|---|---|
| G4-fatgraph-sigma | 0.972 | 4.22 | 1.45 | 48 | 0.109 | 0.975 | 0.281 | 0.391 |
| G8-shuffled | 0.972 | 4.22 | 1.45 | 48 | 0.109 | 0.975 | 0.281 | 0.391 |

## Recuperação por cobertura de entidades (auditoria)

_(nenhuma condição rodou com `retrieval.face_coverage`)_

## Estatísticas do grafo

| condition | V | E | F | C | genus | max face | bigons | leaf faces | collapses | consolid. | incongr. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| G4-fatgraph-sigma | 2982 | 3711 | 213 | 101 | 359 | 719 | 44 | 69 | 0 | 0 | 11 |
| G8-shuffled | 2982 | 3711 | 213 | 101 | 359 | 719 | 44 | 69 | 0 | 0 | 11 |
| G9-genus | 2982 | 3711 | 511 | 101 | 210 | 141 | 122 | 69 | 0 | 0 | 11 |

## Custo em tokens de LLM

| condition | calls | cached | tokens ingest | tokens QA | total | wall |
|---|---|---|---|---|---|---|
| G4-fatgraph-sigma | 4338 | 4242 | 780,864 | 2,792,966 | 3,573,830 | 280s |
| G8-shuffled | 4338 | 2364 | 780,864 | 2,778,151 | 3,559,015 | 5562s |
| G9-genus | 4338 | 2448 | 780,864 | 2,773,811 | 3,554,675 | 5216s |
