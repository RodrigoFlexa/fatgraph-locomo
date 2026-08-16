# Arquitetura

Como as peças se encaixam, e por que estão onde estão.

```
                       .env  ─────────┐  (segredos + escolha de modelo)
                                      ▼
 configs/base.yaml ──▶ configs/conditions/X.yaml ──▶ Config ──▶ --set
                                                       │
                                                       ▼
                                                    Runner
                        ┌──────────────────────────────┼──────────────────────────┐
                        ▼                              ▼                          ▼
                    Ingestor                     FaceRetriever                 Baseline
        (memory/: extração, entidades,   (retrieval/: índice, walk_face,   (baselines/: B1,B2,B3)
         políticas de σ, curadoria)        montagem do contexto)
                        │                              │                          │
                        ▼                              ▼                          │
                    FatGraph ──── serialize ──▶ artifacts/graphs/                 │
                    (core/: α, σ, φ,                   │                          │
                     faces, Euler)                     ▼                          ▼
                                                    Answerer ◀────────────── llm/ (Azure | fake)
                                                       │
                                                       ▼
                                        evaluation/ (scorer oficial + report)
                                                       │
                                                       ▼
                                  results/<condição>/{metrics.json, predictions.jsonl}
                                                       │
                                         ┌─────────────┴─────────────┐
                                         ▼                           ▼
                                    fgl report                notebooks/nbutils.py
```

`fgl report` e os notebooks chamam a **mesma** função de carga
(`fgl.evaluation.report.load_results`) e as mesmas funções de tabela. Um número
no terminal e o mesmo número num gráfico não podem divergir.

---

## Camadas

### `fgl.core` — a matemática

Não conhece LLM, LoCoMo nem configuração. Só half-edges, α, σ, φ, faces, Euler e
as operações topológicas (`collapse_bigon`, `whitehead_flip`). Testável e testado
isoladamente. Se você quiser usar fatgraphs para outra coisa, é este o módulo que
se leva.

Invariantes checados por `check_invariants()`:

* α é involução sem ponto fixo, e as duas metades concordam em `edge_id`;
* σ não tem duplicatas, e o índice de posição está fresco;
* atributos de aresta estão sincronizados entre as duas metades;
* `V − E + F = 2C − 2g` com gênero inteiro e não-negativo por componente.

### `fgl.memory` — conversa → grafo

`FactExtractor` (com cache independente de condição), `EntityResolver` (cascata
exata → embedding → LLM), `SigmaTime`/`SigmaAgent`, `Curator` (colapso de bígonos
e consolidação de faces). Escreve toda decisão em JSONL.

### `fgl.retrieval` — grafo → contexto

`Embedder` trocável, `VectorIndex` (numpy exato, FAISS opcional), `FaceRetriever`
(âncoras + `walk_face` com orçamento compartilhado) e `Answerer`.

### `fgl.llm` — uma única interface

`LLMClient.complete(prompt, ...)`. `AzureLLM` faz backoff exponencial com jitter e
respeita `Retry-After`. `FakeLLM` é determinístico e offline. Toda chamada passa
por um cache em disco com chave
`sha256(provider|deployment|temperature|max_tokens|seed|json_mode|system|prompt)`,
o que torna qualquer reexecução praticamente gratuita.

### `fgl.evaluation` — o scorer oficial, sem reimplementação

`scorer.py` traz as funções do `task_eval/evaluation.py` upstream verbatim, mais
as duas correções que elas exigem (C7 e C13.5). Há um teste que compara nossa
implementação com a original em 9000 pares — 0 divergências.

---

## Decisões estruturais

**Por que `src/` layout.** Impede importar acidentalmente o pacote a partir do
diretório de trabalho: o que os testes importam é o que foi instalado. Erros de
empacotamento aparecem imediatamente, não no dia da publicação.

**Por que `Paths` em vez de caminhos relativos.** Todo caminho é resolvido contra
a raiz do projeto (o diretório com `pyproject.toml`). CLI, testes e notebooks
concordam sobre onde as coisas estão, independentemente do diretório atual.
`FGL_PROJECT_ROOT` sobrescreve quando necessário.

**Por que `--set` vem por último.** Um override explícito nunca pode ser
descartado em silêncio — nem por `--dry-run`. Esse bug existiu na primeira versão
da CLI e há um teste (`test_set_wins_over_dry_run`) que o impede de voltar.

**Por que segredos não passam por YAML.** Configs são versionadas e vão para o
manifesto de resultados. O que entra ali é `<set:9f3c1a2b>`, não a chave.

**Por que o cache de fatos não conhece a condição.** É a única forma de garantir
que B3 e G1 consomem exatamente os mesmos fatos, que é o que faz a ablação medir
topologia e não variação de extração.

**Por que `faces()` precisa de dois índices.** `_edge_index` e `_sigma_pos` são as
únicas concessões à eficiência num código que prioriza clareza — sem eles
`faces()` seria quadrático, e a ingestão de 10 conversas ficaria inviável. Ambos
são reconstruídos a cada mutação e validados por `check_invariants()`. Ver
`DECISIONS.md` D20 e D21.

---

## Estendendo

**Outro backend de LLM:** implemente `LLMClient._call` e registre em
`fgl.llm.client.build_llm`. Nada mais muda.

**Outro embedder:** implemente `Embedder.encode` e registre em `build_embedder`.

**Outra política de σ:** herde `SigmaPolicy`, implemente `position()` e registre em
`build_sigma_policy`. É o ponto de extensão mais interessante — ver
`COERENCIA.md` C9 para por que uma política melhor que `sigma-time` importa.

**Outro dataset:** produza `Conversation`/`Session`/`Turn`/`Question` de
`fgl.data`. O resto do pipeline não sabe que o LoCoMo existe.

**Nova condição:** um YAML em `configs/conditions/` com `extends: base.yaml`.
`fgl config list` e `fgl run-all` a encontram sozinhos.
