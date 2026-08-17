# Decisões de implementação

Onde a especificação foi omissa, escolhi a opção mais simples e a registrei
aqui. Onde ela estava incorreta, a correção está em `COERENCIA.md` e é
referenciada abaixo pelo código `Cnn`.

---

## D1 — Gênero por componente conexa

`euler()` devolve `EulerStats(V, E, F, C, genus, components)` com `genus` sendo
a soma dos gêneros por componente. `as_tuple()` devolve `(V, E, F, g)`, a
assinatura que a spec pede. Ver **C1**, **C2**.

## D2 — Atributos de aresta guardados nas duas metades

A spec põe `state` e `level` em `HalfEdge`, mas ambos são propriedades da
**memória** (a aresta), não de uma metade. Guardar em um só lado exigiria saber
qual é o "canônico"; guardar nos dois pode dessincronizar.

Escolha: guardar nos dois **e** proibir escrita direta. `EDGE_LEVEL_ATTRS`
lista os atributos de aresta; `set_edge_attr()` é o único caminho de escrita e
escreve nas duas metades; `check_invariants()` falha se alguma divergir.
Campos acrescentados: `edge_id`, `shadowed`, `children`, `provenance`, `meta` —
os três primeiros são exigidos pela §3.6 da própria spec, que os menciona sem
declará-los na dataclass.

## D3 — Semântica de `pos1`/`pos2` em `add_edge`

`pos` é o índice de `list.insert`: a meia-aresta nova fica **naquele** índice.
`None` acrescenta ao fim. Para laços (`v1 == v2`), `pos2` é interpretado na
lista **já atualizada** por `h1` — é o que torna possível montar o toro padrão
(`add_edge(v, v, ..., pos1=1, pos2=3)`), usado no teste de gênero 1.

## D4 — Texto e embedding das duas metades

A spec quer "texto do fato, visto da perspectiva deste vértice". Gerar dois
textos por LLM dobraria o custo de extração sem ganho mensurável no F1.

Escolha: as duas metades recebem o mesmo `fact_text` (autocontido por
construção — o prompt de extração exige resolver pronomes e nomear ambas as
entidades) e **compartilham o vetor**. O gancho existe: se `fact` expuser
`text_from_v1`/`text_from_v2`, `add_edge` os usa. Uma fase futura pode
preenchê-los sem tocar no core.

## D5 — Âncora por aresta, não por meia-aresta

As duas metades de uma memória têm o mesmo texto e o mesmo vetor: as top-m
âncoras seriam m/2 memórias duplicadas. A recuperação deduplica por `edge_id`
antes de cortar em `top_m_anchors`. A meia-aresta escolhida define o **ponto de
partida** e portanto o sentido da travessia — informação que sobrevive.

## D6 — `sigma-time` com timestamps empatados

Todos os fatos de uma sessão herdam o timestamp da sessão, então empates são a
regra, não a exceção. Convenção: em empate vence a **posição mais à direita**
(`>=` na varredura), de modo que fatos da mesma sessão entram em ordem de
extração. Comportamento determinístico e documentado.

## D7 — Incongruência julgada entre arestas irmãs

Ver **C12**. Comparação limitada às 3 arestas irmãs mais recentes sobre o mesmo
par de vértices. Nunca apaga: marca `state="incongruente"` e grava
`meta.conflicts_with`.

## D8 — "Vértices extremos" de uma face

Ver **C11**. `v_start` = vértice da meia-aresta canônica inicial; `v_end` =
primeiro vértice distinto a partir de `len//2`. A aresta de consolidação entra
adjacente às meias-arestas da própria face nos dois lados, para funcionar como
corda (divide a face) em vez de ponte (funde faces e sobe o gênero).

## D9 — Resumo de trilha para o `sigma-agent` sem LLM

A spec pede "para cada meia-aresta incidente, a face que passa por ela resumida
em 1 linha". Resumir por LLM cada trilha de cada meia-aresta candidata daria
`grau × fatos` chamadas por inserção — inviável.

Escolha: o resumo é a concatenação dos textos da face truncada em
`ingest.sigma_agent_trail_chars` (160 por padrão). Custo do `sigma-agent`: **1
chamada por (fato, vértice)**, ou seja 2 por fato. A janela mostrada ao agente é
limitada a `sigma_agent_max_trails` (8) meias-arestas mais recentes, e o índice
devolvido é remapeado para a lista cíclica completa — sem isso, um hub de grau
60 geraria prompts de dezenas de milhares de tokens.

Quando o vértice tem grau ≤ 1 não existe escolha (uma única ordem cíclica) e
**nenhuma chamada é feita**. Teste:
`test_sigma_agent_is_a_no_op_when_there_is_no_choice`.

## D10 — Laços rejeitados na ingestão

Um fato cujas duas entidades resolvem para o mesmo vértice ("Caroline gosta de
Caroline") é degenerado. `ingest.allow_self_loops: false` os descarta e conta em
`n_skipped_self_loops`. O **core** suporta laços (D3) — a rejeição é política de
ingestão, não limitação estrutural.

## D11 — Índice vetorial em numpy por padrão

Ver **C14**. Busca exata por produto interno sobre vetores normalizados. FAISS
fica atrás de `index.backend: faiss` e degrada para numpy sozinho se o import
falhar (registrado no manifest do `metrics.json`).

## D12 — Backends offline determinísticos

`FakeLLM` + `HashingEmbedder` (`configs/conditions/test_offline.yaml`, ou `--dry-run` em
qualquer comando) executam **todo** o pipeline sem rede, sem download de modelo e
sem gasto. É o que a suíte de testes usa, e o que permitiu validar as 6
condições de ponta a ponta sobre os dados reais do LoCoMo antes de qualquer
chamada paga. Os números produzidos assim não têm valor científico — só servem
para exercitar o código.

## D13 — Cache de extração independente da condição

O caminho do cache é
`artifacts/facts/<deployment>-<hash do prompt>/<sample_id>/session_NNN.json`.
Não contém o nome da condição. É o que garante o requisito da §6 de que B3 e G1
consumam exatamente os mesmos fatos. Teste dedicado:
`test_facts_cache_is_condition_independent`.

## D14 — Recall@k

A spec pede recall@k com k ∈ {5,10} sobre os `turn_ids` das arestas
recuperadas. Para as condições de fatgraph, "k" é ambíguo (k âncoras? k fatos?).

Escolha: `recall@k` = recall das evidências anotadas sobre os `turn_ids` das
**top-k arestas por score de âncora** — comparável termo a termo com B2 (top-k
turnos) e B3 (top-k fatos). Reportamos ainda `recall_context`, o recall sobre o
contexto efetivamente enviado ao modelo (todas as faces percorridas), que é a
métrica que de fato explica o F1. Ambos usam a definição do `eval_question_answering`
oficial.

## D15 — Abstenção

String exata: `Not mentioned in the conversation` (constante
`locomo.ABSTAIN_ANSWER`). Verificado contra a regra oficial da categoria 5 — ver
**C7**. Abstém-se quando (a) nenhuma face foi recuperada, ou (b)
`retrieval.incongruent_abstain` está ligado e **todas** as arestas do âncora de
rank 0 estão `incongruente`. A condição (b) é deliberadamente conservadora: uma
única memória incongruente no contexto não deve suprimir uma resposta que as
outras sustentam.

## D16 — Truncamento do B1

Ver **C10**. Guard em 110k tokens, descarte da sessão **mais antiga** primeiro
(comportamento upstream), contador `truncated_conversations` exposto. Com
`gpt-4o-mini` o contador fica em zero: a maior conversa tem ~24k tokens.

## D17 — Stemmer opcional

Ver **C13.5**. Sem NLTK o F1 usa stemmer identidade e o `metrics.json` grava
`"stemmer": "identity"`. Nunca falha silenciosamente.

## D18 — Retomada e idempotência

Grafos são persistidos em `artifacts/graphs/<condição>/<sample_id>.json|.npz` e
reusados por `fgl qa`. Chamadas de LLM são cacheadas por
`sha256(provider|deployment|temperature|max_tokens|seed|json_mode|system|prompt)`.
Reexecutar uma condição interrompida custa aproximadamente zero. `fgl ingest --force` reconstrói.

## D19 — Whitehead flip

Implementado e testado (grafo teta, dois vértices de grau 3), **desligado** por
`curation.whitehead_flip: false`. Recusa laços (contração indefinida) e
endpoints de grau < 3 (não há diagonal alternativa). Verifica `C`, `genus` e
`F` depois do movimento e levanta `TopologyViolation` se algum mudar.

## D20 — Clareza sobre eficiência

Regra da spec seguida. As concessões foram três, todas exigidas pelo requisito
de `faces()` ser O(|H|), e todas validadas por `check_invariants()`:

1. `_sigma_pos` (`half_edge -> posição`), sem o qual `sigma_next` seria O(grau);
2. `_edge_index` (`edge_id -> (h1, h2)`), sem o qual `edge_half_edges` varreria
   todo o conjunto de meias-arestas e `_components()` — chamado por `faces()` —
   seria O(|E|·|H|), isto é, quadrático;
3. `_components_cache`, invalidado por qualquer mudança estrutural, porque
   `face_of` é chamado uma vez por posição candidata pelo `sigma-agent`.

Os itens 2 e 3 foram encontrados **na verificação final**, medindo: antes da
correção, `faces()` escalava 14× para 4× de arestas.

## D21 — Rotação mínima por algoritmo de Booth

`face_id` canoniza a sequência cíclica de arestas pela menor rotação
lexicográfica. A implementação óbvia
(`min(range(n), key=lambda i: doubled[i:i+n])`) é O(n²) e passou a dominar
`faces()` assim que as faces ficaram longas — e por **C9** elas ficam:
faces de comprimento 200+ são a norma, não a exceção.

Trocado pelo algoritmo de Booth, O(n). Verificado contra a implementação ingênua
em 4000 sequências com repetições e prefixos ambíguos: 0 divergências. Medição
depois da correção (grafos aleatórios):

| \|H\| | faces() |
|---|---|
| 3 000 | 3,8 ms |
| 12 000 | 22,2 ms |
| 48 000 | 140,3 ms |

Há um teste de regressão (`test_faces_is_linear_in_the_number_of_half_edges`)
que falha se o comportamento voltar a ser quadrático.

## D22 — Expansão por sigma na recuperação (condição G4)

**Problema.** A recuperação era inteiramente *anchor-centric*: âncoras por
cosseno e, como única expansão, `walk_face`. Isso basta para single-hop — a
resposta está no fato mais parecido com a pergunta — mas o segundo salto de uma
pergunta multi-hop é, por construção, *não* parecido com a pergunta: ele só se
torna relevante depois que o primeiro fato revela a entidade-ponte. Nenhuma
etapa do pipeline reconsultava o grafo condicionada ao primeiro salto.

**Observação.** No formalismo, "duas memórias que compartilham uma entidade" é
exatamente "duas meias-arestas na mesma órbita de `sigma`". O salto já estava no
grafo, em `sigma_next`, e não era consultado. `phi = sigma∘alpha` contém esses
vizinhos, mas sai do vértice a cada passo: só volta à entidade depois de uma
volta na superfície, em geral além de `budget_tokens`.

**Implementação.** `FaceRetriever.sigma_neighborhood` percorre a órbita nos
**dois** vértices da aresta-âncora (a ponte pode estar em qualquer um dos dois),
opcionalmente reordenada por similaridade com a pergunta — reordenar importa
porque sob `sigma-time` o sucessor cíclico é apenas o fato cronologicamente
adjacente. Note que isso **não** é o k-NN global: os candidatos são restritos à
órbita, isto é, pergunta-se "dentre os fatos *sobre esta entidade*, qual
responde?", que é o ponto todo.

O orçamento de sigma é separado *antes* do laço de faces (`sigma_budget_frac`),
senão a face do âncora 0 consome tudo e o salto nunca roda; e o truncamento por
`max_facts_in_prompt` protege os fatos de sigma, que entram por último e seriam
os primeiros a cair.

**Degenerescência encontrada ao testar.** Numa **estrela**, a expansão é inútil:
vértices de grau 1 devolvem `sigma` a si mesmos, então `phi` degenera em marchar
pela órbita do próprio hub e a face já entrega os vizinhos em ordem. O ganho
existe quando os vizinhos têm grau > 1 — que é o caso quando as entidades de
fato se repetem entre memórias. Por isso `sigma_dup` e `sigma_scanned` são
contados e reportados: `dup/scanned` alto significa "a face já cobria a órbita"
(problema de topologia), `scanned ≈ 0` significa "as órbitas estão vazias"
(problema de ingest). Pedem correções opostas, e sem os dois contadores um
resultado nulo seria indistinguível de um bug. Fixado em
`test_star_graph_gains_nothing_from_sigma`.

**Isolamento experimental.** G4 difere da G1 em três chaves e nada mais
(`condition`, `retrieval.sigma_expand`, `paths.graphs_condition`), verificado em
`test_g4_differs_from_g1_only_in_retrieval_and_graph_reuse`. `graphs_condition`
faz a G4 ler os grafos **da G1**, byte a byte, de modo que o delta não pode ser
atribuído a variação de extração ou de resolução de entidades. Com o flag
desligado, o caminho de código é o antigo e G1–G3 continuam reproduzindo os
números guardados.

## D23 — Recuperação por cobertura de entidades (G5) e a combinação (G6)

**Problema.** Mesmo com D22, o `argmax` continuava sendo sobre meia-aresta: a
unidade de decisão era o fato solto e a face vinha de brinde como expansão. A
pergunta que o sistema fazia era "que fato PARECE a pergunta?" — a pergunta de
qualquer RAG, e a razão de o single-hop ser fácil.

**Inversão.** A face passa a ser a unidade recuperada e as meias-arestas apenas
a pontuam:

    score(f) = agg_sim(f, q) + w · cobertura(f, Q)

onde `Q` são os vértices que a pergunta nomeia. A parcela de cobertura é
**estrutural**: uma face que passa por `Melanie` e por `Bangkok` é candidata a
conter a ponte mesmo que nenhum fato dela se pareça com a pergunta — que é
exatamente o que o cosseno não pode expressar, e exatamente o motivo de o
multi-hop falhar. `faces_through_vertex` torna o conjunto candidato barato:
limitado pelo grau dos vértices, não pelo tamanho do grafo.

`agg_sim` é `max`/`top2`, nunca `mean`: a média penaliza faces longas, que são
justamente as que atravessam sessões.

**Linker próprio, não o `EntityResolver`.** O resolver **cria** um vértice
quando nada casa, e durante o QA só se pode LER a memória (spec seção 5). O
`QuestionLinker` é read-only e sem chamada de LLM: n-gramas normalizados contra
nomes e aliases, e só então vizinhos por embedding acima de um limiar. Um miss
é um miss. Fixado em `test_linker_never_creates_a_vertex`.

**Geodésica.** Quando nenhuma face cobre 2+ entidades, o que encadeia as duas é,
por definição, um caminho entre seus vértices — comprimento 2 sendo o caso
dominante. BFS com profundidade máxima 3.

**Teto de fatos por face, medido e não suposto.** Faces têm 200+ memórias
(C9). Sem teto, uma única trilha coberta enche `max_facts_in_prompt` sozinha e
sufoca tanto o caminho dos âncoras quanto a expansão por sigma — observado numa
corrida offline em que a G6 aparecia com sigma zerado. Daí
`coverage_max_facts_per_face`, e daí ele ser menor na G6 (12) do que na G5 (20):
lá a cobertura divide o prompt com dois outros mecanismos.

**Ortogonalidade (G6).** Os dois mecanismos atacam falhas diferentes: sigma
expande A PARTIR de um âncora certo ao qual falta o segundo salto; a cobertura
escolhe QUAL trilha recuperar quando o âncora é irrelevante. Ordem dentro do
orçamento: cobertura, âncoras, sigma — cada um com fatia reservada ANTES do
laço, senão a face do âncora 0 consome tudo. `validate()` recusa fatias que
somem ≥ 1, o que deixaria o caminho clássico sem contexto.

**Contrafactuais.** Cada mecanismo reporta o recall que o contexto teria SEM
ele (`recall_context_no_sigma`, `recall_context_no_coverage`,
`recall_context_anchors_only`). A diferença contra `recall_context` é o efeito,
por categoria, sem precisar de uma corrida extra.
