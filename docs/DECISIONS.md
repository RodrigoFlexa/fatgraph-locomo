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

## D24 — Grafo bipartido turno×entidade sem LLM na ingestão (condição L1)

**Problema.** O diagnóstico da estrela (`docs/COERENCIA.md`, `T1_topical.yaml`)
é medido, não hipotético: 81% dos vértices com grau 1, os dois falantes como
os dois vértices de maior grau em toda conversa. Causa: o extrator de triplas é
generativo — inventa frase nova para a "outra" entidade quase toda vez, e sem
frase recorrente a resolução de entidade não tem o que fundir. T1 tentou
corrigir isso de DENTRO do paradigma de triplas, reprompting o mesmo extrator
para escolher entidades melhores. É uma correção probabilística: o LLM segue a
instrução ou não, fato a fato, e mesmo seguindo ainda precisa inventar frase
consistente para a entidade recorrer. G4–G10 foram todos testados sobre esse
substrato e não tinham topologia real para caminhar.

**Desenho.** L1 não extrai triplas. Constrói o grafo do jeito que um índice
construiria: um vértice por TURNO, um vértice por ENTIDADE canônica, uma aresta
por "entidade E foi mencionada no turno T" — achado deterministicamente por
`fgl.memory.ner.NonGenerativeExtractor` (spaCy `en_core_web_sm`, NER +
noun-chunks), zero chamadas de LLM na ingestão. Mesma jogada que o Zero-Mem
(arXiv:2607.29377) faz para seu grafo entidade-contexto, adaptada ao
formalismo de ribbon graph. Consequência direta: grau de entidade volta a ser
sinal real — grau 1 significa "mencionada uma vez", não "o extrator nunca
reusou uma frase para isto".

`sigma` não precisa de `SigmaPolicy`: no vértice-turno, ordem de leitura cai de
graça da ordem em que `NonGenerativeExtractor` devolve os candidatos (ordem do
documento); no vértice-entidade, ordem cronológica cai de graça da ordem em que
sessões/turnos são visitados (já ordenados). Ao contrário de `SigmaTime`, que
busca a posição de inserção entre timestamps fora de ordem, aqui basta
`add_edge(pos1=None, pos2=None)` (append) nos dois lados — ver
`fgl.memory.ingest_bipartite`.

**Falante nunca é vértice, decidido por dado e não por intuição.** Este ponto
mudou de posição ao longo da conversa que levou a este design (registrado
porque a mudança não foi assinalada na hora, e deveria ter sido). A decisão
final veio de contar o LoCoMo real: dos 282 exemplos de categoria 1
(multi-hop), 222 (79%) são fatos de UMA pessoa espalhados entre sessões —
respondidos pela órbita de sigma de uma única entidade, sem ponte nenhuma. Só
50 (18%) são o padrão genuíno de ponte entre dois falantes ("o que X e Y
fizeram"). Excluir o falante do grafo inteiramente (nunca um vértice, apenas
metadado no vértice-turno) evita recriar o hub por outro caminho, e o caso
minoritário é coberto à parte por `_boost_speaker_coverage`
(`fgl.retrieval.bipartite`): quando a pergunta nomeia os dois falantes, garante
que ao menos um turno de cada sobreviva ao corte por orçamento — sem precisar
que o falante seja um vértice para isso.

**Recuperação sensível a grau.** `BipartiteRetriever` classifica cada entidade
ligada pela sua degree antes de decidir o que fazer com ela: grau 1 é acerto
direto; `2 ≤ grau < bridge_max_degree` é entidade real, órbita inteira
enumerada; `grau ≥ bridge_max_degree` é hub, nunca enumerado, só usado como
filtro/bônus — a mesma distinção que `retrieval.sigma_skip_hub_degree` já fazia
para expansão de sigma no grafo de triplas, mas medida fresca na distribuição
DESTE grafo (`bridge_max_degree=18`, logo após p99 medido em `fgl ingest L1 -n
10`: p50=1, p75=2, p90=4, p95=7, p99=17, max=128) — o limiar do grafo de
triplas não transfere, os dois grafos não têm a mesma forma (aqui é ~1 aresta
por menção, lá é ~1 por fato extraído). A ponte entre duas entidades ("o que
Caroline pintou na praia e o que Melanie pintou com aquarela") é achada por
interseção de vizinhança — os turnos candidatos de cada entidade ligada, olhando
as OUTRAS entidades de cada um; a que aparece nos dois lados é a ponte. Nenhum
cosseno necessário para achá-la: a própria incidência é o sinal. A busca densa
(embeddings de turno) roda em pé de igualdade desde o início, não como
fallback: NER não pega adjetivo, sentimento, nem o que a categoria 3
(open-domain) pergunta.

**Resolução temporal determinística.** `fgl.memory.temporal` resolve datas
relativas ("last Saturday") contra o timestamp da sessão, no momento da
ingestão. Medido: 69,5% dos turnos-evidência da categoria 2 (temporal) contêm
uma frase desse tipo, e nada no pipeline existente resolvia isso antes.
`dateparser` sozinho não é confiável apontado direto para o texto do turno —
verificado empiricamente: `dateparser.search.search_dates` numa frase inteira
produz falso positivo em palavras curtas ("an", "a"); `dateparser.parse` numa
frase JÁ isolada falha para "last/next + dia da semana" mesmo funcionando para
o nome do dia isolado. Por isso o módulo só resolve span que o spaCy já isolou
como DATE/TIME, tenta o caminho rápido primeiro, cai para `search_dates` só no
caso qualificado por dia da semana, e nunca reporta uma resolução que não
moveu a data (`dt.date() == base.date()` é rejeitado, exceto "today"/"this
day") — sem esse último filtro, "this week"/"this month" voltam exatamente na
data-base e seriam reportados como se tivessem resolvido algo.

**Degenerescências encontradas ao medir, não supostas.** `fgl ingest L1 -n 10`
(10 conversas reais, custo zero) expôs duas antes de qualquer QA rodar:

1. Em toda conversa, o vértice de maior grau era "thank" (74–115) seguido de
   "photo" (70–112) — LoCoMo tem agradecimento e compartilhamento de imagem
   constantes, um ato comunicativo, não um tópico, mesma categoria de "thing"/
   "lot". Corrigido acrescentando esses termos a `_GENERIC_NOUNS`
   (`fgl.memory.ner`) — medido, não suposto de antemão.
2. "Mel" (apelido de Melanie, usado como vocativo o tempo todo) virava vértice
   próprio de grau 58 em conv-26 — réplica mais branda do problema exato que
   este design existe para evitar, com outra grafia. `_is_speaker_mention`
   (`fgl.memory.ingest_bipartite`) passou a excluir também prefixo/sufixo de
   comprimento ≥ 3 do nome de qualquer falante, não só igualdade exata.
3. `doc.noun_chunks` do spaCy descarta silenciosamente frases cujo parser de
   dependência degenera num turno sem sujeito — caso real, turno D2:8 da
   conv-1: "Researching adoption agencies — it's been a dream...". "agencies"
   recebe `dep_="dep"` (fora do conjunto fixo de padrões que `noun_chunks`
   reconhece), e a frase inteira ("adoption agencies", a resposta-ouro da
   pergunta associada) nunca vira candidato. Como as arestas de dependência
   `compound` continuam corretas mesmo quando o parse acima do substantivo é
   ruim, `_compound_fallback` (`fgl.memory.ner`) percorre essas arestas
   diretamente para recuperar a frase — restrito a resultado com 2+ tokens
   (medido: sem essa restrição, 251/1451 turnos = 17,3% ganhavam candidato
   espúrio, interjeição maiúscula mal-rotulada NOUN pelo spaCy; com a
   restrição, 13/2760 = 0,5%, e o que sobra é majoritariamente real —
   "adoption agencies", "ice cream", "coconut milk"). O resíduo era
   vocativo — "Thanks Nate" como composto de dois tokens — capturado à parte
   por `_mentions_speaker`, que agora testa cada palavra do candidato contra
   o nome dos falantes, não só a string inteira.

**Medido após as três correções**, `fgl ingest L1 -n 10`: `degree_1_frac` médio
51,3% (era 81% no grafo de triplas), `hub_share` médio 2,09% (era ~45%) — abaixo
dos limiares de alerta do próprio projeto (`STAR_DEGREE1_FRAC=0.6`,
`STAR_HUB_SHARE=0.35`); zero chamadas de LLM confirmado pela própria saída do
CLI ("LLM: 0 calls"); os vértices de maior grau em cada conversa passaram a ser
tópicos reais ("dog", "painting", "yoga", "photography"), não ruído.

**Verificado, não descartado como bug**: no smoke test offline
(`llm.provider=fake`, `embeddings.provider=hashing`), duas perguntas sem
sobreposição lexical com nenhuma entidade do grafo ("What did Caroline
research?", categoria 1; uma pergunta open-domain de categoria 3) vieram com
`recall_context=0.0`. Rodar a mesma pergunta sob G1 com o mesmo harness offline
reproduz exatamente o mesmo `recall_context=0.0` — confirmando que a causa é o
`HashingEmbedder` não ter sinal semântico real (nem para o fallback do linker
de entidade, nem para o backstop denso), não um defeito específico da L1. G1
inclusive falha numa pergunta temporal onde a L1 acerta
(`recall_context=1.0`) no mesmo smoke test. Validação de qualidade real desta
classe de pergunta depende de um embedder semântico de verdade, a mesma
limitação que toda outra condição já tem sob este harness.

**Limitação declarada, não escondida.** Sem detecção de incongruência: fazer
igual ao caminho de triplas reintroduziria exatamente a dependência de LLM que
esta condição existe para remover; heurística determinística de negação é o
próximo passo natural, não tentado aqui. Categoria 5 (adversarial, 22,5% do
benchmark) é pouco tocada: só 4/446 perguntas adversariais não têm sobreposição
de vocabulário nenhuma com a conversa, então um sinal "nada ligou → abster" 
dispara em bem menos de 1% delas. `bridge_max_degree` não é um número fixo do
código — é recomendado medir de novo (`fgl diagnose --bipartite` ou `fgl ingest
L1 -n 10` e inspecionar `graph_stats`) antes de confiar nele numa base de dados
diferente, porque a forma do grafo bipartido depende de quanto texto informal
tem cada conversa.

## D25 — Correções na L1 medidas na própria L1 (formato de data, teto de fatos, partição por falante)

Três defeitos encontrados relendo `results/L1-bipartite/predictions.jsonl` e os
grafos da própria condição. Nenhum é hipótese: cada um vem com o número que o
delatou.

1. **Formato de data.** 47,7% das respostas temporais da L1 saíram em ISO
   (`2023-05-07`) contra gold `7 May 2023` — F1 = 0,0 mesmo com
   `recall_context = 1.0`, porque o scorer oficial tokeniza e as duas strings
   não têm um token em comum. Na G4, com o mesmo prompt e o mesmo modelo, isso
   acontece em 0,6% das respostas: o que muda é que a L1 injeta datas
   resolvidas no contexto (`fgl.memory.temporal`), e o modelo copia o formato
   que vê. Corrigido nos dois lugares — `ResolvedDate.render()` passou a emitir
   formato natural, e a regra 3 de `prompts/answer.txt` agora nomeia o formato
   em vez de pedir "uma data". Recomputando o F1 oficial sobre as predições
   existentes só com a data normalizada: temporal 0,178 → ~0,466, micro 0,445 →
   ~0,491.

2. **Teto de fatos, não de tokens.** 27,5% das perguntas eram truncadas pelo
   limite de 40 *fatos*, e só 3,0% chegavam aos 2000 tokens de orçamento —
   contexto médio 1256 tokens, 37% do orçamento declarado nunca gasto. E as
   perguntas truncadas tinham `recall_context` e F1 *maiores* (0,750/0,440
   contra 0,679/0,356): o teto cortava a metade boa. `max_facts_in_prompt`
   40 → 80, `budget_tokens` inalterado, então a comparação com as outras
   condições continua a custo idêntico. Junto veio um defeito irmão: os dois
   recuperadores pediam `top_m_anchors * 4` candidatos ao canal denso, um
   número anterior ao teto de fatos, então mesmo com o teto em 80 a L1 gerava
   ~32 candidatos e continuava gastando metade do orçamento. A largura passou a
   acompanhar `max_facts_in_prompt`.

3. **Partição por falante.** 98,5%–99,7% das perguntas do LoCoMo nomeiam
   exatamente um dos dois falantes, e quando nomeiam, o turno de evidência é
   desse falante em 96% (single-hop), 100% (multi-hop, 244/244), 98%
   (temporal) e 100% (open-domain, 72/72) dos casos — enquanto 24% de todo
   contexto recuperado era turno do outro falante. `bipartite.speaker_partition`
   descarta esses candidatos. Isso **não** faz do falante um vértice: lê
   `meta["speaker"]`, que é atributo do turno, então a topologia não muda e o
   hub que a exclusão do falante existe para evitar não volta. Tem piso
   (`speaker_partition_min`) porque nas poucas perguntas em que quem falou não
   é quem foi nomeado, um filtro voraz troca um erro de ranking por um contexto
   vazio — que é estritamente pior, já que força abstenção.

**Medido junto, com `fgl slots-oracle -C L1` (zero chamadas de LLM):**
`recall_context` multi-hop 0,456 → 0,714 e single-hop 0,736 → 0,850 nas
primeiras conversas, com o contexto passando de 1001 para ~1920 tokens dentro do
mesmo orçamento de 2000.

## D26 — Vocabulário de slots tipados sobre episódios (condição L2)

**O que a medida diz.** Reconstruindo os grafos da própria L1 e cruzando com a
evidência anotada: só 0,5% dos turnos de evidência estão ausentes do grafo, mas
dos turnos de evidência que a L1 **não recuperou**, a fração que compartilhava
ao menos uma entidade com a pergunta é 13% (single-hop), 7% (multi-hop), 10%
(temporal), 5% (open-domain). Ou seja: a lacuna de recall não é cobertura nem
ranking — 87%–95% dos erros são turnos para os quais o grafo de incidência
entidade×turno não oferece caminho nenhum.

Inspecionando esses erros, quatro pontes faltam, e cada uma vira um tipo de
vértice em `fgl.memory.slots`:

| ponte | exemplo medido | tipo |
|---|---|---|
| a resposta do diálogo | "What kind of dance piece did Gina's team perform?" → `"We just did a contemporary piece called 'Finding Freedom.'"` — nenhum substantivo em comum | `episode` |
| o predicado | "What did James **adopt** in April 2022?" → "I **adopted** a pup" | `predicate` |
| o tipo | "What **foods** does Audrey like?" → "**Roasted chicken** is one of my favorites" | `type` (hiperônimo WordNet) |
| a pessoa | 98,5%–99,7% das perguntas nomeiam um falante; a L1 apaga o falante do grafo de propósito | `actor` |

**Zero LLM na memória, como na L1**: uma passada de spaCy por turno entrega
noun chunks, lemas de verbo, spans PERSON e spans DATE de uma vez; WordNet e o
resolvedor de datas já existente fazem o resto. O canal de tipos se desliga
sozinho com flag registrada em `graph_stats["wordnet_types"]` se o corpus não
estiver instalado, porque é aditivo.

**O que o ribbon graph faz aqui, e o que não faz.** `sigma` num vértice de slot
é cronológico, então "tudo que esta pessoa/predicado/conceito tocou, em ordem" é
leitura de lista e não ranking. `sigma` num vértice de episódio segue
`SLOT_ORDER`, o que torna os **cantos** do episódio os pares (quem, fez-o-quê),
(fez-o-quê, com-o-quê), (com-o-quê, quando) — e uma pergunta do LoCoMo é
exatamente um canto com um lado em branco. **Faces não são usadas**: nos grafos
medidos da L1 uma única face já contém 2954 meias-arestas, e para "tudo que X
fez" o objeto certo sempre foi a órbita, que é o vértice. Tipar os vértices é o
que daria à face uma chance de significar algo — é o experimento seguinte, não a
tese desta condição.

**Abstenção determinística.** Uma pergunta adversarial nomeia uma combinação que
nunca aconteceu: isso é um canto que não existe. Dois formatos, ambos sem LLM —
`missing_slot` (todo o vocabulário de conteúdo da pergunta está ausente da
memória) e `empty_corner` (o conteúdo existe, mas nunca num episódio que essa
pessoa domina). O teste de posse é de **maioria** (`corner_actor_min`), não
"contribuiu alguma coisa": um episódio é um par adjacente, os dois falantes
aparecem em quase todos por construção, e com limiar zero o canto existiria em
todo lugar. Sai **desligado** (`abstain_on_empty_corner: false`) de propósito —
é o único mecanismo da condição capaz de apagar uma resposta correta, então
liga-se a partir da taxa de falso positivo medida por `fgl slots-oracle`, não
porque o mecanismo é elegante.

**Unidade de índice ≠ unidade de emissão.** O episódio é o que torna a memória
*ligável*; é a unidade errada para *pagar*. Três tentativas medidas, todas ao
mesmo orçamento de 2000 tokens:

* emitir episódios inteiros → 18,9 unidades contra as 58,7 da L1; tudo que
  precisa de largura perdeu;
* emitir turnos mas esvaziando um episódio antes do próximo → 54,5 unidades,
  multi-hop ainda perdendo, porque 55 turnos vindos de ~18 episódios
  consecutivos cobrem 18 regiões da conversa onde os 58 turnos independentes da
  L1 cobrem 58, e evidência multi-hop é espalhada por definição;
* um turno por episódio em rodadas → regiões recuperadas, single-hop
  0,858 → 0,682, porque o turno que *responde* é muito frequentemente o
  vizinho do turno que *casa* — que é a razão de existir do episódio.

A forma final pontua **turnos**: similaridade própria, mais o que o turno herda
do seu episódio (os canais tipados e `sibling_frac` da melhor similaridade dos
irmãos). A regra da resposta vira aritmética em vez de agrupamento, e o par sobe
junto porque as duas metades carregam a mesma herança.

## D27 — `fgl slots-oracle`: comparar modelos de memória sem gastar LLM

Responder custa; recuperar não. E medido na L1, condicionado à evidência estar
no contexto, ela já empata com a baseline de contexto inteiro nas categorias
substantivas (single-hop 0,641 contra 0,653; multi-hop 0,360 contra 0,392;
adversarial 0,639 contra 0,630) usando 5,4% dos tokens — não há folga de
resposta a encontrar, todo ponto restante é recuperação.

Então `fgl slots-oracle` recupera para as 1986 perguntas sob cada condição, não
responde nenhuma, e reporta `recall_context` por categoria mais as duas coisas
sem as quais essa comparação não significa nada (quantas unidades e quantos
tokens cada modelo realmente gastou) e a matriz de confusão da abstenção
determinística. Os limiares em `TARGETS` não são nota de corte para o artigo:
são regra de parada, para abandonar um modelo que não move a recuperação antes
que ele custe alguma coisa.

## D28 — Moldagem de resposta, e o resultado negativo que ela produziu

A métrica é F1 de tokens contra uma referência de 3 palavras, então resposta certa embrulhada em frase é pontuada como parcialmente errada. Medido no run da L2, restrito às perguntas cuja evidência anotada estava no prompt:

```
single-hop  n=759  F1 0.651  melhor janela contígua da própria predição 0.745  irrecuperável 13%
multi-hop   n=109  F1 0.378  melhor janela 0.492                               irrecuperável 29%
```

49% das predições single-hop já contêm uma substring que pontua 1.0 — o modelo sabe a resposta e a acolchoa. Daí `fgl.evaluation.shaping`: regras que só **apagam**, cada uma individualmente ligável e individualmente precificada (`fgl reshape --ablate`), aplicadas offline sobre `predictions.jsonl` e repontuadas com o scorer oficial. Isso é grátis, retroativo e — o ponto — **justo**: aplicado a uma condição é vantagem de prompt, aplicado a todas é correção de métrica. O `predictions.jsonl` original nunca é sobrescrito.

Uma invariante é assertada, não torcida: moldagem **não pode** mover a nota adversarial. A regra da categoria 5 é um teste de substring por "not mentioned"; um aparo que cortasse essa string converteria abstenções corretas em respostas erradas silenciosamente. `rescore_rows` levanta exceção se a média adversarial se mexer.

**Resultado medido, e é negativo.** Sobre B1, B3 e a L1 antiga:

| | single-hop | temporal | micro |
|---|---|---|---|
| B1-full-context | +0.003 | −0.000 | +0.001 |
| B3-rag-facts | −0.000 | +0.003 | +0.000 |
| L1 (run pré-ISO) | +0.002 | **+0.288** | +0.048 |

Ou seja: **a moldagem recupera exatamente uma coisa, a data ISO** — e reproduziu +0.288 em temporal, batendo a estimativa que eu tinha feito à mão, o que valida a ferramenta. Fora isso, +0.003 no single-hop. Num run que já tem o fix do ISO no prompt, o ganho restante é ~zero.

Puxando os maiores gaps para ver o porquê, o acolchoamento **não é** moldura ("eu acho que", "foi em"). É sintagma nominal cheio:

```
gold 'meditation'        pred 'a meditation course at a retreat near a lake'
gold 'hiking'            pred 'hiking with my church friends'
gold 'Woodhaven'         pred 'Woodhaven, a small town in the Midwest'
gold 'Horseback riding'  pred 'used to go horseback riding with my dad'
```

Testei então um aparo **sintático** com spaCy (descartar PP finais, apostos, orações relativas, mantendo a cabeça): single-hop **0.653 → 0.475 (−0.178)**, e pior mesmo aplicado só às predições mais longas que o gold. Descartado. O teto de 0.745 é oráculo — escolhe a melhor janela sabendo o gold — e **não é alcançável por pós-processamento determinístico nenhum**.

Consequência para o planejamento, registrada porque contradiz o que eu havia dito antes: single-hop 0.65 **não** sai só com aparo. A alavanca teve que virar prompt (D29).

## D29 — Precisão em vez de brevidade, e enumeração por órbita

**Precisão, não brevidade** (`prompts/answer.txt` v3). O comprimento médio da predição já é igual ao do gold (4.2 tokens contra 4.2 no single-hop), então instrução global de "seja mais curto" encolheria os 65% que já estão certos. O que separa é outra coisa: 35% das predições são mais longas que o gold e pontuam **0.49**, contra **0.74** das demais. A regra nova nomeia o que sobra — modificador, lugar, companhia, finalidade que a pergunta não pediu — em vez de pedir concisão.

**Enumeração por órbita** (`fgl.retrieval.slots`, passo 2b; `prompts/answer_set.txt`). Categoria 1 é pontuada por `f1_multi`: o gold é lista, a nota é média por item do gold. Um gold de quatro itens limita uma resposta de um item a ~0.25 por construção, por mais correto que esse item esteja. E as predições multi-hop são majoritariamente de um item — só 20% contêm substring valendo 1.0 contra o gold inteiro, contra 49% no single-hop: estão **incompletas**, não erradas.

Uma pergunta de conjunto não é respondida pelo episódio mais parecido; é respondida pela **órbita inteira**. E `σ` num vértice de slot já é essa órbita, em ordem cronológica — sem ranking e sem segunda passada de recuperação. O mecanismo: pegar o slot específico **mais raro** que a pergunta ligou (o que discrimina), intersectar sua órbita com os episódios que o ator nomeado possui, e levantar todos os membros para o prompt. É o único ponto do desenho em que a estrutura ribbon faz algo que um recuperador denso não faz: a resposta é uma lista, e a rotação **é** a lista.

Mais raro e não todos: enumerar a órbita de um slot comum inundaria o orçamento com tudo que a conversa já tocou — o erro de hub que este projeto já cometeu uma vez.

**A detecção lê só o texto da pergunta, nunca a categoria do gold.** Roteamento pela categoria tornaria o mecanismo inutilizável fora deste benchmark e seria ler o gabarito em tempo de inferência. O custo dessa disciplina são algumas perdas em perguntas de categoria 1 fraseadas no singular. Taxa medida (conv-26): multi-hop 47%, single-hop 20%, temporal 0%, open-domain 15%, adversarial 17% — dispara onde as listas moram e não dispara em temporal.

## D30 — Constantes viram estimadores, e a calibração vira um número medido

Três coisas foram feitas de uma vez, e as três respondem à mesma objeção: os números do L2 foram escolhidos **olhando para as respostas anotadas**. A objeção está certa, e o que ela custa não é honra — é portabilidade: nenhum desses números é herdável por um segundo corpus, e nenhum revisor consegue conferi-los.

O critério que separa "método" de "método ajustado a este dataset" é estreito:

> **O parâmetro precisa dos rótulos de ouro para ser fixado?**

Se precisa, é dívida de calibração. Se pode ser estimado do corpus não anotado no momento da construção, é só um algoritmo com um estimador dentro.

### 1. Cada literal virou um estimador (`fgl.memory.calibration`, condição `L2d`)

| era | virou | por quê |
|---|---|---|
| `hub_degree: 60` | quantil 0.99 da distribuição de grau **daquele tipo** | contagem absoluta é bug latente, não falta de elegância: num corpus 10x maior *todo* slot cruza 60 e o mecanismo se desliga sozinho. Por tipo porque as escalas são incomparáveis — um ator incide sobre metade dos episódios, um conceito sobre três |
| `concept_link_threshold: 0.75` | quantil 0.995 da distribuição de cosseno conceito↔conceito observada | um cosseno absoluto é propriedade do *encoder* tanto quanto da tarefa; trocar o modelo de embedding muda o que 0.75 significa |
| `actor_prior_floor: 0.35` / `full: 0.5` | `1/n_falantes` e a mediana da participação do falante dominante | com 8 participantes o piso cai para 0.125 sozinho — o prior fica **mais** forte, que é o correto, porque nomear um entre oito exclui muito mais do que um entre dois |
| `QUESTION_NOUN_STOP` (lista manual) | `df_pergunta(w) / df_memória(w) >= ratio` | palavra de tópico é comum nas perguntas *porque* é comum nas conversas, então a razão fica perto de 1; palavra de template é comum nas perguntas e ausente do que alguém disse. É o análogo do IDF do lado da pergunta, e vale para qualquer conjunto templatizado |
| granularidade de tempo `month` | **parâmetro removido**: indexa ano/mês/dia, a pergunta emite todos os níveis que nomeia | quem escolhe o nível é o **amortecimento por grau que já existia** — vértice de ano incide sobre quase todo o corpus e é apagado por `1/(1+log(grau))`; vértice de dia pontua quase cheio. Nenhum peso novo, nenhuma regra nova |

Esse último ponto é o mais forte do lote e vale registrar como tal: a resolução múltipla de tempo **não precisou de mecanismo novo**. O amortecimento por grau já era um seletor de especificidade, e a granularidade de tempo era um caso particular dele que estava sendo resolvido à mão. Remover o parâmetro melhorou o argumento a favor do resto do desenho.

O que **não** foi calibrado: `dense_weight`, `actor_weight`, `predicate_weight`, `concept_weight`, `type_weight`, `time_weight`, `sibling_frac`, `slot_damping`. São o único grupo que codifica uma afirmação **sobre o modelo** e não sobre o corpus ("um casamento de conceito diz mais que um palpite de hiperônimo" é uma ordenação que este desenho afirma). Fingir derivá-los seria vestir decisão de projeto de medição. O que eles ganham em vez de estimador é uma curva.

`L2_slots.yaml` fixa `calibration: absolute`, `question_stop: literal`, `time_granularities: month` **de propósito**: L2 é a condição de que os números reportados saíram, e uma condição que mudasse de comportamento em silêncio tornaria falsa cada medição acima neste arquivo. `L2d_derived.yaml` é o mesmo modelo com todo estimador ligado. **A diferença entre as duas é a dívida de calibração, medida em vez de discutida** — `fgl slots-oracle -C L2 -C L2d`, orçamento idêntico, zero LLM.

### 2. Sensibilidade em vez do ótimo (`fgl slots-sweep`)

Reportar o valor que venceu uma varredura, e só ele, não diz nada sobre o método depender dele. Duas situações muito diferentes produzem a mesma linha de config: a métrica é **chata** no intervalo e 60 foi pego num platô (então o número não é resultado, e dizer isso é a defesa mais forte disponível); ou a métrica tem um **pico** em 60 (então o número **é** o resultado, foi obtido olhando dados anotados, e é fragilidade a declarar).

O comando varre um knob por vez sobre o oracle sem LLM e reporta, por knob:

- `sensitivity` = `(melhor − pior)/melhor` — quanto o knob consegue mover a métrica;
- `plateau_frac` = fração dos valores dentro de 1% do melhor — largura da região boa;
- **`tuning_gain` = valor entregue − mediana do intervalo** — quanto o valor escolhido bate um valor pego às cegas do mesmo intervalo. **É a dívida de calibração daquele knob, em pontos de recall.**

A soma dos `tuning_gain` é a `estimated_calibration_debt`: a resposta honesta para "quanto do recall reportado veio de ter as anotações?". Ignora interações entre knobs (é varredura *one-at-a-time*), então é ordem de grandeza e está rotulada como tal — mas é uma estimativa da quantidade certa, e é o número que um leitor merece ao lado de um score.

Veredictos: `flat` (o knob não é resultado), `shallow`, `peaked` (é resultado, declare), `cliff` (está no ótimo mas o vizinho despenca — frágil a qualquer deslocamento de corpus).

### 3. Premissas declaradas e verificáveis (`docs/ASSUMPTIONS.md`, `fgl scope-check`)

Um método com condições de escopo **declaradas** é um método; com condições **escondidas** é um método ajustado a um benchmark — e os dois podem ser tecnicamente idênticos. A diferença é só se está escrito e se dá para conferir.

Sete condições (S1–S7), cada uma com enunciado, o que é medido, critério e — a parte que importa — **para o que o desenho degrada quando ela falha**. Premissa sem caminho de degradação declarado é requisito escondido, não premissa.

Duas classes, e a distinção carrega peso. `runtime` é computável do que uma implantação teria (transcrições, no máximo o texto das perguntas) — roda em dados não anotados. `audit` precisa de evidência ou resposta anotada: são exatamente as medições que *produziram* o desenho e exatamente as que um corpus novo não vai conseguir rodar. **S3 ("a evidência é do participante nomeado", 96–100% no LoCoMo) é `audit`** — é a dependência que não pode ser removida por engenharia, só declarada, e por isso não conta na contagem de condições de runtime satisfeitas.

Uma ressalva registrada em vez de enterrada: `question_stop=derived` é ajustado sobre o *texto* das perguntas que serão respondidas. Não usa rótulo nenhum, então não é vazamento para efeito de recall — mas é **transdutivo**, e uma implantação que responde uma pergunta por vez não pode fazê-lo. `literal` e `none` são os fallbacks honestos; o sweep precifica a diferença.

### O que continua faltando, e não está escondido

- **Não há split.** Os knobs do L2 foram varridos nas mesmas 10 conversas de que o número final saiu. Enquanto não houver *leave-one-conversation-out*, a palavra correta continua "varrido contra o oracle", não "tunado no dev, avaliado no test". Nenhum dos três itens acima fecha esse furo.
- **Um único corpus.** Nada aqui prova portabilidade; prova que o método *pode* rodar sem olhar para as respostas. A prova é rodar o config congelado num segundo benchmark (LongMemEval é o alvo óbvio — gerador diferente, sessões com distratores, e casos sem resposta que exercitam o teste de canto) e reportar o número seja qual for.

## D31 — Do salto único à propagação e à conexão (L3, L4)

O run que motivou isto: L1 → L2 levou `recall_context` de 0.614 a 0.770 (+0.156) e comprou **+0.016 de micro F1**. Decompondo com n_adversarial = 446 / 1986:

| | delta |
|---|---|
| categorias substantivas (n=1540) | +0.037 |
| adversarial (n=446) | −0.058 |
| contribuição adversarial ao micro | −0.013 |
| **micro líquido** | **+0.016** |

Duas leituras, e as duas viraram trabalho:

1. **A adversarial comeu 45% do ganho bruto.** Não é ruído, é mecânica: com recall 0.770 o contexto quase sempre contém algo plausível, então o modelo para de se abster. Melhorar a recuperação piorou a abstenção.
2. **A tese "não há folga de geração" está falsificada pelos próprios dados.** Se todo ponto restante fosse recuperação, +0.156 de recall teria comprado muito mais que +0.037 nas substantivas.

(Registrado também: o `recall@10` caindo 0.271 → 0.214 é artefato. `top_edges` expande episódios em incidências e a L2 tem ~23 por episódio, então k=10 nem cobre um episódio. Não perseguir.)

### O movimento de Whitehead: não, e o próprio código já dizia

Está implementado (D19) e desligado, mas a razão para não ajudar é mais forte que "está desligado": `whitehead_flip` levanta `TopologyViolation` se genus, F ou C mudarem — corretamente, porque o movimento é um *spine move* na mesma superfície espessada. O docstring de `transpose_sigma` já registrava isso: *"Whitehead flips do not [change the surface]... the smallest such alteration is a transposition."*

**Whitehead não pode ajudar, por teorema.** O que muda a superfície é a transposição em σ, que `maximize_faces` já hill-climba.

### O argumento em forma de teorema, e por que ele não virou a implementação

Números da L2: V=20978, E=68710, F=108, C=10, genus=23822.

- Comprimento médio de face: **2E/F = 1272 meias-arestas** (a maior, 11166). Uma face desse tamanho é a conversa inteira.
- Bipartido ⇒ toda face tem comprimento ≥ 4 ⇒ **F ≤ E/2 = 34.355**, e daí genus ≥ 6.699.
- Estamos em F=108 de um teto de 34.355 (**0,3%**) e genus 3,6× acima do mínimo.

E num grafo bipartido o mergulho de genus mínimo é **quadrangular**: toda face é um 4-ciclo `e1 — s_a — e2 — s_b — e1`, ou seja **um par de episódios que concorda em duas coisas diferentes**. Isso é literalmente a junção multi-hop, e multi-hop era a pior categoria (0.376).

Duas objeções honestas mataram a rota via rotação:

1. Um mergulho quadrangular seleciona E/2 dos 4-ciclos para ladrilhar a superfície — escolhidos por topologia, não por semântica. Dá uma seleção *canônica*, não uma *boa*. É exatamente por isso que o G8-shuffled deu chato.
2. `maximize_faces` é O(passes × Σ_v deg² × |H|) com |H|=137k e vértices de ator de grau nas centenas. Não roda.

**Conclusão: testar o objeto sem a maquinaria.** Num grafo bipartido episódio↔slot, um passeio de 2 saltos a partir dos slots da pergunta é exatamente "episódios que compartilham um slot com um episódio que a pergunta nomeou" — a junção — e o caso fechado (voltar a um episódio já alcançado por um slot *diferente*) é o 4-ciclo. O passeio é a versão mole e ponderada do objeto que a teoria aponta, sem rotação, sem genus, sem mergulho.

### L3 — o mesmo grafo, lido por propagação

A observação de partida: **o score estrutural da L2 já é uma iteração de random walk with restart.** Os slots da pergunta são o vetor de personalização, a incidência é a transição, e `1/(1+log deg)` é uma normalização por grau feita à mão. Escrito assim, a limitação salta: o passeio para no primeiro passo.

Três coisas fazem ele funcionar em vez de borrar:

**Um hub é filtro, nunca ponte.** O modo de falha de todo passeio em grafo com hub: a massa entra no vértice de ator (incidente a metade dos episódios) e sai espalhada uniformemente. O corte de grau calibrado por tipo (D30) ganha aqui o trabalho para o qual sempre foi melhor: um hub pode *receber* massa no salto 1, onde age como o filtro que a L2 já usa, e nunca pode *retransmitir*. Uma regra, dita uma vez em `SlotRetriever.is_hub`, obedecida pelo scorer, pelo passeio e pela métrica da L4. **É a cola conceitual das três condições.**

**O passeio é não-retornante.** O salto 2 de um passeio comum é dominado por massa que vai `slot → episódio → mesmo slot` e volta: ele re-pontua a semente e chama isso de junção. Rastreando fluxo em **meias-arestas dirigidas** em vez de vértices e subtraindo de cada aresta a própria contribuição de entrada, obtém-se o operador de Hashimoto — e a estrutura de meias-arestas que ele quer o repositório já tem, como `alpha`. Custa um `bincount` a mais por salto.

**A redução é exata.** `hops=1` + `normalization=none` + `dense_seed=0` reproduz a L2 turno por turno e score por score (`tests/test_propagation.py`). Portanto a varredura de `propagation.hops` é uma curva cujo ponto mais à esquerda **é o número publicado da L2**. E a L3 empresta os grafos da L2 byte a byte (`paths.graphs_condition: L2-slots`) e herda todo o resto por subclasse — o delta não pode ser outra coisa.

Também trocado: `normalization: sym` (a normalização espectral `D^-1/2 A D^-1/2`) amortece o **lado do episódio** também, o que `1/(1+log deg)` nunca fez — ele só olhava o lado do slot.

### L4 — a leitura que pergunta como as coisas se conectam

Toda leitura até aqui pergunta a mesma forma de coisa: *quais episódios estão perto do que a pergunta mencionou?* É uma soma, então um episódio ganha casando um slot com força, e nada na aritmética consegue dizer "e também os outros dois".

Uma pergunta multi-hop não pede proximidade, pede: **qual o menor pedaço desta memória que segura todos estes juntos?** Isso é group Steiner tree, a formulação clássica de busca por palavra-chave em grafos (BANKS, BLINKS, DPBF). Duas consequências:

**Um canal com um E lógico.** A relaxação por estrela enraizada — para cada raiz candidata, a soma das distâncias a todos os terminais — é uma conjunção por construção: uma raiz que não alcança um terminal sai da interseção por mais perto que esteja dos outros. É a forma que multi-hop precisa.

**Abstenção com resolução.** O teste de canto é binário e, medido, um mau negócio: 20/446 adversariais a custo de 38/1540 falsos positivos (+0.004 contra −0.010 em micro). **Substituído, não reajustado.** O custo de conexão é contínuo, e o limiar é a cauda superior do custo de tuplas de slots **aleatórias** do mesmo tamanho nesta mesma memória — "estas coisas ficam mais longe uma da outra que 95% das combinações arbitrárias aqui". Sem gabarito, coerente com D30.

A métrica: entrar num slot custa `1 + log(grau)` — rotear por algo que quarenta episódios mencionam é caro — e acima do corte de hub não é atravessável. Mesma regra do passeio. Sem isso, todo par de episódios fica a dois passos pelo vértice de falante e a estrutura colapsa (o fracasso clássico de keyword-search em grafos).

L4 é a única condição da família que não é um isolamento: L1..L3 e L2d mudam uma coisa para que um delta signifique algo; **L4 existe para responder a outra pergunta — as peças compõem?** Cada componente já foi medido sozinho, que é o que faz disto uma síntese e não uma pilha esperançosa. Constrói grafos próprios porque o tempo multirresolução muda o conjunto de vértices.

### A hierarquia é real, não documentada

```
SlotRetriever         L2   quais episódios TOCAM os slots?     um salto
  └─ PropagationRetriever  L3   quais são ALCANÇADOS a partir deles?  passeio
       └─ UnifiedRetriever  L4   quais os SEGURAM JUNTOS?             conexão
```

Subclasses, não irmãos copiados. Um teste verifica que a L3 sobrescreve exatamente `_structural_channels` e mais nada — se pudesse divergir no parser da pergunta, no prior de ator ou na política de emissão, nenhum delta medido seria interpretável.

### `fgl hop-profile` — o portão, rodado antes e não depois

Um passeio mais longo só acha evidência *alcançável naquele número de saltos*. O comando mede onde ela está: salto 1 / 2 / 3 / inalcançável, para toda a evidência e — a que importa — só para a que a condição **errou**. Se as falhas estão no salto 2, a L3 tem alvo e o tamanho daquele balde é o teto dela. Se estão inalcançáveis, nenhum passeio neste grafo as acha e a resposta é outro ingest, não outra leitura. Custo: zero LLM.

Reporta junto os números de quadrangulação acima (F contra o teto E/2, genus contra o piso, pares de episódios compartilhando 2+ slots não-hub), porque são a versão quantitativa do argumento e saem numa passada.

### Custo medido

Smoke test com 192 turnos e 50 perguntas: L2 14.6 ms/pergunta, L3 15.4, L4 16.1 (incluindo a calibração do null). ~10% a mais de tempo de recuperação, desprezível contra a chamada de LLM. A L3 não paga ingest nenhum (empresta os grafos da L2).

## D32 — O que o primeiro oracle da linha L disse (e ele disse "não" para a L3)

Rodado nas 10 conversas, zero LLM. `recall_context` por categoria:

| condição | single | multi | temporal | open | adversarial | tokens | unidades |
|---|---|---|---|---|---|---|---|
| L1-bipartite | 0.824 | 0.639 | 0.855 | 0.548 | 0.204 | 1887 | 59.4 |
| L2-slots | 0.908 | 0.629 | 0.897 | 0.561 | 0.855 | 1993 | 57.8 |
| **L2d-derived** | **0.910** | **0.641** | **0.899** | **0.565** | **0.866** | 1993 | 58.1 |
| L3-propagation | 0.881 | 0.632 | 0.893 | 0.567 | 0.781 | 1984 | 62.4 |
| L4-unified | 0.908 | **0.652** | 0.894 | 0.528 | 0.825 | 1994 | 57.9 |

### 1. A L2d ganha da L2 em **todas as cinco categorias**

E é a condição em que nenhum número foi escolhido olhando para as respostas. Os
limiares derivados não são um empate honroso — são melhores. O que move é o
corte de hub por tipo: derivado dá `concept=16` contra o absoluto 60, ou seja
**a L2 estava deixando passar como discriminante um conceito incidente a até 59
episódios**. O `concept_link` derivado também caiu de 0.75 para 0.55, e a lista
de moldura derivada tem 8 palavras contra as 31 escritas à mão.

Isto encerra a objeção de calibração de D30 da melhor forma possível: a versão
sem gabarito é a versão boa.

### 2. O `hop-profile` disse "não" para a L3 **antes** de ela rodar, e estava certo

Evidência que a L2 **errou**, por salto:

| categoria | n | salto 1 | salto 2 | salto 3 | inalcançável |
|---|---|---|---|---|---|
| multi-hop | 362 | 0.986 | 0.003 | 0.000 | 0.011 |
| open-domain | 111 | 0.973 | 0.000 | 0.000 | 0.027 |
| single-hop | 86 | 0.988 | 0.000 | 0.000 | 0.012 |

**99% da evidência errada já está a um salto.** E o oracle confirmou: L3 perde
em single-hop (0.881 vs 0.908) e em adversarial (0.781 vs 0.855), empata em
multi-hop.

**Este é o resultado mais importante do lote, e reenquadra o projeto de novo.**
O diagnóstico da L1 — "87–95% das falhas são turnos sem caminho no grafo" — era
sobre o grafo de entidades não tipado. Os slots tipados **resolveram isso por
completo**: com predicado, tipo, tempo e ator no vocabulário, cada episódio tem
~23 incidências e a pergunta semeia ~15 slots, então o salto 1 já toca quase
tudo. Alcançabilidade deixou de ser a restrição.

Ou seja: **o problema é inteiramente de ordenação, não de alcance.** E isso
casa com a outra medição que já estava incomodando — `recall_context` subiu
0.156 de L1 para L2 e o micro F1 subiu 0.016. Um passeio mais longo aumenta o
alcance de um grafo cujo alcance já está saturado; só pode adicionar ruído.

Registrado também: 84.365 pares de episódios compartilham 2+ slots não-hub. Os
4-ciclos que um mergulho quadrangular viraria faces **existem em profusão** — e
não ajudam, porque não é deles que a recuperação está precisando. É o argumento
de D31 fechado por medição em vez de por especulação.

**Erro de desenho meu, registrado:** a L3 shipou com `normalization: sym` *e*
`hops: 2`, então o delta L2→L3 confunde duas mudanças, apesar de o arquivo da
condição dizer que ela isola a propagação. Separar custa 4 minutos:
`fgl slots-oracle -C L3 --set propagation.hops=1 --set propagation.normalization=none`
tem que reproduzir a L2 **exatamente** (é a redução, agora verificável em dados
reais e não só no teste unitário), e trocar só a normalização diz se `sym`
sozinho vale alguma coisa.

### 3. A L4 é a melhor em multi-hop, e só nela

0.652 contra 0.641 da L2d: +0.011, o único ganho sobre a L2d em qualquer
categoria. **O canal de conexão funciona** — a conjunção do group Steiner é a
única coisa na linha inteira que mexeu multi-hop para cima. Modestamente, mas na
categoria certa e pelo motivo certo.

Custou open-domain (0.528 vs 0.565) e adversarial (0.825 vs 0.866). Os dois
números são suspeitos por causa dos bugs abaixo.

### 4. Dois bugs que uma rodada inteira pagou

**A L4 nunca recebeu o corpus de perguntas.** `_build_retriever` perguntava
`inspect.signature(cls)` se o construtor aceitava `question_corpus`; as duas
subclasses tomam `*args, **kwargs`, então a resposta era `False`. A L4 declara
`question_stop: derived` e rodou as 10 conversas com a lista legada de 31
palavras. Uma lista mais restritiva liga menos substantivos da pergunta — o que
é uma explicação própria para a queda em open-domain, **independente** do
`dense_seed`. Duas causas candidatas, uma observação: `dense_seed` fica como
está até a re-rodada limpa decidir.

Só foi pego porque a calibração registra proveniência: o relatório imprimiu
`question_noun_stop=fallback` ao lado de um config pedindo `derived`. O
princípio de D30 ("fallback é registrado, nunca silencioso") pagou o próprio
custo na primeira vez que importou.

**`steiner.abstain: true` nunca agiu.** O motivo era computado e reportado, mas
a *ação* continuava presa a `slots.abstain_on_empty_corner`, que a L4 fixa em
false. Sinal medido, publicado na tabela, e completamente inerte — visível
apenas como `empty ctx = 0` numa condição que dizia estar se abstendo. Agora
existe `_abstention_acts()`, que uma subclasse sobrescreve, e a L4 respeita o
próprio flag.

E aí a decisão vira uma medição em vez de um acidente: 10/446 adversariais
pegos para 28/1540 falsos positivos ≈ +0.002 contra −0.007 em micro. **Pior que
o teste de canto que ela substituiu.** Então `abstain: false`, dito no arquivo.

Não é fracasso, é achado: o custo de conexão é sinal real e contínuo, mas na
cauda 0.95 quase nunca dispara numa pergunta real (4 `far_apart` em 1986). Os
terminais de uma pergunta são muito mais próximos entre si do que slots
aleatórios — a distribuição nula, corretamente derivada e sem gabarito, é a
**classe de referência errada**. A certa seriam os terminais de *outras
perguntas*, o que é outro estimador e outro experimento.

### Veredito para o run com LLM

**L2d.** É melhor que a L2 em todas as categorias, é a condição que carrega o
argumento metodológico, e o delta L2→L2d é limpo. A L3 não deve rodar. A L4
só depois de a re-rodada do oracle confirmar que os bugs eram a causa das duas
quedas.
