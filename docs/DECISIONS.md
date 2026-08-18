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
