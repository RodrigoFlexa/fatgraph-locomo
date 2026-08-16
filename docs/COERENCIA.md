# Auditoria de coerência da especificação

Este documento registra a verificação pedida antes da implementação: o que na
spec está correto, o que está matematicamente errado, o que é ambíguo e o que é
inviável como escrito. Cada item tem um **veredito** e a **correção adotada**.

Resumo: o desenho conceitual é sólido e a espinha dorsal (α, σ, φ, faces, Euler)
está correta. Encontrei **4 erros que quebrariam a execução ou os números**
(C1, C3, C6, C7), **4 ambiguidades** que precisavam de convenção explícita
(C2, C4, C11, C12), **3 desalinhamentos com o protocolo oficial do LoCoMo**
(C8, C10, C13) e **1 risco estrutural** só visível depois de rodar (C9).

Legenda: 🔴 erro · 🟡 ambiguidade · 🔵 desalinhamento · ⚪ observação

---

## C1 🔴 A fórmula de Euler da spec só vale para grafos conexos

**Spec (§1):** "Fórmula de Euler: V − E + F = 2 − 2g fornece o gênero g. Use-a
como invariante de verificação."

**Problema.** `V − E + F = 2 − 2g` vale para um fatgraph **conexo**. Para `C`
componentes conexas a fórmula é

```
V − E + F = 2C − 2g
```

Um grafo de memória do LoCoMo é rotineiramente desconexo: entidades citadas uma
única vez em sessões distintas formam ilhas. Aplicando a fórmula literal a um
grafo com 2 componentes esféricas obtém-se `g = −1`; com uma componente extra,
`g = −2`. O "invariante de verificação" passaria a acusar violação em grafos
perfeitamente válidos — ou, pior, seria desligado por parecer quebrado.

**Correção.** `FatGraph.euler()` roda union-find sobre os vértices, calcula
`g_i` por componente e devolve `genus = Σ g_i`, além de `C`. A fórmula literal
da spec continua exposta como `EulerStats.genus_connected_formula` para
rastreabilidade — e o teste `test_disconnected_graph_needs_the_per_component_formula`
verifica que ela dá `−1.0` exatamente no caso em que a versão correta dá `0`.

Verificado empiricamente: no smoke run com dados reais, `C` variou de 1 a >1
conforme a conversa e o resultado do resolvedor de entidades.

---

## C2 🟡 Vértices isolados quebram a integralidade do gênero

**Problema.** Um vértice de grau 0 não incide em nenhuma meia-aresta, logo não
participa de nenhum ciclo de φ e contribui `F += 0`. Para um único vértice
isolado: `V=1, E=0, F=0` ⇒ `2C − 2g = 1` ⇒ `g = 0.5`. Gênero fracionário.
Vértices isolados aparecem naturalmente (curadoria que remove a última aresta,
entidade criada e depois nunca ligada).

**Correção.** Convenção padrão de ribbon graphs: o vértice é um disco e seu
bordo é uma face. `faces()` emite uma **face trivial** (`half_edges = ()`) por
vértice de grau 0. Recupera-se `V=1, E=0, F=1 ⇒ g=0`.
Teste: `test_isolated_vertex_contributes_a_trivial_face`.

---

## C3 🔴 "Face de comprimento 2" não é sinônimo de redundância

**Spec (§2):** "`collapse_bigon(face_id)` — colapsa face de comprimento 2
(merge das **duas arestas**, união de proveniências)".
**Spec (§3.5):** "para cada face de comprimento 2 (...) LLM julga redundância".
**Spec (H3):** "curadoria de faces de comprimento 2 reduz redundância".

**Problema.** Uma face de comprimento 2 tem dois casos combinatoriamente
distintos:

| caso | topologia | significado semântico |
|---|---|---|
| (a) duas arestas **paralelas** distintas entre o mesmo par de vértices | bígono genuíno | possível redundância |
| (b) uma **única** aresta percorrida duas vezes por φ | folha / ponte | nenhuma redundância |

O caso (b) é o mais comum de longe: **toda aresta pendente gera uma face de
comprimento 2**. Num grafo extraído de conversa, uma fração enorme das arestas
liga um hub (falante) a uma entidade mencionada uma só vez — todas folhas.
A spec fala em "merge das duas arestas", que no caso (b) simplesmente não
existe: só há uma. Implementada ao pé da letra, a curadoria ou explodiria com
`IndexError`/comportamento indefinido, ou apagaria memórias legítimas — e H3
seria testada contra o fenômeno errado.

**Correção.** `Face.is_leaf_face` distingue os dois casos.
`collapse_bigon` levanta `NotABigonError` no caso (b) e também quando as duas
arestas não ligam o mesmo par de vértices. `Curator.collapse_redundant_bigons`
só considera bígonos genuínos, e `stats()` reporta `n_leaf_faces` e
`n_bigon_faces` separadamente, para que H3 possa ser avaliada sobre o
denominador certo.
Testes: `test_leaf_face_traverses_the_same_edge_twice`,
`test_collapse_refuses_a_leaf_face`.

---

## C4 🟡 A verificação de gênero no `collapse_bigon` é tautológica

**Spec (§2):** "`collapse_bigon` DEVE recomputar `euler()` antes/depois e lançar
exceção se o gênero mudar."

**Observação.** Colapsar um bígono genuíno remove 1 aresta e funde 2 faces em 1:
`V − (E−1) + (F−1) = V − E + F`. A característica de Euler é preservada por
construção; o gênero **nunca** pode mudar se a operação estiver correta. A
verificação portanto não é um critério de curadoria — é um *guard de regressão*
contra bug de implementação.

**Correção.** Mantida (é útil), mas fortalecida: `_assert_topology_preserved`
verifica `C`, `genus`, `ΔE = −1` **e** `ΔF = −1`. Só assim o guard detecta de
fato uma implementação errada. Documentado como guard, não como semântica.
Teste: `test_bigon_collapse_preserves_genus_and_merges_provenance`.

---

## C5 🟡 Colisão de nomes: condição "F1" × métrica F1

**Spec (§6):** as condições fatgraph chamam-se `F1`, `F2`, `F3`; a métrica
principal do LoCoMo também se chama F1.

**Problema.** "F1 de F1" é ilegível em tabelas, gráficos e nomes de diretório —
e é fonte real de erro na hora de ler resultados.

**Correção.** Condições renomeadas para `G1`/`G2`/`G3` (G de *graph*), com o id
da spec preservado no comentário de cada YAML. `B1`/`B2`/`B3` ficam como estão.

---

## C6 🔴 Id de face por *conjunto* de arestas colide

**Spec (§2):** "`faces()` (...) cada uma com id estável = hash ordenado do
**conjunto** de edge-ids".

**Problema.** Duas coisas quebram:

1. Um conjunto perde a ordem. Faces distintas podem ter o mesmo conjunto de
   arestas — no triângulo de referência as duas faces têm exatamente
   `{e1, e2, e3}`, mudando só a orientação. O id colidiria já no exemplo
   sintético que a própria spec manda montar.
2. Um conjunto perde a multiplicidade. Uma face que percorre a mesma aresta duas
   vezes (caso C3b) teria id igual ao de uma face menor.

E o id é usado como chave de "estável há ≥ k sessões" na consolidação (§3.6);
colisão aqui contamina diretamente a decisão de consolidar.

**Correção.** `face_id()` usa a **menor rotação lexicográfica da sequência
cíclica ordenada** de edge-ids. É invariante à meia-aresta de partida (que é o
que "estável" precisa) e sensível a inserções na face (que é o sinal que a
consolidação quer detectar).
Teste: `test_face_id_is_rotation_invariant_and_order_sensitive` — inclui o par
`["e1","e2","e1","e3"]` vs `["e1","e1","e2","e3"]`, que um hash de conjunto
confundiria.

---

## C7 🔴 O código oficial do LoCoMo quebra na categoria 5

**Spec (§4):** "verificar no código de avaliação do LoCoMo qual é a string
esperada para adversariais e usar a convenção deles".

**Achado.** Verificado no repositório oficial (branch `code`):

* `task_eval/evaluation.py::eval_question_answering` faz
  `answer = str(line['answer'])` para **toda** categoria, antes do dispatch;
* as 446 perguntas adversariais guardam `adversarial_answer`, e **444 delas não
  têm a chave `answer`**. A função levanta `KeyError` nesses itens;
* a regra de pontuação da categoria 5 é, literalmente,
  `'no information available' in output.lower() or 'not mentioned' in output.lower()`.

Ou seja: a convenção da spec (`Not mentioned in the conversation`) **está
correta**, mas o código oficial não roda sobre o dataset oficial sem um remendo.

**Correção.** `src/fgl/evaluation/scorer.py` vendoriza as funções oficiais
(`normalize_answer`, `f1_score`, `f1`, dispatch por categoria) verbatim, e o
loader (`fgl/data/locomo.py::_parse_question`) normaliza o gold da categoria 5 para a
string de abstenção. A regra de pontuação da categoria 5 é idêntica à oficial,
então o número reportado é comparável.
Testes: `test_adversarial_rule_is_a_substring_check`,
`test_every_category_is_present_and_adversarial_gold_is_normalised`
(esse último roda contra o arquivo oficial e confirma as 446 adversariais).

---

## C8 🔵 O protocolo oficial da categoria 5 é múltipla escolha; o nosso não é

**Achado.** `task_eval/gpt_utils.py` (linhas 246–252) transforma cada pergunta
adversarial em **múltipla escolha de duas opções**: `(a) Not mentioned in the
conversation` / `(b) <a resposta plausível mas falsa>`, com ordem sorteada.

**Consequência.** A spec pede resposta livre. Resposta livre é
**significativamente mais difícil** que escolher entre duas opções: o modelo
precisa decidir abster-se sem que a abstenção lhe seja oferecida.

**Correção.** Mantida a formulação livre (é a que testa H4 de verdade), mas isso
está registrado explicitamente porque significa que **os números de categoria 5
deste repositório não são diretamente comparáveis aos publicados no paper do
LoCoMo**. A regra de pontuação é a mesma; o protocolo de perguntar, não.

---

## C9 ⚪ Risco estrutural: `sigma-time` produz pouquíssimas faces, e gigantes

**Achado empírico** (smoke run, extração offline, duas conversas independentes):

| condição | conversa | V | E | **F** | g | maior face |
|---|---|---|---|---|---|---|
| G1 (sigma-time) | conv-26 | 56 | 130 | **2** | 37 | 230 |
| G1 (sigma-time) | conv-30 | 62 | 134 | **2** | 36 | 214 |
| G2 (+curadoria/consolidação) | conv-26 | 56 | 134 | 4 | 38 | 193 |
| G2 (+curadoria/consolidação) | conv-30 | 62 | 136 | 2 | 37 | 251 |
| G3 (sigma-agent) | conv-26 | 56 | 136 | 4 | 39 | 176 |
| G3 (sigma-agent) | conv-30 | 62 | 139 | 5 | 37 | 212 |

O padrão se repete nas duas conversas: **2 a 5 faces**, uma delas engolindo
quase todas as memórias.

**Por que.** Para um grafo fixo, `F = 2C − 2g − V + E`; com `E ≫ V` (hubs: os
dois falantes entram em quase todo fato), `F` só cresce se σ for escolhido para
manter o gênero baixo. `sigma-time` insere sempre logo após a meia-aresta mais
recente, o que num hub equivale a ordem cronológica pura — e isso é próximo do
pior caso para contagem de faces.

**Consequência para as hipóteses.** Uma "face" de comprimento 230 não é uma
narrativa; é a memória inteira. Nesse regime H1 degenera: `walk_face` com
orçamento de 2000 tokens devolve, na prática, "as ~10 memórias seguintes ao
âncora dentro da trilha", isto é, uma **janela local** — o que ainda pode ser
melhor que k-NN (a ordem importa), mas não é o mecanismo que a hipótese
descreve. E H2 fica sobrecarregada: o sigma-agent não está só melhorando a
ordem, está determinando se existe estrutura de faces.

**Não é um erro da spec** — é uma propriedade do desenho que só aparece rodando.
Registrado aqui porque muda a leitura dos resultados. Mitigações já no código:
`walk_face` respeita orçamento; `stats()` reporta o histograma completo por
sessão, então a degeneração é observável em `metrics.json` sem trabalho extra.
Mitigação possível na fase 2: penalizar gênero na política de inserção, ou
limitar o grau dos hubs (nós-falante) por particionamento temporal.

Efeito colateral de engenharia: faces longas tornaram quadrático o cálculo do id
canônico de face. Corrigido com o algoritmo de Booth — ver `DECISIONS.md` D21.

---

## C10 🔵 O truncamento do B1 é desnecessário com gpt-4o-mini

**Spec (§6):** "B1 full-context: conversa inteira no prompt (truncar pela janela
do modelo, do início se necessário; documentar)".

**Medição** (contagem real sobre as 10 conversas):

| conversa | turnos | tokens estimados |
|---|---|---|
| conv-30 | 369 | 11 678 |
| conv-26 | 419 | 15 698 |
| conv-43 | 680 | **23 908** (máximo) |

O máximo é ~24k tokens contra os 128k de janela do `gpt-4o-mini`. **Nenhuma
conversa é truncada.** O ponto importa porque truncar "do início" descartaria
justamente as primeiras sessões, onde vive a evidência da maioria das perguntas
multi-hop — o que enviesaria B1 para baixo e inflaria artificialmente o ganho
das condições de memória.

**Correção.** O guard existe (`baselines.full_context_max_tokens: 110000`) e
conta quantas conversas truncou (`truncated_conversations`), mas com o
deployment padrão o contador fica em zero. Documentado em `fgl/baselines/full_context.py`.

---

## C11 🟡 Uma face é um ciclo — não tem "vértices extremos"

**Spec (§3.6):** "o resumo vira aresta level=2 entre os **vértices extremos**".

**Problema.** Face = órbita de φ = trilha **fechada**. Não há extremos.

**Correção.** Convenção documentada (`fgl/memory/curation.py::_extremes`): `v_start` é o
vértice da meia-aresta canônica inicial da face; `v_end` é o primeiro vértice
**distinto** encontrado a partir da metade da travessia — o par maximamente
separado ao longo da trilha. Se todos coincidem, a face não é consolidada.

Efeito colateral que a spec não menciona: **a aresta de consolidação muda a
topologia**. Uma corda inserida ao lado das próprias meias-arestas da face a
divide em duas (`E+1`, `F+1`, gênero preservado); se ela acabar ligando faces
diferentes, o gênero sobe. `fgl.memory.Curator.consolidate` registra `euler_before`/
`euler_after` em JSONL e emite `consolidation_genus_change` quando isso ocorre.

---

## C12 🟡 Incongruência: a spec compara faces, o fenômeno é entre fatos

**Spec (§3.4):** "se a nova face contradisser outra face sobre os mesmos
vértices (julgamento por LLM)".

**Problema.** Comparar duas faces inteiras via LLM é caro (faces de comprimento
230, ver C9) e mal definido: uma face contém dezenas de fatos sobre dezenas de
vértices; "contradizer sobre os mesmos vértices" não é um predicado avaliável
sobre esse objeto.

**Correção.** A contradição é julgada onde ela mora: entre a **aresta nova** e
as **arestas irmãs** que ligam o mesmo par de vértices (as únicas que podem
contradizê-la sobre "os mesmos vértices"), limitado às 3 mais recentes para
limitar custo. O prompt distingui explicitamente contradição de mudança
legítima ao longo do tempo, usando os timestamps. Nada é apagado: a aresta é
marcada `incongruente` e ganha `meta.conflicts_with`.

---

## C13 🔵 Detalhes do protocolo oficial que a spec omite

Todos verificados no repositório oficial e replicados, porque afetam o F1
diretamente:

1. **Categoria 2 (temporal)** — o pipeline oficial acrescenta à pergunta
   `" Use DATE of CONVERSATION to answer with an approximate date."`
   (`gpt_utils.py:243`). Sem isso o modelo responde em formato incompatível com
   o gold e o F1 despenca. Replicado em `Question.prompt_question()`.
2. **Categoria 3 (open-domain)** — o gold é truncado no primeiro `;`
   (`evaluation.py:204`). Replicado em `score_question`.
3. **Categoria 1 (multi-hop)** — usa a variante multi-resposta de F1, que separa
   predição e gold por vírgula. Replicado em `f1_multi`.
4. **Legendas de imagem** — turnos com `img_url` entram no contexto como
   `[shares <blip_caption>]` (`evaluation_stats.py:21`). Replicado em
   `Turn.rendered`; a extração de fatos também recebe a legenda.
5. **Stemming** — o F1 oficial aplica `PorterStemmer` do NLTK. Se o NLTK não
   estiver instalado, o scorer aqui cai para stemmer identidade **e grava
   `"stemmer": "identity"` no `metrics.json`**, para que ninguém compare
   números produzidos com stemmers diferentes sem perceber.

---

## C14 ⚪ Viabilidade e custo

* **Python 3.11+** (spec §pré-âmbulo) — o código usa `from __future__ import
  annotations` e roda em **3.10+**. Sem custo, amplia compatibilidade.
* **FAISS** (spec §2, §4) — mantido como backend opcional; o padrão é uma busca
  exata em numpy. Para ~10³ meia-arestas por conversa, FAISS não traz ganho e
  adiciona uma dependência pesada. A interface `VectorIndex` é a mesma nos dois
  casos e cai para numpy sozinha se o import falhar.
* **Custo estimado das 6 condições, 10 conversas, 1986 perguntas, gpt-4o-mini:**

  | fase | chamadas | tokens (ordem) |
  |---|---|---|
  | extração (uma vez, compartilhada) | ~272 | ~0,9 M |
  | QA das condições G1/G2/G3 e B2/B3 | ~9 900 | ~12 M |
  | sigma-agent (G3) | ~2 600 | ~4 M |
  | incongruência + curadoria + consolidação | ~2 000 | ~3 M |
  | **B1 full-context** | 1 986 | **~41 M** |

  B1 domina o custo (~2/3 do total) — o que justifica a spec deixá-lo por
  último. Total na ordem de 60 M tokens; a preços correntes de gpt-4o-mini isso
  fica na casa de uma dezena de dólares. O cache agressivo por hash de prompt
  torna re-execuções praticamente gratuitas.

---

## C15 ⚪ Hipótese H4 tem elo fraco com o mecanismo

**Spec:** "(H4) faces incoerentes/estado Incongruente melhoram a recusa em
perguntas adversariais."

**Observação.** As 446 perguntas adversariais do LoCoMo não são sobre
*contradições* na conversa — são sobre fatos **nunca mencionados** (ex.: "What
did Caroline realize after her charity race?" quando não houve corrida). O
mecanismo que produz a recusa correta nesses casos é "as faces recuperadas não
contêm a informação", não o estado `incongruente`.

A detecção de incongruência continua valiosa (evita responder com um fato
superado), mas **H4 como escrita provavelmente não será confirmada por
construção do benchmark**. Os dois mecanismos foram separados no código e são
medidos separadamente: `abstention_rate` por categoria e
`edges_by_state.incongruente` nas estatísticas de grafo. Recomendação: reformular
H4 como "a taxa de abstenção correta cresce quando a recuperação por faces
retorna contexto vazio ou incongruente", que é verificável com o que já é logado.

---

## O que na spec está correto e foi seguido sem mudança

* Definição de α (involução sem ponto fixo), σ (ordem cíclica) e φ = σ∘α — e a
  identificação de faces com órbitas de φ. Correto e implementado literalmente.
* `walk_face` com orçamento, devolvendo sequência **ordenada** — correto e
  central: é o que diferencia de k-NN.
* Whitehead flip preserva o gênero e |F| — correto (é um movimento de spine da
  mesma superfície) e verificado por teste no grafo teta.
* B3 como ablação crítica com **os mesmos fatos** de G1 — corretíssimo e
  implementado por cache de extração independente de condição, com teste
  dedicado (`test_facts_cache_is_condition_independent`).
* Separação estrita "memória só lê diálogos / QA só lê memória" — implementada e
  estrutural: a fase de QA recebe apenas o `FatGraph` carregado do disco.
* Credenciais só por variável de ambiente, deployment no YAML — seguido à risca.
* Reportar **todas** as categorias, sem filtrar nem subamostrar — seguido; o
  teste `test_the_whole_official_dataset_loads` trava em 1986 perguntas.
