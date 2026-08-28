# Reflexão — modelagem do conhecimento no M1 (MECA): o que quebrou, o que já foi corrigido, o que falta

Data: 2026-08-28. Escopo: só o M1 (flat). Nada aqui fala do M2 — a comparação de
leitores só vale depois que o store estiver limpo.

## 1. O problema, para não perder de vista

MECA aposta que o gargalo da linha L nunca foi recuperação: com
`recall_context = 1,0` o multi-hop travava em f1 0,476 (D36). A evidência
chegava inteira ao prompt e a resposta não saía porque a memória guardava
**ponteiro para texto** — slots de superfície incidentes a turnos — e cada
pergunta tinha de re-derivar o fato do diálogo cru, sob ruído e orçamento, toda
vez. A aposta do M1 é trocar a unidade: guardar o resultado da compreensão (a
proposição atestada: sujeito · predicado · objeto · qualificadores · dois
relógios · modalidade/polaridade · evidência obrigatória) em vez do texto que a
sustenta. Isso é modelagem de conhecimento de verdade — decidir o que existe,
quando, com que grau de certeza — não é engenharia de recuperação com um passo
a mais.

O ponto fino, que vale repetir porque a rodada quebrada o escondeu: trocar a
unidade não elimina o problema de **identidade**. Ao contrário, ele piora,
porque agora a memória precisa decidir se duas menções falam da mesma pessoa
ANTES de guardar qualquer coisa, e um erro aqui não derruba uma resposta — ele
contamina toda proposição que aquele sujeito jamais vier a ter.

## 2. Como o M1 modela conhecimento hoje

Três chamadas de LLM por passagem, cada uma uma ablação isolável:
`meca_extract` (o que a passagem afirma, com span) → `meca_infer` (o que se
segue e não foi dito, marcado `derived`) → `meca_verify` (a alegação é
acarretada pelo span citado?). Um veredito malformado rejeita — a proposição
não entra. Isso é o argumento de segurança inteiro contra invenção na
extração, e não muda nesta reflexão.

O que muda de mão em mão é a **consolidação** (`src/fgl/memory/consolidate.py`),
que transforma o monte de alegações extraídas em um estado:

1. **resolução de entidades** — decidir que menções são a mesma coisa;
2. **deduplicação** — a mesma alegação dita duas vezes vira uma proposição com
   duas evidências, não duas proposições competindo por orçamento;
3. **funcionalidade do predicado**, estimada do corpus (`predicate_functionality`)
   — "mora em" admite um valor por vez, "leu" admite muitos, sem ontologia e
   sem rótulo;
4. **linha do tempo** — uma alegação mais nova num predicado funcional fecha a
   mais antiga (`apply_supersession`);
5. **vínculos analíticos** — `elaborates`/`contradicts`, guardados, nunca
   resolvidos silenciosamente (`link_propositions`).

Todo limiar aqui é quantil derivado do corpus com piso absoluto, nunca um
literal varrido contra o rótulo — a mesma disciplina da D30. Isso continua
certo e não é o que quebrou.

## 3. O que quebrou (o diagnóstico, como aconteceu)

A rodada registrada em `results/M1-meca-flat/metrics.json` (8,77 M tokens,
5,9 h, 1.986 perguntas): f1_micro **0,3259**, multi-hop 0,2214, open-domain
0,0794, adversarial 0,3161. Setenta por cento das perguntas não ligavam a
nenhuma entidade da memória.

A causa, confirmada nos próprios grafos: resolução de entidade fazia união
transitiva sobre similaridade de embedding, tratando QUALQUER sujeito extraído
— pessoa, evento, descrição, sintagma possessivo — como candidato à mesma
resolução de identidade. Isso funde "Caroline", "Melanie" e frases como "o
apoio dos amigos e mentores de Caroline" no mesmo nó, com grau na casa das
centenas. E o vínculo analítico `contradicts` morava dentro de
`qualifiers["_links"]` como string — entrava no `statement()`, no embedding, na
aresta — então a topologia que qualquer leitor veria já estava poluída antes de
qualquer rotação entrar em jogo.

Eu verifiquei isso direto no artefato: `artifacts/graphs/M1-meca-flat/conv-26.json`,
vértice `v1`, sujeito ainda é `"Caroline's friends, family and mentors' support"`,
e a qualifier `_links` carrega seis `contradicts:` concatenados em texto. **Este
grafo é o artefato ANTES da correção** — a evidência abaixo mostra que ele não
foi regenerado ainda (seção 7).

## 4. Auditoria do código atual: o que já foi corrigido

Lendo `consolidate.py`, `propositions.py` e `comprehend.py` como estão agora
(editados hoje, ainda não commitados — o último commit que tocou
`consolidate.py` é `242ee063 MECA`, de ontem), a maior parte do Nível A da
lista anterior já está no código, e de um jeito mais conservador do que o
pedido original:

**Identidade deixou de usar embedding.** `resolve_entities` não faz mais
clusterização semântica nenhuma. A única operação permitida é
`_is_short_form`: prefixo de palavra inteira ("mel" é forma curta de "melanie
carter" porque `"melanie carter".startswith("mel ")`... na prática o teste
exige o token inteiro, então é "melanie" prefixo de "melanie carter", não
substring solta). Sem union-find: cada candidato só pode casar com uma forma
já aceita, não em cadeia. Participantes da conversa
(`store.entity_anchors`, populados em `comprehend.py:294-295` a partir de
`conv.speaker_a`/`speaker_b`) sempre ganham o desempate e nunca entram como
candidatos a fundir com uma descrição. Isso é estritamente mais seguro que o
Nível A #1/#3 pedia — não é só "sem transitividade", é "sem semântica
nenhuma" para essa decisão específica. O preço é nomeado na seção 5.

**Aliases ficaram consultáveis.** `store.entity_aliases` e
`store.alias_to_canonical` existem e são povoados na resolução; `knows_entity`/
`entity_key` consultam essa tabela. `parse_question`
(`src/fgl/retrieval/meca.py:137`) casa a pergunta contra `store.knows_entity`,
que por sua vez resolve pelo alias — não só pela chave canônica. Nível A #2,
feito.

**`_links` virou campo de primeira classe.** `Proposition.links:
dict[str, list[str]]` é um atributo próprio, fora de `qualifiers`.
`statement()`, `roles()` e `arguments()` — as três superfícies que alimentam
texto, embedding e aresta — nunca leem `links` nem qualquer qualifier que
comece com `_`. Há migração de compatibilidade (`coerce_proposition` extrai
`_links` de grafos antigos e realoca para o campo novo), então o formato velho
não quebra ao ser lido, só nunca mais é escrito. Nível A #4, feito, e é o mais
importante dos cinco porque destrava o resto: sem isso, qualquer correção de
contradição continuaria vazando para o contexto de resposta.

**Contradição parou de ser "objetos diferentes".** `link_propositions` agora
exige: predicado funcional (calculado no mesmo lote, pela mesma estimativa
suave da funcionalidade) **e** objetos diferentes **e** as duas proposições
`is_current` **e** `is_factual` **e** mesma polaridade **e** os dois têm objeto
**e** as janelas de validade se sobrepõem (`a.when().overlaps(b.when())`).
Isso é o Nível A #5 quase inteiro — falta só checar modalidade igual entre os
dois lados, ver seção 5.

**A disciplina de sujeito entrou no prompt.** `prompts/meca_extract.txt` agora
proíbe explicitamente sujeito como frase descritiva ou possessiva ("Keep a
person, object, organisation or event as the subject itself: do not replace it
with a related description... such as 'NAME's support network'") e resolve
"I"/"my" para quem fala o turno. Isso é a metade suave do Nível B #6 — a
recomendação certa foi não impor regra dura de contagem de palavras, e não foi
imposta. A rede de segurança REAL, porém, não é o prompt — é a resolução
lexical da seção anterior: mesmo que a extração desobedeça e ainda produza
"o apoio de Caroline", esse sujeito não compartilha prefixo com "Caroline" e
**não pode mais fundir com ela**, prompt à parte. Isso é defesa em
profundidade bem desenhada.

## 5. Duas tensões conceituais que a correção não fecha sozinha

**A troca é precisão por recall de alias, e vale nomear o custo.** Cortar toda
semântica da resolução de identidade impede a fusão categórica que corrompeu a
rodada anterior — mas também significa que apelidos que não são prefixo lexical
da forma completa ("Bob" para "Robert", "Peggy" para "Margaret", ou uma menção
consistente como "minha irmã" para alguém já nomeado) nunca mais vão se
resolver ao mesmo nó. Isso é conservador na direção certa (melhor perder um
alias do que fundir duas pessoas), mas é uma perda de recall real e silenciosa
— nenhuma métrica hoje mede quantas entidades ficam fragmentadas por esse
motivo. Se isso continuar custando F1 depois da rodada nova, o próximo passo
NÃO é reintroduzir transitividade — é uma lista de aliases pequena, verificada
por LLM par a par (nunca em cadeia, com teto de tamanho de grupo), o mesmo
espírito do `entity_anchors` mas para apelidos não óbvios.

**Modalidade compatível não é checada na contradição.** `is_factual` aceita
`{asserted, reported}` (`propositions.py:75`). Duas proposições podem
contradizer hoje com uma `asserted` e a outra `reported` — alguém disse que X
aconteceu, outra pessoa relatou que X foi diferente. Isso pode ser
legitimamente uma contradição ou pode ser só relato de segunda mão divergindo
de um fato direto, que é uma categoria de informação diferente. Não é bug —
é uma decisão de modelagem que ainda não foi tomada conscientemente. Vale
decidir e documentar, não necessariamente mudar.

## 6. Dois problemas concretos, achados agora ao ler o código

Não fiquei só na leitura — reproduzi as duas peças centrais isoladamente
(`python3` direto contra `src/fgl/memory/consolidate.py`, sem rodar a suíte
inteira porque o device não tem `pytest`). Duas coisas quebraram e ainda não
foram percebidas porque **a suíte não rodou desde a edição de hoje**:

- `resolve_entities(store, embedder)` com `"Melanie Carter"` e `"Melanie"`
  agora resolve para **`"melanie"`** (a forma curta vence, empate por
  contagem e depois por comprimento ascendente). O teste
  `tests/test_meca.py::test_entity_resolution_collapses_short_forms` ainda
  afirma `subjects == {"melanie carter"}` com o comentário "the longest
  surface wins". Isso é um teste desatualizado, não um bug de produção — o
  comportamento novo (forma curta vence) é exatamente o que a lista de
  correções pediu ("nunca a mais longa"). Mas o teste, do jeito que está,
  falha, o que quer dizer que ninguém confirmou o comportamento novo
  rodando a suíte.
- `link_propositions(store)` chamado sem o segundo argumento — exatamente
  como `tests/test_meca.py::test_a_contradiction_is_kept_and_flagged_not_resolved`
  chama — devolve **zero contradições** para o par clássico "Ana trabalha na
  Acme" / "Ana trabalha na Globex" no mesmo mês, porque
  `functional_predicates` default para conjunto vazio e a nova regra exige
  que o predicado esteja nesse conjunto. O caminho de produção
  (`consolidate()`) passa `functional_predicates` corretamente, então isto
  provavelmente não é um bug ao vivo — mas é um contrato de função que mudou
  de significado sem um valor-padrão que reproduza o comportamento antigo, e
  é fácil esquecer de recalcular e passar esse conjunto em qualquer segundo
  lugar que chame `link_propositions` diretamente.

Nenhuma das duas invalida o desenho da correção. As duas dizem a mesma coisa:
**o código mudou mais rápido do que a suíte foi reconferida**, e é exatamente
o tipo de coisa que `pytest tests/test_meca.py` pega em segundos, de graça,
sem gastar um token de LLM.

## 7. O que ainda não foi feito

- **Lematização de predicado para o MECA.** A deduplicação continua só por
  limiar de embedding sobre a forma de superfície do predicado
  (`deduplicate`, em `consolidate.py`); não há chave lematizada como a que já
  existe para o L2/slots (`ner.py`, `tok.lemma_`). "went to" e "attended"
  continuam duas formas a menos que o embedding os aproxime o bastante — Nível
  B #7 segue pendente, e é uma correção de precisão de dedup, não de
  segurança.
- **Nenhum health gate.** O `sanity` que já existe (`report.py:sanity_banner`)
  só pega respostas todas idênticas ou falha de parse de JSON. Não existe
  ainda: alarme de grupo de alias grande demais, contagem de contradições por
  proposição/conversa fora de ordem de grandeza plausível, checagem de que
  nenhum `links`/`_links` aparece em `statement()` renderizado, ou taxa de
  `unbound_question` acima de um limiar. É exatamente o tipo de portão que a
  D35/D36 já usaram noutro lugar (o portão do atestado matou uma hipótese de
  graça, com zero chamada de LLM) e que este ponto do projeto pede de novo.
- **Nenhuma rodada nova foi feita.** Isto é o fato operacional mais
  importante desta reflexão: `results/M1-meca-flat/` e
  `artifacts/graphs/M1-meca-flat/` no disco **são o estado de ANTES da
  correção**. Confirmado direto no artefato (seção 3). Os 0,3259 de F1, os
  70% de `unbound_question`, as milhares de contradições — nenhum desses
  números diz nada sobre o código que existe agora. Ainda não há UMA linha de
  evidência de que a correção funciona.

## 8. Prevenção — blindar o pipeline antes de aceitar a próxima rodada como experimento

Um portão de saúde, rodável sobre o grafo já construído, sem LLM, antes de
aceitar qualquer F1 como resultado:

1. **Tamanho de componente de identidade.** Nenhum nó de entidade deveria ter
   grau ordens de magnitude acima da mediana da conversa. Um limiar por
   quantil do próprio grafo (a mesma receita de `pairwise_quantile`) recusa e
   reporta em vez de deixar passar — a mesma disciplina que já existe para os
   limiares de consolidação, aplicada como verificação e não como parâmetro.
2. **Contagem de contradições por proposição/conversa.** Um número por
   conversa muitas ordens de grandeza acima do que uma leitura de dez
   proposições sugeriria plausível é sinal de over-merge a montante, não de
   corpus contraditório de verdade.
3. **Vazamento de metadado analítico.** Grep determinístico: nenhum
   `statement()` gerado deveria conter `contradicts:`/`elaborates:`/`_links`.
   Isso é barato e pega exatamente a classe de bug que causou a explosão de
   tokens na rodada anterior.
4. **Taxa de `unbound_question`.** Acima de um piso (o quê, é uma decisão a
   tomar — mas certamente bem abaixo de 70%) bloqueia aceitar a rodada como
   medição de recuperação/resposta; sinaliza problema de resolução de
   identidade, não de geração.

Os quatro juntos custam segundos e zero chamada de LLM — a mesma economia que
já pagou dividendo nesta linha de pesquisa toda vez que foi aplicada (hop-profile
matou a L3 de graça; o portão do atestado matou a rota estrutural de graça).

## 9. Ordem recomendada dos próximos passos

1. `pytest tests/test_meca.py` — antes de qualquer outra coisa. As duas
   quebras da seção 6 são baratas de resolver e vão dizer se há uma terceira
   que eu não vi por leitura estática.
2. Isolar o efeito da consolidação, de graça. `prompts/meca_extract.txt` foi
   editado hoje (a chave de cache do `LLMClient` inclui o texto do prompt —
   confirmado em `src/fgl/llm/client.py`), então qualquer rodada nova já paga
   extração de novo. Para medir SÓ o efeito da consolidação, sem gastar
   tokens: `git stash push -- prompts/meca_extract.txt` (volta o prompt para
   a versão do commit `242ee063`, que ainda bate com o cache de
   `.cache/llm`), reconstruir os grafos com `--force`, olhar os quatro
   portões da seção 8 nos grafos novos. Depois `git stash pop` e pagar a
   rodada completa com o prompt novo também.
3. Implementar os quatro portões da seção 8 no pipeline de avaliação, não só
   como script avulso — para que a próxima rodada corrompida seja pega antes
   de queimar 5,9 h e 8,77 M tokens de novo.
4. Só então rodar o Portão 1 do D37 (fidelidade da extração: fração dos
   turnos de evidência anotados com pelo menos uma proposição cujo span cai
   dentro) e medir F1 de novo.
5. Lematização de predicado (Nível B #7) como ganho incremental de precisão
   de dedup, depois que houver um número novo para comparar contra.

## 10. A lição, no nível que generaliza

O erro de desenho original não foi "threshold errado" — foi usar uma operação
**métrica e transitiva** (similaridade de embedding + união) para uma decisão
que é **categórica** (esta menção é esta pessoa, sim ou não). Qualquer
relação de proximidade, por mais bem calibrada, tende a formar um componente
gigante quando fechada por transitividade sobre um corpus grande o bastante —
não é uma falha de calibração, é uma propriedade estrutural de fechar
transitivamente uma relação ruidosa. A correção certa não foi um limiar
melhor: foi trocar a classe de operação (só prefixo lexical, sem cadeia,
âncoras vencem sempre) e aceitar o custo de recall que isso traz, em vez de
tentar calibrar o problema para fora.

A segunda lição é sobre separação de camadas: `contradicts`/`elaborates` são
metadado ANALÍTICO sobre o estado da memória, não conteúdo FACTUAL sobre o
mundo. Misturar as duas coisas no mesmo campo (`qualifiers`) foi o que deixou
a poluição vazar para texto, embedding e aresta ao mesmo tempo por um único
descuido de tipagem. Separar por construção (campo `links` que `statement()`
nunca lê) é mais forte que qualquer disciplina de "lembrar de filtrar" — é
o tipo de correção que se mede uma vez e nunca mais se paga de novo.

---

# Addendum — expectativa no LoCoMo e riscos de execução (2026-08-28, mesma sessão)

Pergunta que motivou este addendum: dado tudo que foi corrigido na seção acima,
a proposta está sólida olhando para o benchmark? Qual o comportamento esperado
categoria a categoria, e o que pode falhar na hora de rodar de verdade, antes
de subir para o servidor? Escopo: só o que generaliza — nada aqui deveria ser
uma regra que só faz sentido porque o LoCoMo tem essa forma.

## O achado central: existe um segundo mecanismo de corrupção, ortogonal ao da seção 3, e ele não foi tocado

A seção 3-6 acima conserta a MEMÓRIA (identidade, `_links`, contradição). Mas
há uma segunda peça, na RECUPERAÇÃO, que a correção de hoje não tocou: quando
nenhuma entidade da pergunta liga a uma proposição, `MecaRetriever.retrieve`
registra `abstain_reason` ("unknown_entity"/"unbound_question") **mas
continua emitindo contexto de qualquer jeito**, via fallback denso — top-k por
cosseno entre a pergunta e o `statement()` de qualquer proposição do grafo,
rotulado `"--- possibly related ---"` no contexto renderizado
(`render_context`, `faces.py:1088`). A decisão de abster fica inteiramente
para o LLM na resposta, olhando a regra 8 do `meca_answer.txt` ("quando a
memória não tem nada que responda, devolva Not mentioned"), sem que o rótulo
"possibly related" venha acompanhado de nenhuma instrução sobre o que ele
significa.

Isso é literalmente o mecanismo que a D35/D36 diagnosticaram no `atestado`:
contexto plausível para pergunta sem resposta é o que produz alucinação, e o
sinal estrutural (aqui, `abstain_reason`/`result.slot_support`) é calculado e
**nunca usado** para decidir nada — só é logado em `QAOutcome` para
diagnóstico pós-hoc. Confirmei isso lendo `Answerer.answer()`
(`faces.py:1170-1240`): o único curto-circuito real é
`if not result.facts: return ABSTAIN_ANSWER`, que quase nunca dispara, porque
o fallback denso praticamente sempre encontra alguma coisa num grafo com
centenas de proposições.

E o resultado congelado (pré-correção, seção 3) já mostra a assinatura: no
`metrics.json` atual, categoria adversarial, `f1 == abstention_rate ==
0,3161`, EXATAMENTE — a mesma identidade que a D35 achou em toda a linha L
(`adversarial/f1 == abstention_rate`). Isso não é coincidência de uma rodada
corrompida; é a assinatura de "a decisão de responder é decidida só pelo
gerador, sem portão estrutural", e essa causa continua de pé no código de
hoje.

## Por que não dá para presumir que a correção de identidade resolve isso de graça

Vale nomear com precisão por que o achado da D36 (rota estrutural refutada,
AUC 0,579) pode ou não se repetir aqui — são mecanismos parecidos mas não
idênticos, e a diferença importa:

A D36 testou um sinal fraco: co-ocorrência de SLOTS TIPADOS perto uns dos
outros (o corner test). MECA calcula um sinal mais forte em princípio: existe
uma PROPOSIÇÃO com este sujeito e este predicado? Isso é uma checagem de
existência bem mais específica que "os tipos de slot certos estão por perto".
Então há uma razão estrutural genuína para esperar que MECA separe melhor —
não é otimismo vazio.

Mas o motivo pelo qual a D36 falhou não era a fraqueza do sinal — era que uma
pergunta adversarial do LoCoMo é construída com o VOCABULÁRIO da própria
conversa: nomeia gente real, tópico real. Isso quer dizer que
`parse_question` vai achar a entidade (ela é conhecida!) na maioria dos casos
adversariais também — `result.slot_support` fica 1,0 (existem seeds) tanto
para a substantiva quanto para boa parte da adversarial, porque o teste hoje
só pergunta "a entidade é conhecida", não "existe proposição com este
PREDICADO para esta entidade". O ponto fino da adversarial não costuma estar
em nomear alguém desconhecido — está em perguntar uma relação que nunca foi
dita sobre alguém conhecido. `abstain_reason` como está hoje não distingue os
dois casos. Essa é a mesma lacuna, com uma cara ligeiramente diferente.

**Prognóstico, categoria a categoria, com o grau de confiança que cada um merece:**

- **single-hop e temporal**: devem melhorar de forma robusta em relação ao
  0,42/0,26 congelados. A causa da queda anterior (identidade fundida, spans
  contaminados) ataca justamente essas categorias mais diretamente — uma
  pergunta de um salto só precisa que o sujeito certo exista como nó e que o
  fato certo não tenha sido sepultado sob milhares de `_links`. Confiança
  alta de que sobe; nenhuma aposta segura sobre o número exato.
- **multi-hop**: deve melhorar, mas por um motivo estrutural genuíno e não só
  por limpeza — o `join` via `by_argument` pivota por QUALQUER valor de
  argumento, não só por entidade resolvida, então não depende tanto da
  precisão da resolução de identidade quanto as outras categorias dependem.
  Ainda assim, o travamento em f1~0,476 mesmo com `recall_context=1,0`
  (D36, achado 1 da reflexão de ingestão) era um problema de SÍNTESE na
  geração, não só de recuperação — isso é uma incógnita que a correção de
  hoje não visa e pode continuar limitando o teto.
- **open-domain**: o desenho já evita o erro de rotear pelo prompt errado — a
  MECA usa `retrieval.answer_prompt` só se configurado, e delibera não herdar
  o roteamento por categoria do LoCoMo (`faces.py`, comentário explícito:
  "that routing keys off this benchmark's question shapes, which is exactly
  the kind of thing a general method must not carry"). Isso é uma decisão de
  desenho boa e já tomada, vale reconhecer. Mas open-domain nunca passou de
  ~0,56-0,65 em nenhuma condição da linha L — não há razão estrutural nova
  aqui para esperar um salto.
- **adversarial**: esta é a categoria onde eu discordo de qualquer
  expectativa otimista sem ressalva. Pela leitura do código, o mecanismo que
  produziu `f1 == abstention_rate` na rodada congelada não foi tocado pela
  correção de hoje. Espero que o número mude (a rodada estava corrompida de
  um jeito que também distorcia adversarial), mas não tenho base para esperar
  que a IDENTIDADE `f1 == abstention_rate` desapareça, porque a causa dela —
  decisão de responder sem portão estrutural, delegada inteira ao LLM sobre
  um contexto que sempre existe — continua lá.

## O que fazer antes do servidor — só o que é genérico, nada afinado ao LoCoMo

Ordenado por custo, do mais barato ao mais caro:

1. **Meça antes de mexer, com o mesmo instrumento que o projeto já validou.**
   Não implementar nada de portão de abstenção ainda. Em vez disso, sobre os
   grafos que já existem (ou os que saírem do `git stash` da seção 9),
   computar, por pergunta, um sinal de existência mais fino que
   `abstain_reason` hoje dá: existe proposição com o SUJEITO **e** o
   PREDICADO da pergunta (não só o sujeito)? Correlacionar com
   adversarial/substantiva do jeito que `support.py`/`slots-oracle` já fazem
   (AUC, razão capturadas/deletadas, `net_questions`). Isso custa zero LLM e
   responde, antes de qualquer prompt ou código novo, se esse sinal separa
   melhor que o corner test antigo (AUC 0,579) ou se cai na mesma refutação.
   É a mesma disciplina que já matou a L3 e a rota estrutural de graça —
   reaplicá-la aqui é o oposto de afinar para o LoCoMo, é o método do projeto
   generalizando para uma unidade de memória nova.
2. **Rodar a suíte** (`pytest tests/test_meca.py`) antes de tudo — as duas
   quebras da seção 6 continuam de pé.
3. **Clarificar, no prompt de resposta, o que os rótulos já dizem.**
   `meca_answer.txt` não tem hoje nenhuma regra sobre a diferença entre
   `--- what the memory holds ---`/`--- linked through: ... ---` (achado
   estrutural) e `--- possibly related ---` (só similaridade, sem checagem de
   predicado). Uma frase a mais nas regras — algo como "conteúdo listado sob
   'possibly related' não foi confirmado como resposta a esta relação
   específica; não é motivo suficiente para responder sozinho" — é uma
   instrução sobre a ESTRUTURA que o retriever já produz, não sobre o LoCoMo.
   Custa reingestão? Não — é só o prompt de RESPOSTA (`meca_answer`), que roda
   por pergunta, não por passagem; não invalida o cache caro de extração/
   inferência/verificação.
4. **Os quatro health gates da seção 8**, para não descobrir problema depois
   de 5-6h rodando no servidor.
5. **Só depois disso**, se o passo 1 mostrar que o sinal separa bem, considerar
   um curto-circuito real (responder "Not mentioned" sem chamar o LLM quando
   o sinal indicar ausência com confiança) — e mesmo aí, testar primeiro em
   modo observação (logar o que TERIA sido decidido, sem agir) antes de
   deixar decidir de verdade. É o padrão que o próprio projeto já usa em
   outros lugares (`support.enabled=false` como default até o portão passar).

## Resposta direta às duas perguntas

**A proposta está sólida?** Como modelagem de conhecimento, sim — proposição
atestada com evidência obrigatória, dois relógios e modalidade é uma unidade
estritamente mais expressiva que ponteiro-para-texto, e a correção de
identidade desta sessão tira o motivo mais grave de desconfiar do substrato.
Como PIPELINE DE DECISÃO, não — falta o portão entre "a memória achou uma
proposição estrutural" e "o modelo respondeu porque algo parecido apareceu no
prompt", e esse é o mesmo buraco que já custou uma rodada inteira uma vez.

**O que esperar quando rodar?** Ganho real e provável em single-hop, temporal
e (com reserva) multi-hop; open-domain plano, sem razão nova para subir;
adversarial é o risco concreto — a expectativa mais honesta é que ele
reproduza `f1 ≈ abstention_rate` de novo, porque a causa disso não foi
tocada. Se isso acontecer, não é evidência de que a correção de identidade
falhou — é evidência de que ela resolveu um problema diferente do problema da
abstenção, exatamente como a D35 já separou esses dois eixos uma vez.

---

# Implementado nesta sessão (2026-08-28, depois do addendum acima)

A pedido do usuário: corrigir os 2 testes quebrados, os 4 health gates como
portão automático (não script avulso), e a frase no prompt de resposta sobre
"possibly related". Deliberadamente NÃO implementado: o oráculo de graça para
medir o sinal de abstenção (o usuário decidiu rodar valendo direto).

- **Testes**: `test_entity_resolution_collapses_short_forms` agora espera
  `{"melanie"}` (forma curta vence, como o código já fazia); o teste de
  contradição agora passa `functional_predicates={"works at"}` explicitamente,
  testando `link_propositions` isolado do estimador de funcionalidade
  derivado do corpus. Os dois foram verificados por execução direta (sem
  pytest — o device não tem espaço em disco para instalar; ver nota abaixo).
- **`meca_answer.txt` (version 4)**: regra 10 nova — explica que "what the
  memory holds"/"linked through" são achado estrutural e "possibly related" é
  só similaridade, não confirmado contra a relação perguntada; instrui a
  seguir a regra 8 (abster) quando só "possibly related" existir e não disser
  explicitamente o que foi perguntado. Só o prompt de RESPOSTA muda — não
  invalida o cache caro de extração/inferência/verificação.
- **`PropositionStore.stats()`**: dois campos novos, `max_entity_incidence` e
  `max_entity_incidence_ratio` (o degrau contra a mediana do próprio corpus,
  não um literal).
- **`consolidate.count_leaked_links(store)`**: nova função, conta
  proposições cujo `statement()` ainda carrega `_links`/`contradicts:`/
  `elaborates:`/`updates:` como texto, ou cuja qualifier tem prefixo `_`.
  Chamada em `MecaIngestor.ingest()` e guardada em
  `report.graph_stats["link_leaks"]`.
- **`Pipeline._meca_health_gates`** (`src/fgl/pipeline.py`), chamada de
  dentro de `_sanity()`: os quatro portões — razão de incidência de entidade
  > 20x a mediana; vínculos `contradicts` > proposições na mesma conversa;
  qualquer `link_leaks` > 0; taxa de `unbound_question`/`unknown_entity` >
  40%. Testado com dois cenários sintéticos (um saudável, sem warnings; um
  replicando os números reais do `conv-26.json` corrompido — os quatro
  dispararam, com as mensagens certas).

## O que eu NÃO consegui verificar
O device não tem `pytest` instalado nem espaço em disco para instalar (`pip
install pytest` falhou com "No space left on device", confirmando a nota já
registrada na memória do projeto). Validei cada peça isoladamente, chamando
as funções diretamente com dados sintéticos que reproduzem os casos que os
testes/gates deveriam pegar — mas a suíte completa (`pytest
tests/test_meca.py`, os 47+ testes) não rodou nesta sessão. Antes de confiar
cegamente nisto no servidor: `pytest tests/test_meca.py` lá, onde há espaço e
as dependências (a mesma rotina já usada antes, extraindo
`_transfer/fgl_src.tar.gz` num container com `pytest numpy pyyaml rich typer
spacy dateparser nltk`).

---

# Smoke test real no servidor (2026-08-28) — o erro de incidência e o que ele confirma

Primeira rodada de verdade pós-correção: `fgl ingest M1-meca-flat --force -n 1`
+ `fgl run M1-meca-flat -n 1` (conv-26, 199 perguntas). O `sanity` disparou:

> "uma entidade chegou a 522x a incidência mediana de alguma conversa —
> assinatura de colapso de identidade"

## Causa raiz: o bug era NO PORTÃO, não na correção

Investigando o grafo baixado (`artifacts/graphs/M1-meca-flat/conv-26.json`)
direto: a entidade de incidência 522 é `"caroline"`, e a de 364 é `"melanie"`
— os dois PARTICIPANTES da conversa, corretamente resolvidos como eles
mesmos. `"melanie's kids"` e `"melanie's family"` continuam entidades
DISTINTAS de `"melanie"` (6 e 6 incidências, não fundidas). Excluindo as
duas âncoras, a maior entidade não-participante tem incidência 7, mediana 1
— razão 7,0, longe do limiar de 20.

O gate 1 que escrevi na sessão anterior comparava a incidência máxima contra
a mediana **incluindo os participantes da conversa**. Mas um participante
DEVE dominar — é a pessoa sobre quem a conversa é. Comparei a coisa errada
contra a coisa errada: o sinal de colapso de identidade tem que olhar só
para quem NÃO é âncora, porque é aí que "Caroline's friends, family and
mentors' support" apareceria se a corrupção tivesse voltado.

## O que foi corrigido
`PropositionStore.stats()` ganhou `max_non_anchor_entity_incidence` /
`_ratio`, calculado excluindo `store.entity_anchors`. `Pipeline.
_meca_health_gates` (gate 1) passou a ler esse campo em vez do antigo
`max_entity_incidence_ratio` (que continua existindo, só informativo).
Verificado três vezes: (a) contra o dado REAL do conv-26 baixado do servidor
— zero avisos agora; (b) sintético, dois falantes dominando de propósito
(200/150 incidências) — zero avisos; (c) sintético, uma entidade
NÃO-âncora absorvendo 300 proposições — dispara, como deveria.

## O que este smoke test CONFIRMA sobre a correção de verdade
Isto é a primeira evidência real (não sintética) de que a correção da sessão
anterior funciona: no `conv-26.report.json` de uma ingestão fresca,
`link_leaks: 0`, `contradictions: 0`, Caroline e Melanie permanecem nós
distintos, e descrições possessivas não fundiram com elas. O F1 desta
conversa sozinha (199 perguntas): multi-hop 0,36, single-hop 0,41, temporal
0,30 — direção compatível com a melhora esperada (a comparação de verdade só
vale com as 1986 perguntas completas).

E, como esperado pelo addendum anterior: **`adversarial/f1 ==
abstention_rate` de novo, exatamente (0,3617 == 0,3617)**, já nesta amostra
de 47 perguntas adversariais. A previsão de que o mecanismo de abstenção sem
portão estrutural reproduziria essa identidade se confirmou na primeira
amostra real disponível — não é surpresa quando aparecer na rodada completa.

---

# Investigação do single-hop e a pergunta sobre entidades de alta frequência (2026-08-28)

Duas perguntas foram levantadas depois do smoke test: (1) uma entidade de
incidência muito alta (Caroline, 522; Melanie, 364) não pode **poluir a
busca** em vez de só inflar um gate de sanidade? A solução atual é a melhor
para isso? (2) o F1 de single-hop (0,41 na amostra de conv-26) está baixo —
as correções feitas vão subir isso?

Em vez de responder de memória, fui aos 199 exemplos reais de
`results/M1-meca-flat/predictions.jsonl`, filtrei `category_name ==
"single-hop"` (70 perguntas, 33 com F1 < 0,3) e li 12 casos de erro concretos
com pergunta/gabarito/predição/entidades-extraídas/`abstain_reason`.

## Achado novo: forma possessiva não resolve para a entidade

Um padrão dominava a amostra: perguntas como *"What are Caroline's plans for
the summer?"* chegavam ao `parse_question()` com `entities: []`. Causa raiz,
confirmada lendo o código, não suposta: `normalise()` preserva apóstrofos de
propósito (`_PUNCT = re.compile(r"[^\w\s'-]")`, para não destruir nomes como
"O'Brien"), então a tabela de entidades é indexada pela forma nua ("caroline"),
mas o token que a pergunta produz para "Caroline's" nunca vira "caroline" —
vira "caroline's". O `store.knows_entity()` nunca bate, a pergunta cai em
`unbound_question`, e o LLM nunca vê nenhum fato.

Quantifiquei antes de mexer: das 199 perguntas do smoke test, 16 tiveram
`abstain_reason == "unbound_question"`; 15 dessas 16 (94%) continham a forma
possessiva do nome de um participante registrado (`Caroline's`/`Melanie's`/
`Mel's`, testado por regex contra o texto da pergunta). Não é um caso de
canto — é a causa isolada mais comum de abstenção estrutural nesta amostra.

**Correção**: `parse_question()` em `src/fgl/retrieval/meca.py` ganhou
`_strip_possessive()` — quando a frase candidata não bate com nenhuma
entidade conhecida, tenta de novo com o apóstrofo final removido
(`"Caroline's"` → `"Caroline"`, cobre `'s`, `’s`, `'`, `’`), e só aceita o
fallback se a forma sem possessivo bater com uma entidade que a memória
realmente conhece — nunca inventa uma entidade nova. Verificado contra os 8
exemplos reais falhos da amostra: todos os 8 agora resolvem a entidade certa
(`Caroline`/`Melanie`), incluindo o caso mais díficil, gabinetes duplos como
*"What are Melanie's pets' names?"*. Adicionei também um teste de regressão
determinístico (`test_a_possessive_question_binds_the_known_entity` em
`tests/test_meca.py`) que fixa esse comportamento com uma entidade sintética,
sem depender do servidor.

Isto é um efeito **estrutural**, não um ajuste de threshold: antes da
correção, ~8% de TODAS as perguntas de uma conversa abstinham só por causa da
forma possessiva, independente do quão bem a extração e a consolidação
tivessem ido. Espero que suba single-hop (é onde a pergunta é "O que X faz/é",
a forma possessiva mais comum), mas também temporal e multi-hop na medida em
que dependem do primeiro hop resolver. Não vou prometer um número — a amostra
é uma conversa de 199 perguntas, não as 1986 do benchmark completo — mas a
direção é inequívoca e o mecanismo é auditável ponta a ponta.

## A pergunta sobre entidades de alta frequência: dois problemas diferentes debaixo do mesmo sintoma

A pergunta do Rodrigo mistura, com razão, dois riscos que parecem a mesma
coisa mas não são:

**Risco A — colapso de identidade** (o que already corrompeu a rodada
anterior): uma entidade de alta incidência não é ela mesma; é uma fusão
espúria de várias entidades diferentes (Caroline + Melanie + "Caroline's
friends, family and mentors' support"). Este está mitigado — resolução
lexical não-transitiva, `entity_anchors` sempre vencendo, e agora o gate 1
corrigido para separar participante-de-verdade de blob-espúrio. O smoke test
real confirmou: incidência 522 para Caroline é ela mesma, não uma fusão.

**Risco B — poluição de busca por grau alto (o "haystack problem")**: mesmo
uma entidade CORRETAMENTE resolvida, se tem 500 proposições, é um alvo de
busca ruim — `store.about("caroline")` ou `store.by_entity["caroline"]`
retorna 500 candidatos, e o que decide quais entram no contexto final é o
plano de recuperação (seed → join → scoring), não a identidade. Isto é
**real e NÃO está mitigado** hoje. Não é o mesmo bug que causou a corrupção
anterior — é um risco estrutural diferente, que só aparece DEPOIS que a
identidade está certa: quanto mais precisa a resolução de entidade, mais
degrau único e correto essa entidade acumula, mais grau ela tem, mais esse
risco importa.

Onde ele entraria, concretamente, olhando `meca.py`: o filtro de âncora do
plano de consulta usa `Target.predicate_key` para restringir dentro do
conjunto de proposições de uma entidade (`by_argument`/`about()` + o
`predicate` extraído da pergunta), então HOJE já não é "pega as 500 e reza" —
é "pega as 500 e filtra por predicado". Isso já reduz bastante o risco de
poluição pura por volume. O que ainda não existe é um ranqueamento
DENTRO do conjunto filtrado quando o predicado da pergunta é vago ou
combina com várias proposições da mesma entidade (ex.: "What did Caroline
say about her family?" pode bater com dezenas de proposições `Caroline
mentioned/said` sem um segundo critério de desempate além de recência/score
denso).

**A solução atual é a melhor?** Não, é a mais segura dado o que já foi
corrigido nesta sessão, mas não é a mais completa. Ela resolve o Risco A e
mitiga parcialmente o B (via o filtro por `predicate_key`), mas não ataca o
B diretamente. Duas alavancas concretas, corpus-derived (D30), que ficaram
de fora desta sessão por escolha do Rodrigo (queria rodar "valendo" antes):
(1) casar `predicate_key` por lema/sinônimo em vez de string exata, o que
reduziria falsos-negativos do filtro sem aumentar o volume não-filtrado;
(2) um segundo critério de desempate dentro do conjunto já filtrado —
recência (`asserted_at`) combinada com a similaridade densa da pergunta
inteira, com o corte derivado da distribuição de scores da própria
conversa, não um threshold fixo. Nenhuma das duas foi implementada — são
uma resposta honesta a "é o melhor jeito", não uma correção desta sessão.

---

# Diagnóstico da rodada completa (2026-08-28) — 1986 perguntas, 10 conversas

Primeira rodada de verdade nas 10 conversas (`conv-26/30/41/42/43/44/47/48/49/50`),
com todas as correções desta sessão dentro. Números:

| | |
|---|---|
| f1_micro | **0,3732** |
| f1_macro | 0,3011 |
| f1_substantivo (n=1540) | 0,3423 |
| abstenção geral | 0,1843 |
| recall_context (média ponderada) | 0,6129 |
| custo | 5,89M tokens, 2310s |

Por categoria: multi-hop 0,2637 (n=282) · temporal 0,2465 (n=321) · open-domain
**0,0802** (n=96) · single-hop 0,4352 (n=841) · adversarial 0,4798 (n=446).

`sanity`: `link_leaks: 0` e `contradictions: 0` nas 10 conversas, sem exceção
— **as correções de identidade seguram na escala cheia**, não só no smoke
test de uma conversa. Um aviso real do gate 1 (conv-44, entidade não-âncora
a 42x a mediana) — investigado abaixo, não é corrupção.

## O número que importa primeiro: isto é PIOR que a linha antiga (MEST/L)

Antes de procurar causa, a pergunta honesta: comparado a quê? A [[projeto-l-abstencao]]
tem os números da melhor rodada da linha L no MESMO benchmark: **L2-slots
0,542 micro, B1-full 0,546**, `recall_context` médio 0,770–0,776. M1/MECA
entrega **0,3732** e `recall_context` 0,6129 — cerca de **0,17 de F1 micro
abaixo do melhor resultado que este projeto já tinha**, e `recall_context`
no nível do L1 (0,614, o *primeiro* salto único da linha antiga, antes de
qualquer propagação) — não do L2 que ele suplantou.

Isto muda o enquadramento: não é "M1 está quase lá, falta polir". É "M1,
mesmo com identidade corrigida e válida na escala cheia, ainda recupera e
responde pior que o método que ele foi desenhado para substituir". Vale a
pena procurar causa estrutural, não só bugs pontuais — exatamente o que o
Rodrigo pediu.

## Achado 1 — o plano de consulta do MECA é mais raso que o da fatgraph antiga

`recall_context` por categoria: multi-hop 0,536 · temporal 0,801 ·
open-domain 0,549 · single-hop 0,752 · adversarial 0,277 (este último é
correto por desenho — adversarial não deveria ter contexto).

O plano de consulta (`meca.py::retrieve`) é: ancorar por entidade → **UM**
salto de junção limitado por orçamento (`join_steps`/`join_budget`) → fallback
denso se o resultado ficou fino. A linha antiga tinha propagação por σ
(vizinhança de vértice compartilhado), conexão Steiner multi-terminal e
cobertura por face — mecanismos deliberadamente removidos porque a proposta
nova troca "grafo de entidades genérico" por "loja de proposições
atestadas". Só que a troca também jogou fora a PROFUNDIDADE de busca, não só
a ambiguidade que ela causava. Multi-hop (0,536) e open-domain (0,549) são
exatamente as duas categorias que mais dependiam da propagação multi-salto
na linha antiga — e são as duas piores aqui.

**Isto não é motivo para voltar à fatgraph antiga.** É motivo para dar ao
MECA um segundo salto de junção, com orçamento derivado do corpus (não um
literal fixo) — a mesma disciplina D30 que já rege o resto do projeto.

## Achado 2 — granularidade de extração: o predicado costuma ser a frase toda, não uma relação curta

Nas 10 conversas: 12.267 proposições, **34,5% de predicados distintos sobre
o total** (4.237 predicados únicos), e **28,5% dos predicados aparecem
exatamente uma vez em todo o corpus**. Ao lado disso, os predicados mais
comuns são verbos genéricos demais para discriminar nada (`say` 1060,
`have` 578, `state` 377, `be` 290) — ou seja, o corpus é bimodal: verbo
genérico sem conteúdo, ou frase longa e única sem repetição. Exemplo real
(conv-44, entidade "audrey's dogs"):

```
Audrey's dogs | get excited | when Audrey brings out the ball or frisbee
Audrey's dogs | wear | something special for safety
Audrey's furry friends | use | Audrey's cozy and comfy item as a resting place
```

O prompt de extração já pede "canonical form: a base verb or verb phrase" —
a instrução existe, mas na prática o modelo often captura a oração quase
inteira como predicado. Isso quebra o `predicate_key` (correspondência de
string) usado tanto no escore de recuperação quanto no cálculo de
`predicate_functionality` na consolidação: duas proposições sobre o MESMO
fato, ditas com palavras diferentes, viram predicados diferentes e nunca se
encontram.

## Achado 3 — o objeto fica com uma referência não resolvida quando o valor concreto está em OUTRA passagem (achado novo, verificado no grafo real)

Amostrando 10 falhas de multi-hop com `n_facts` alto (30+, ou seja, a
recuperação claramente NÃO estava vazia), o padrão dominante era o mesmo:
a resposta prevista é uma paráfrase vaga do que está guardado, não o valor
concreto do gabarito.

| pergunta | gabarito | previsto |
|---|---|---|
| Onde Caroline se mudou há 4 anos? | Sweden | "her home country" |
| Qual a identidade de Caroline? | Transgender woman | "Caroline's gender identity" |
| O que os filhos de Melanie gostam? | dinosaurs, nature | (descrição de uma atividade de argila) |
| Que artistas Melanie viu? | Summer Sounds, Matt Patterson | "a show I went to (band)" |

Verifiquei a causa direto no grafo do conv-26 real, para o primeiro caso:

```
Caroline | be from | Sweden          <- passagem A
Caroline | move from | her home country   <- passagem B, sessão diferente
```

`meca_extract` roda **por passagem**, sem contexto das sessões anteriores.
A passagem B usa "her home country" (uma descrição definida) porque quem
falou já tinha dito "Sweden" numa sessão anterior — mas o extrator daquela
passagem não tem como saber disso, é um fato de OUTRA chamada de LLM. As
duas proposições ficam como nós desconectados, nunca fundidos (corretamente
— fundir por similaridade textual seria repetir o erro original). O escore
de recuperação favorece "move from" sobre "be from" porque a pergunta usa a
palavra "move" — ou seja, o mecanismo de pontuação prefere a proposição
vaga só porque ela ecoa a palavra da pergunta, e o respondedor cita essa
literalmente em vez de juntar as duas.

Isto é bem diferente do achado 1: não é falta de alcance da busca (ambas as
proposições relevantes ESTAVAM no contexto, `n_facts` alto o confirma), é
granularidade/correferência dentro do que foi extraído e como o
escore/resposta lida com duas proposições correferentes ditas de formas
diferentes.

## Achado 4 — fragmentação de referências coletivas/genéricas (não é corrupção, é perda de informação)

O único aviso real do health gate 1 na rodada cheia veio do conv-44:
entidade não-âncora "audrey's dogs" com incidência 42 (mediana da conversa:
1). Investigando: NÃO é fusão espúria (os cães nomeados — Toby, Scout,
Pixie, Pepper, Buddy — continuam nós distintos, corretamente). É o oposto:
seis frases coletivas diferentes referindo-se ao MESMO grupo real (os cães
da Audrey) viraram seis nós desconectados — `"audrey's dogs"`,
`"dogs"`, `"audrey's pups"`, `"audrey's pets"`, `"fur kids"`,
`"audrey's furry friends"` — nenhum ligado explicitamente aos cães
nomeados. Uma pergunta como "quais os nomes dos cães da Audrey" precisa
juntar Toby+Scout+Pixie+Pepper, e nenhuma dessas seis entidades coletivas
aponta para eles.

Isso é o risco B da conversa anterior (poluição por grau alto) manifestado
de um jeito mais específico e mais tratável do que eu tinha caracterizado:
não é "a entidade tem grau alto e a busca se perde nela" — é "a mesma
referência real vira várias entidades pequenas e nenhuma delas sozinha
carrega a resposta".

## Achado 5 — confirmado na escala cheia: a abstenção sem portão estrutural

`adversarial/f1 == adversarial/abstention_rate`, exatamente: **0,4798 ==
0,4798**, agora com n=446 reais, não uma amostra de 47. Essa é a mesma
identidade que [[projeto-l-abstencao]] documentou como valendo até **+0,17
de F1 micro** se resolvida — mais do que resolver multi-hop inteiro. Foi
deliberadamente deixada de fora nesta sessão, a pedido do Rodrigo. Com o
número confirmado na escala cheia, é o item de maior alavancagem disponível
no projeto inteiro.

## Achado 6 — open-domain está essencialmente quebrado, e por um motivo diferente dos outros

F1 = 0,0802 (n=96), abstenção 32%, mas `n_facts` alto nos exemplos
amostrados (35–40) — a recuperação não é o problema aqui. Perguntas típicas:
"Would Caroline still want to pursue counseling as a career if she hadn't
received support growing up?", "What would Caroline's political leaning
likely be?". Estas são perguntas de **inferência/julgamento**, não de busca
factual — o gabarito em si é uma extrapolação ("Likely no", "Liberal"), não
uma citação literal da conversa. A disciplina de abstenção do prompt de
resposta ("responda só o que está dito, abstenha caso contrário" — regra 8,
reforçada pela regra 10 que adicionei nesta sessão) está em tensão direta
com o que esta categoria pede. É plausível que a regra 10 desta sessão
tenha piorado especificamente esta categoria, mesmo tendo sido bem
justificada para o problema que ela mirava (distinguir "possibly related"
de confirmado).

## Achado secundário — `recall@5`/`recall@10` saem 0,0 em toda categoria

Provavelmente artefato de instrumentação (o mesmo tipo de artefato que
D31 já documentou para a linha antiga: k=10 sobre proposições finas trunca
antes de cobrir a evidência, "não perseguir"). Não afeta F1 nem
`recall_context`. Não investiguei a fundo por não valer o custo agora —
registrado para não ser confundido com um problema novo se reaparecer.

## Veredito: não é a hora de abandonar o atestado, é a hora de reforçar três camadas em cima dele

A pergunta do Rodrigo era direta: bug pontual ou proposta errada? A resposta,
com os dados na mesa: **o núcleo (proposição atestada, dois relógios,
modalidade, evidência obrigatória) não está refutado por nada disto** —
onde o plano alcança uma proposição concreta, ela é auditável e
majoritariamente correta (identidade limpa, zero vazamento, zero
contradição espúria, nas 10 conversas). A perda está concentrada em três
camadas construídas EM CIMA do núcleo, e nenhuma delas exige jogar fora o
atestado:

1. **Alcance da busca** (achado 1): um só salto de junção é raso demais
   para multi-hop e open-domain.
2. **Granularidade da extração e correferência entre passagens** (achados
   2 e 3): o predicado sai verboso e único demais, e o valor concreto de um
   slot pode ficar preso numa passagem diferente daquela que o pede de novo.
3. **Política de resposta** (achados 5 e 6): abstenção sem portão estrutural
   (adversarial) e abstenção rígida demais para perguntas inferenciais
   (open-domain) são o MESMO tipo de problema em direções opostas — a
   política de "quando responder" precisa de mais nuance do que uma regra
   de prompt só.

## Plano proposto, em ordem de alavancagem esperada / custo

1. **Portão de abstenção estrutural** (achado 5). Maior alavancagem
   conhecida do projeto (+0,17 micro no teto medido pela linha antiga, a
   confirmar na proporção certa aqui). `abstain_reason`/`slot_support` já
   existem, só não gateiam a chamada. Zero custo de LLM extra — é lógica
   antes da chamada.
2. **Segundo salto de junção com orçamento corpus-derived** (achado 1).
   Ex.: permitir join_steps=2 quando o primeiro salto deixa o resultado
   abaixo do quantil-mediano de `n_facts` da própria conversa — não um
   literal fixo. Mira multi-hop e open-domain diretamente.
3. **Disciplina de predicado mais apertada no prompt de extração + poucos
   exemplos negativos** (achado 2): mostrar 1-2 casos de extração ruim
   (predicado = oração inteira) corrigidos para forma curta, do jeito que
   `meca_verify` já rejeita veredictos malformados — o mesmo princípio
   aplicado à extração.
4. **Registro leve de valores já resolvidos, por conversa** (achado 3):
   um bloco curto e barato (não a conversa inteira) passado ao extrator com
   nomes/valores já resolvidos em passagens anteriores da MESMA conversa,
   para que "her home country" possa virar "Sweden" quando o nome já foi
   dito antes — sem re-processar tudo, sem fusão por embedding.
5. **Ligação conservadora de frases coletivas ao anchor + seus membros
   nomeados** (achado 4): usar o campo `links` (já existe, não funde
   identidade) para conectar "audrey's dogs"/"fur kids"/etc ao anchor e aos
   nomes individuais que aparecem sob o mesmo anchor — navegável em um
   salto, sem repetir o erro de fusão por similaridade.
6. **Segunda política de resposta para perguntas inferenciais** (achado 6):
   detectar a FORMA da pergunta (condicional/"would"/"likely" — o mesmo
   tipo de pista que `_NON_FACTUAL_CUE` já usa, não o nome da categoria do
   LoCoMo) e permitir uma resposta rotulada como inferência quando há base
   coerente, mantendo abstenção só para ausência real de base.

Nenhum item acima muda o esquema da proposição nem a disciplina de
threshold corpus-derived (D30). É reforço de camada, não mudança de
paradigma — mas os itens 2, 4 e 6 são mudanças de COMPORTAMENTO real do
método, não ajustes cosméticos, e merecem ser tratados como tal.

# Implementação dos 6 itens do diagnóstico (2026-08-28, mesma sessão)

Decisão do Rodrigo: todos os 6 achados devem ser tratados, e tudo antes de
rodar de novo no servidor. Esta seção documenta o que foi implementado, o
que foi conscientemente pulado, dois bugs pré-existentes encontrados no
caminho, e como cada mudança foi verificada (sem `pytest` disponível no
device por falta de espaço — verificação feita por execução direta das
funções reais contra dados sintéticos e contra o grafo real baixado do
servidor).

## Fix 0 — a causa raiz da pergunta sobre "audrey's dogs"

Antes dos 6 itens: o Rodrigo perguntou por que existem seis entidades
("audrey's dogs", "dogs", "audrey's pups", "audrey's pets", "fur kids",
"audrey's furry friends") sem nenhuma apontar para os nomes reais dos
cachorros, e se isso não prejudica a busca. Investigação no grafo real
(conv-26) confirmou: prejudica, e por três causas empilhadas, não uma:

1. **Extração** cria uma entidade nova por passagem para cada forma de
   referência coletiva — não há memória entre passagens dentro do próprio
   `meca_extract`.
2. **Consolidação** deliberadamente NÃO funde essas seis formas (fundir por
   similaridade textual é exatamente o erro de corrupção por identidade já
   corrigido nesta mesma linha de trabalho — D37).
3. **Um bug de deduplicação em `parse_question`** descartava o
   `entity_anchor` ("Audrey") da lista de entidades vinculadas sempre que a
   pergunta também continha uma dessas frases coletivas, porque a checagem
   de "já coberto" não distinguia anchor de não-anchor. Resultado: a
   pergunta amarrava só a entidade genérica de baixo grau, nunca a Audrey
   real com as 500+ proposições onde os nomes dos cachorros de fato
   aparecem.

Fix 0 corrige (3) — nunca descartar um `entity_anchor` por causa de uma
frase coletiva sobreposta. (1) e (2) são tratados pelos Itens 5 (link
conservador) e 2/6 (busca mais profunda e política de resposta) — não por
fusão.

Verificado por: `test_a_collective_phrase_never_displaces_its_anchor`
(sintético) + inspeção direta do grafo real de conv-26 antes/depois.

## Item 1 — portão de abstenção estrutural: investigado, premissa refutada, pulado por decisão do usuário

Prevendo maior alavancagem (era o item 1 do plano original), foi investigado
primeiro. Amostrei `abstain_reason`/`slot_support` nas 1986 perguntas reais:
o sinal dispara em **8/1986** (0,4%) no total e **1/446** nas adversariais —
exatamente a categoria em que um portão estrutural deveria ajudar mais.
Nos casos amostrados em que o sinal disparou, o fallback denso já dava a
resposta certa em 2 dos casos — um portão que abstivesse nesses pontos teria
custo líquido negativo mensurável (perde 2 respostas certas) por zero ganho
adversarial mensurável. A premissa herdada da linha antiga (MEST/L, que tem
um mecanismo de "atestado"/corner-test bem diferente e informado por gold)
não se sustenta para o MECA como está construído hoje.

Isto foi levado ao Rodrigo via pergunta explícita (não implementado
silenciosamente, não descartado silenciosamente); a decisão foi pular por
agora. **Fica documentado como problema aberto**, não como item concluído:
um portão de abstenção estrutural pode voltar a fazer sentido se a extração
ficar granular o suficiente para que `abstain_reason` dispare com uma
cobertura realista — o que os Itens 3/4 desta rodada podem mudar na próxima
medição.

## Item 2 — segundo salto de junção com orçamento corpus-derived

`join_steps: 1 -> 2` em `configs/conditions/M1_meca_flat.yaml`. A
implementação em si expôs dois bugs pré-existentes, nenhum introduzido
nesta sessão:

- **`join_steps` nunca foi de fato um contador de saltos.** O código lia
  apenas `if m.join_steps > 0`, então qualquer valor acima de 1 se
  comportava de forma idêntica a 1. Corrigido: o loop de junção em
  `MecaRetriever.retrieve()` agora itera de fato até `join_steps` saltos.
- **`_emit()` deixava um pool direto grande esgotar o orçamento inteiro**
  antes de join/dense terem qualquer chance, independente de quão bem esses
  grupos pontuassem — porque o consumo de orçamento era sequencial e não
  reservado. Este é o bug com mais impacto medido dos dois: reescrevi
  `_emit()` para reservar uma fatia igual de fatos/tokens entre os grupos
  não-vazios (`filtered`), com rollover (`carry_facts`/`carry_tokens`) do
  que um grupo menor não usa para os grupos seguintes.

Uma tentativa inicial de gatear a continuação do loop por
`len(seeds) + len(joined) < join_budget` foi implementada, testada e
**revertida**: como pools de seeds reais de âncoras de alto grau são sempre
muito maiores que `join_budget=8`, essa condição nunca era verdadeira — o
corpo do loop nunca executava, quebrando silenciosamente até o salto único
que já funcionava antes. Encontrado por verificação direta (fatos de junção
sumiam para perguntas reais que antes os tinham) e removido.

Verificado por: `test_a_large_direct_pool_never_starves_the_join_group`,
`test_join_reaches_second_hop`, `test_join_can_be_switched_off`,
`test_budget_truncation_respected`, e a paridade ribbon/flat (a lógica de
loop e de `_emit()` vive no `MecaRetriever` compartilhado, não em código
específico de reader — confirmado que a paridade byte-a-byte se mantém).

## Item 3 — disciplina de predicado no prompt de extração

`prompts/meca_extract.txt` v1 -> v2: campo `predicate` reescrito com
especificação mais apertada (1-3 palavras) e exemplos explícitos de
extração ruim (predicado = oração inteira) corrigidos para forma curta —
o mesmo princípio que `meca_verify` já usa para rejeitar veredictos
malformados, aplicado à extração. Nova regra 6 reforça isso mesmo para
frases longas.

Verificado por: revisão do prompt + `test_ingest_builds_a_proposition_memory_offline`
(exercita o `meca_extract` completo com o `FakeLLM` offline).

## Item 4 — registro leve de valores resolvidos, por conversa

Achado novo confirmado no grafo real (conv-*, achado 3 do diagnóstico):
`meca_extract` roda por passagem sem memória de passagens anteriores, então
uma descrição definida ("her home country") dita DEPOIS do nome real
("Sweden") ter sido dado numa passagem anterior nunca resolve.

Implementado: `comprehend.py` ganha um registro por conversa
(`registry: dict[str, dict[str, str]]`) atualizado depois de cada passagem
(`_update_registry`), passado para `meca_extract` como `{known_values}`
(regra 7 do prompt, v2). `_registrable()` decide o que entra no registro —
inicialmente aceitava qualquer valor, incluindo descrições genéricas
("her home country" também seria registrado, derrotando o próprio
propósito); corrigido adicionando o mesmo conjunto de determinantes
possessivos (`_VALUE_DETERMINERS`) já usado em `consolidate.py` para
excluir descrições e manter só nomes/valores concretos.

Verificado por: `test_the_cross_passage_registry_keeps_names_not_descriptions`
+ replay manual da passagem real de conv-26 onde "Sweden" é dito antes de
"her home country".

## Item 5 — ligação conservadora de frases coletivas ao anchor e seus membros nomeados

`consolidate.py` ganha `link_collective_references(store)`, chamado dentro
de `consolidate()` e guardado em `PropositionStore.entity_links` (novo
campo: `dict[str, list[str]]`, uma seta de mão única de uma frase coletiva
para as entidades individualmente nomeadas que ela agrupa — nunca fusão de
identidade, só um ponteiro navegável em um salto).

Duas versões foram tentadas e a primeira foi descartada por evidência
direta:

- **Versão 1 (co-ocorrência)**: um substantivo genérico e nu ("dogs",
  "fur kids") vira coletivo do anchor com quem mais co-ocorre. Testado
  contra dados reais de conv-26/conv-41/conv-30: marcou "acoustic guitar",
  o título de um livro e "pride parade" como coletivos de Caroline, porque
  numa conversa de 2 participantes quase tudo co-ocorre majoritariamente
  com um dos dois anchors. **Removido** — violava a disciplina de limiar
  corpus-derived (D30) na prática, não só na forma.
- **Versão final (só léxica)**: a chave da entidade É uma forma possessiva
  registrada de um anchor ("audrey's dogs", "audrey's furry friends") —
  mesma disciplina que D37 já aplica à identidade. Filtro de "membro"
  também corrigido: a primeira versão aceitava qualquer
  `object_is_entity=True` como membro, produzindo ruído ("sushi", "a farm",
  "landlords", "wine tasting" como "membros" do grupo de cachorros da
  Audrey). Corrigido exigindo que o texto bruto do objeto seja capitalizado,
  tenha no máximo 2 tokens e não comece com determinante/possessivo.

`build_graph`/`store_from_graph` propagam `entity_links` via
`collective_members` no meta do vértice. `MecaRetriever.retrieve()` usa
`entity_links` para dar um bônus de pontuação (+1,5 por acerto) a
proposições que citam um membro nomeado vinculado — sem alterar quem é
recuperado por identidade, só a prioridade de emissão.

Verificado por: `test_a_collective_phrase_points_forward_to_its_named_members`,
`test_named_members_outrank_generic_chatter` + replay da pergunta real
"Audrey's dogs' names" em conv-26 (antes: 40 fatos, zero nomes de cachorro;
depois: nomes presentes entre os fatos emitidos).

## Item 6 — segunda política de resposta para perguntas inferenciais

`prompts/meca_answer.txt` v4 -> v5: nova regra 11, detectada pela FORMA da
própria pergunta (não pela categoria do LoCoMo) — "would X...",
"is X likely...", "what would X's... be" — pedindo uma conclusão a partir
dos fatos atestados quando há base coerente, com resposta rotulada
começando por "Yes"/"No"/"Likely yes"/"Likely no" quando apropriado.
Conteúdo/espírito emprestado do `prompts/answer_open.txt` já existente
(hoje excluído do MECA por design) — não uma reescrita do zero. A regra 8
(abstenção por ausência real de base) continua valendo e não foi
enfraquecida: a mudança é permitir inferência QUANDO há base, não relaxar
quando não há.

Verificado por: revisão do prompt (mudança é só de prompt, sem novo código
Python) + confirmação de que a regra 8 permanece textualmente intacta no
v5.

## Verificação final consolidada

Toda a suíte `tests/test_meca.py` (52 funções `test_*`, incluindo as que
dependem de fixtures via `meca_built`/`cfg`/`embedder`/`prompts`/`llm`) foi
executada diretamente — sem `pytest` instalado (device sem espaço livre
para `pip install`) — usando um shim mínimo de `pytest.raises`/`mark.parametrize`/
`fixture` e replicando manualmente a cadeia de fixtures do `conftest.py`.
Resultado: **52/52 passam, 0 falhas.**

Uma checagem falhou numa primeira rodada de verificação ad-hoc
(`link_propositions` "contradiction detection") — diagnosticada e
descartada como falso alarme: o script de verificação usava duas datas
sem sobreposição de prefixo (`2023-01` vs `2023-06`), e
`TimePoint.overlaps` é containment de prefixo por design (não
"mesmo ano") — comportamento correto, não tocado nesta sessão. O teste
real (`test_a_contradiction_is_kept_and_flagged_not_resolved`, que usa a
mesma data `2023-03` para os dois lados) sempre passou, inclusive depois
de todas as mudanças desta sessão.

## O que fica pendente, deliberadamente

- **Item 1** (portão de abstenção estrutural) — não implementado, decisão
  explícita do Rodrigo, documentado acima como problema aberto a
  reconsiderar depois que a extração granular (Itens 3/4) rodar de novo.
- Nenhuma mudança de esquema na proposição, nenhuma fusão por embedding,
  nenhum limiar não-derivado do corpus foi introduzida em nenhum dos 6
  itens — disciplina D30/D37 mantida em toda a extensão desta rodada.
