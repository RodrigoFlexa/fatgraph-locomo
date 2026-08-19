# Premissas e condições de escopo do modelo L2

> Um método com condições de escopo **declaradas** é um método.
> Um método com condições de escopo **escondidas** é um método ajustado a um
> benchmark. Tecnicamente os dois podem ser idênticos — a diferença é
> inteiramente se as premissas estão escritas e se alguém consegue conferir.
>
> Este arquivo escreve. `fgl scope-check` confere.

---

## 0. Por que este documento existe

O L2 foi construído medindo erro por erro no LoCoMo. Isso produziu uma
arquitetura defensável — vértices tipados, episódio como unidade de índice,
cantos da rotação como objeto de consulta, ator como partição multiplicativa —
e também produziu um conjunto de números escolhidos *olhando para as respostas
anotadas*: `hub_degree: 60`, `concept_link_threshold: 0.75`,
`actor_prior_floor: 0.35`, a granularidade de tempo por mês e a lista
`QUESTION_NOUN_STOP`.

Nada disso é um hack. Não há exploração do avaliador, não há rótulo lido em
tempo de inferência, não há atalho que só funcione ali. Mas também não é
portátil: os números que fazem o método funcionar bem vieram da distribuição de
perguntas de um gerador específico, e um segundo corpus não herda nenhum deles.

O critério que separa os dois casos é estreito e mecânico:

> **O parâmetro precisa dos rótulos de ouro para ser fixado?**

Se precisa, é dívida de calibração. Se pode ser estimado do corpus não anotado
no momento da construção, é só um algoritmo com um estimador dentro.

Três coisas foram feitas a respeito, e este documento é a terceira:

| | o quê | onde |
|---|---|---|
| **1** | tornar cada constante um estimador medido no corpus | `fgl.memory.calibration`, condição `L2d` |
| **2** | medir quanto cada número importa, em vez de reportar só o ótimo | `fgl slots-sweep` |
| **3** | declarar as premissas e torná-las verificáveis | este arquivo + `fgl scope-check` |

---

## 1. As condições de escopo

Cada condição tem um identificador (`S1`…`S7`), um enunciado, o que é medido,
o critério, e — a parte que importa — **para o que o desenho degrada quando ela
falha**. Uma premissa sem caminho de degradação declarado não é uma premissa, é
um requisito escondido.

Duas classes, e a distinção carrega peso:

- **`runtime`** — computável a partir do que uma implantação realmente teria:
  as transcrições e, no máximo, o texto das perguntas. Pode ser rodada em dados
  não anotados, ou seja, pode ser rodada em produção e não só num paper.
- **`audit`** — precisa da evidência ou da resposta anotada. São exatamente as
  medições que *produziram* o desenho, e exatamente as que um corpus novo não
  vai conseguir rodar. Ficam aqui, e ficam rotuladas, porque fingir que o
  desenho não veio delas seria a versão desonesta deste arquivo.

---

### S1 — Diálogo entre poucos participantes nomeados `runtime`

**Enunciado.** O corpus é diálogo entre um conjunto pequeno e fixo de
participantes nomeados, de modo que "quem disse isto" é uma *partição* da
memória e não um atributo de texto livre.

**Medido.** Média de falantes distintos por conversa.
**Critério.** `<= 4` para o prior de ator ser uma partição forte.
**No LoCoMo.** 2, por construção do dataset.

**Degrada para.** Com `slots.calibration=derived` o prior se re-deriva sozinho
— `floor = 1/n_falantes`, `full = mediana da participação do falante dominante
no episódio` — em vez de precisar de nova varredura. Repare na direção: com 8
participantes o piso cai para 0.125, ou seja, o prior fica **mais** forte, o que
é correto, porque nomear um entre oito exclui muito mais do que nomear um entre
dois. Acima de ~8 participantes o que quebra não é o prior e sim o segmentador:
um episódio deixa de ser um par adjacente e vira uma fatia de reunião.

**O que isto *não* assume.** Não assume dois falantes. O `L1` assumia, ao
apagar o falante do grafo; o L2 não — o ator é um tipo de vértice como
qualquer outro, e o número de vértices desse tipo é livre.

---

### S2 — A pergunta nomeia exatamente um participante `runtime`

**Enunciado.** Uma pergunta identifica exatamente um participante, então o slot
`actor` da tupla de consulta está preenchido.

**Medido.** Fração de perguntas que casam com exatamente uma chave de ator.
**Critério.** `>= 0.80` para o prior valer a pena.
**No LoCoMo.** 98.5–99.7%.

**Degrada para.** O prior já é silencioso quando nenhum ator é ligado — ele é
uma multiplicação que simplesmente não acontece — então um corpus que falha
aqui *perde* o canal em vez de ser prejudicado por ele. O mesmo vale para o
teste de canto, que se abstém de se abster quando a pergunta não nomeia
ninguém.

**Este é o número mais específico do benchmark de todos.** 98.5% é uma
propriedade de como o LoCoMo gera perguntas (a partir de eventos relatados por
um dos dois participantes), não uma lei de diálogos. Uma reunião com cinco
pessoas, um chat de suporte em que a pergunta menciona o cliente mas a resposta
está na mensagem do agente, uma conversa em grupo sobre alguém ausente — em
todos esses S2 cai. O desenho sobrevive porque o canal é opcional por
construção; o *desempenho* não sobrevive, e é isso que S2 avisa antes de você
rodar qualquer coisa.

---

### S3 — A evidência é do participante nomeado `audit`

**Enunciado.** Quando a pergunta nomeia um participante, a evidência é um turno
desse participante.

**Medido.** Fração de perguntas de ator único cuja evidência anotada inclui um
turno daquele ator.
**Critério.** `>= 0.90` — é literalmente a estatística que o prior
multiplicativo codifica.
**No LoCoMo.** 96–100% (multi-hop 244/244, open-domain 72/72).

**Por que é `audit`.** Precisa da anotação de evidência. Um corpus novo não vai
ter. Esta é a dependência que o desenho tem do LoCoMo e que **não pode ser
removida por engenharia** — só declarada.

**Degrada para.** `actor_prior_floor` é o que preserva o resíduo: um episódio
para o qual a pessoa nomeada não contribuiu é rebaixado, nunca deletado. O
piso derivado (`1/n_falantes`) é a forma livre de corpus de decidir quanto
resíduo manter.

---

### S4 — Granularidade temporal das perguntas `runtime`

**Enunciado.** Referências de tempo nas perguntas caem numa granularidade que a
memória também indexa.

**Medido.** Granularidade mais fina nomeada, entre as perguntas que citam uma
data.
**Critério.** *Nenhum.* Isto **era** o parâmetro.

**O que mudou.** A versão original indexava um vértice por mês, e a justificativa
no código era "mês é a granularidade que as perguntas do LoCoMo realmente
usam". É uma observação verdadeira sobre um gerador de perguntas e uma péssima
razão para um parâmetro: um assistente de produtividade pergunta por dia, um
corpus jurídico por ano, e um grão por corpus teria que ser remedido toda vez —
a partir das perguntas, que é exatamente a dependência que o método não deveria
ter.

**O parâmetro foi removido, não reajustado.** Com
`slots.time_granularities=year,month,day` toda data é indexada em todos os
níveis que suporta, a pergunta emite todos os níveis que nomeia (mais fino
primeiro, os mais grossos atrás como *backoff*), e **quem escolhe o nível é o
amortecimento por grau que já existia**: um vértice de ano incide sobre quase
todo o corpus e `1/(1+log(grau))` o apaga; um vértice de dia incide sobre
punhado de episódios e pontua quase cheio.

Vale sublinhar o que isso significa para o desenho como um todo: a resolução
múltipla não precisou de nenhum peso novo nem de nenhuma regra nova. É o melhor
argumento disponível a favor de o amortecimento ter sido feito por grau em
primeiro lugar — ele já era um mecanismo de seleção de especificidade, e a
granularidade de tempo era um caso particular dele resolvido à mão.

**Custo.** No máximo três slots de tempo por data distinta em vez de um.

---

### S5 — O conjunto de perguntas vem de um template `runtime`

**Enunciado.** As perguntas são geradas por um template, então parte de seus
substantivos é *moldura* e não conteúdo — e as palavras de moldura são uma
propriedade do gerador, não da língua.

**Medido.** Frequência documental do substantivo mais comum das perguntas, e
quantos dos mais comuns a lista manual já continha.
**Critério.** `>= 0.05` significa que o conjunto é templatizado o bastante para
a filtragem do lado da pergunta estar fazendo trabalho real.
**No LoCoMo.** "conversation" em 194/1986 perguntas (9.8%), "date" em 118
(5.9%), "type" em 46, "answer" em 33.

**Degrada para.** `slots.question_stop=derived` estima o conjunto de moldura em
vez de nomeá-lo, contrastando duas distribuições que o sistema já tem:

```
framing(w)  ⟺  df_pergunta(w) >= min_df
           e   df_pergunta(w) / max(df_memória(w), piso) >= min_ratio
```

A razão é o que carrega o argumento. Uma palavra de tópico é comum nas
perguntas *porque* é comum nas conversas — "cachorro" é perguntado porque
alguém falou de cachorro, então as duas frequências andam juntas e a razão fica
perto de 1. Uma palavra de template é comum nas perguntas e quase ausente do
que alguém disse, porque vem do gerador e não do corpus. Isso é o análogo do
IDF do lado da pergunta, e vale para *qualquer* conjunto templatizado, não só
para este.

**A ressalva honesta, dita aqui e não enterrada.** O estimador é ajustado sobre
o *texto* das perguntas que serão respondidas. Isso é **transdutivo**. Não usa
rótulo nenhum — nem resposta, nem evidência, nem categoria — então não é
vazamento no sentido que importa para um número de recall, mas é uma
dependência de ver a distribuição de consultas de antemão, e uma implantação
que responde uma pergunta por vez não pode fazer isso. `question_stop=literal`
e `question_stop=none` são os dois fallbacks honestos para esse cenário, e
`fgl slots-sweep --knob slots.question_stop` precifica a diferença.

---

### S6 — O par adjacente é uma unidade real `runtime` (precisa do grafo)

**Enunciado.** A memória atômica é o par pergunta/resposta: uma réplica carrega
o valor e o turno acima carrega o tópico, então os dois têm que dividir uma
unidade de índice.

**Medido.** Fração de episódios com conteúdo de pelo menos dois falantes.
**Critério.** `>= 0.60`, senão o episódio é só um turno com enchimento.

**Degrada para.** Em corpora com forma de monólogo (documentos, notas de um só
autor) o episódio colapsa em direção ao turno e `sibling_frac` deixa de comprar
qualquer coisa — o modelo degrada para "L1 com slots tipados" em vez de
quebrar.

---

### S7 — A escala de grau do corpus `runtime` (precisa do grafo)

**Enunciado.** Um slot de grau alto não discrimina nada, então é tratado como
filtro em vez de enumerado — e "alto" tem que ser relativo a *este* corpus.

**Medido.** Fração de slots no ou acima do corte absoluto (`slots.hub_degree`),
por tipo.
**Critério.** `<= 0.10` em qualquer tipo, senão o número absoluto está comendo
o grafo.

**Por que isto é um bug e não só falta de elegância.** `hub_degree: 60` é uma
contagem absoluta. Num corpus dez vezes mais longo, *todo* slot cruza 60 e o
grafo inteiro vira hub — o mecanismo se desliga sozinho sem avisar. O comentário
que estava no config ("meça no histograma de grau do seu próprio grafo antes de
confiar no número") era um reconhecimento explícito disso; a correção é o
código medir em vez de o comentário pedir.

**Degrada para.** `slots.calibration=derived` troca a contagem absoluta por um
quantil da distribuição de grau **daquele tipo**, que é livre de escala por
construção. Por tipo porque os tipos têm escalas incomparáveis: um ator incide
sobre metade dos episódios e um conceito sobre três — um corte único não
consegue dizer isso.

---

## 2. Os pesos de canal *não* foram calibrados para longe

`dense_weight`, `actor_weight`, `predicate_weight`, `concept_weight`,
`type_weight`, `time_weight`, `sibling_frac`, `slot_damping` continuam
literais, e de propósito.

Eles são o único grupo de números que codifica uma afirmação **sobre o modelo**
e não sobre o corpus. "Um casamento de conceito diz mais do que um palpite de
hiperônimo" é uma ordenação que este desenho *afirma*, e afirmar isso é o que
uma ablação serve para testar. Fingir derivá-los seria vestir uma decisão de
projeto de medição.

O que eles ganham, em vez de um estimador, é uma curva: `fgl slots-sweep`
reporta para cada um a sensibilidade, a largura do platô e o `tuning_gain`.
Um peso com `tuning_gain ≈ 0` não foi realmente ajustado, seja qual for o
histórico de varreduras.

---

## 3. Como rodar

```bash
# as condições de escopo, só com o corpus (sem ingestão, sem LLM)
fgl scope-check -C L2

# incluindo S6 e S7 (constrói/reusa os grafos, ainda sem LLM)
fgl scope-check -C L2 --with-graphs

# quanto cada número importa, e a dívida de calibração estimada
fgl slots-sweep -C L2 --html artifacts/sweep_L2.html --out artifacts/sweep_L2.json

# um knob só, com valores próprios
fgl slots-sweep -C L2 -k slots.hub_degree --values 15,30,60,120,240

# a mesma arquitetura com todo estimador ligado, lado a lado com a versão
# varrida, no mesmo orçamento de tokens e sem nenhuma chamada de LLM
fgl slots-oracle -C L2 -C L2d
```

A última linha é a que fecha o argumento. `L2` são os números varridos, `L2d`
são os mesmos mecanismos com cada limiar estimado do corpus não anotado. **A
diferença entre os dois é o preço de não ter olhado para as respostas** — e ela
é medida, não discutida. Uma diferença pequena é a defesa mais forte que o
método tem; uma grande é um achado que vale reportar em vez de esconder, e o
`slots-sweep` diz em qual knob ela mora.

---

## 4. O que ainda falta (e não está sendo escondido)

- **Não há split.** Os knobs de `L2` foram varridos nas mesmas 10 conversas /
  1986 perguntas de que o número final saiu. Enquanto não houver
  *leave-one-conversation-out* — varrer em 9, avaliar na 10ª, repetir — a
  palavra correta continua sendo "varrido contra o oracle" e não "tunado no dev,
  avaliado no test". Este é o furo metodológico principal e nenhum dos itens
  acima o fecha.
- **Um único corpus.** Nada aqui prova portabilidade; prova apenas que o método
  *pode* rodar sem olhar para as respostas. A prova de portabilidade é rodar o
  config congelado num segundo benchmark (LongMemEval é o alvo óbvio: gerador
  de perguntas diferente, sessões com distratores, e casos sem resposta que
  exercitam o teste de canto) e reportar o número seja ele qual for.
- **`corner_actor_min`** continua absoluto. É um limiar sobre uma *proporção*,
  não sobre uma contagem, então não sofre o problema de escala do
  `hub_degree` — mas 0.5 ("a maioria") é uma escolha e não uma medição.
