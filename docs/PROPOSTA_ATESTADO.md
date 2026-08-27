# Proposta: MEST-A — memória episódica com **atestado de suporte**

> Estado: **implementada** (2026-08-27, D35). O escore, o atestado, o corte de Otsu
> e o objetivo de dois lados do `slots-oracle` estão no código, com 46 testes novos
> e a suíte inteira passando; `support.enabled` sai **False**, então L1–L6 continuam
> byte a byte iguais. Falta rodar o portão (§6.1) e, só se ele passar, os prompts
> por forma de suporte (§2.3).
>
> Os números citados vêm de `results/L2d-derived/metrics.json`,
> `results/L5-conjunction/metrics.json` e da rodada nova de F1 por categoria.

---

## 0. A frase da proposta

> A memória para de devolver **fatos** e passa a devolver um **atestado**: que
> *forma de suporte* a pergunta tem nesta memória, com testemunha. A geração é
> condicionada ao atestado, não a uma pilha plana de fatos.

Quatro formas de suporte, quatro políticas de orçamento, quatro prompts. A decisão
de responder deixa de ser um efeito colateral do LLM lendo 2000 tokens e passa a ser
uma saída estrutural do grafo, calibrada pelo corpus e auditável.

---

## 1. O número que reorganiza o projeto

Nas métricas em disco, para toda condição:

```
adversarial/f1  ==  adversarial/abstention_rate     (0.5762 == 0.5762 no L2d)
```

**Adversarial não é uma categoria de pergunta. É a medição direta da política de
abstenção**, e vale 446 de 1986 perguntas (22,5%) — mais que multi-hop (282) e
open-domain (96) somados.

### Decomposição da rodada nova contra a de 20/08 (mesma condição L2d)

| | 20/08 | rodada nova | delta |
|---|---|---|---|
| f1 substantivo (n=1540) | 0,5263 | **0,5347** | **+0,008** |
| abstenção em adversarial (n=446) | 0,5762 | **0,2420** | **−0,334** |
| **micro (n=1986)** | **0,5375** | **0,4690** | **−0,069** |

O suporte às perguntas **respondíveis melhorou**. Todo o prejuízo de micro é a
taxa de abstenção. O sistema ficou melhor no que sabe e muito pior em saber que
não sabe — e o segundo efeito é 8× maior que o primeiro.

### O teto que isso expõe

Com o resto congelado e uma porta de abstenção perfeita:

| cenário | micro | delta |
|---|---|---|
| rodada nova, como está | 0,469 | — |
| L2-slots (melhor atual da linha L) | 0,542 | — |
| B1-full-context (contexto inteiro, 23278 tokens) | 0,546 | — |
| **+ porta de abstenção perfeita** | **0,639** | **+0,170** |
| + recuperar as 83 abstenções erradas em substantivo | **0,655** | **+0,186** |
| porta realista (80% em adversarial, mesmo falso-positivo) | **0,594** | +0,125 |

Comparação de custo-benefício, na mesma unidade:

- resolver **multi-hop por completo** (0,430 → 1,0): +161 perguntas = **+0,081 micro**
- resolver **a abstenção** (0,242 → 1,0): +338 perguntas = **+0,170 micro**

**A decisão de responder vale mais do que resolver multi-hop inteiro**, e é o único
mecanismo do projeto que ninguém atacou de frente.

### Por que ninguém viu

`fgl slots-oracle` — o portão grátis, o instrumento em que toda a calibração da D30
foi decidida — otimiza `recall_context`. Mas em 22,5% do benchmark **recall é
anticorrelacionado com o comportamento correto**: recuperar contexto plausível para
uma pergunta sem resposta é exatamente o que produz alucinação. A calibração
derivada (concept_link 0,55 em vez de 0,75; stoplist de 8 palavras em vez de 31;
tempo em três resoluções) alarga a recuperação, ganha nas categorias respondíveis e
paga a conta em adversarial — e o oráculo não consegue ver a conta.

Isso é uma armadilha metodológica, não um bug. Corrigi-la é parte da contribuição.

---

## 2. O método

### 2.1 O atestado

A pergunta já chega como tupla de slots com um buraco: `(quem, fez-o-quê,
com-o-quê, quando)`. Antes de pontuar qualquer episódio, a memória classifica como
essa tupla se projeta no grafo:

| forma de suporte | condição estrutural | o que a geração recebe |
|---|---|---|
| **direto** | existe episódio cujos cantos cobrem os slots preenchidos e o buraco | a região concentrada, um prompt extrativo |
| **composto** | nenhum episódio cobre tudo, mas dois cobrem em conjunto por um conector resolvido | A · conector · B, **rotulados como junção**, com instrução de composição |
| **em conflito** | o buraco é preenchido de formas inconsistentes em tempos diferentes | ambos os estados em ordem temporal, prompt de atualização |
| **ausente** | os slots preenchidos não coocorrem em lugar nenhum acima do piso derivado | **quase nenhum contexto** + prompt de abstenção com a testemunha negativa |

O atestado é uma propriedade **topológica da projeção da pergunta no grafo de
slots**, computada antes de qualquer chamada de LLM. Não é um classificador de
template de pergunta — isso seria engenharia reversa do benchmark. É coocorrência
medida na própria memória, com limiar derivado da distribuição do corpus, exatamente
como `calibration.py` já faz para hub e concept_link.

### 2.2 Suporte como escore, não como teste binário

O `corner test` de hoje (`abstain_on_empty_corner`, desligado) é um único predicado
binário e mede 20/446 TP contra 38/1540 FP — quase break-even, por isso está
desligado. A proposta é substituí-lo por um **escore de suporte** contínuo, agregando
sinais que o grafo já tem:

- massa de coocorrência do par de slots mais específico da pergunta, amortecida por grau;
- fração dos slots nomeados pela pergunta que existem no grafo (um slot inexistente é
  o sinal mais forte de todos, e hoje é ignorado);
- concentração do escore: suporte real é concentrado, ruído é plano — usar a entropia
  da distribuição de escores, não só o topo;
- posse pelo ator (`corner_actor_min`), que já existe;
- e o negativo do canal de ponte: se nem uma ponte sintetizada liga os terminais,
  é evidência de ausência.

Corte no quantil da distribuição do escore **sobre o conjunto de perguntas**, sem
rótulo. Isso é transdutivo — mesmo status já declarado para `question_stop: derived`
em `ASSUMPTIONS.md`, com os mesmos fallbacks honestos.

O que se reporta não é um ponto: é a **curva de operação** (abstenção correta em
adversarial × abstenção errada em substantivo). Um sistema que sempre abstém tira 1,0
em adversarial e zero no resto; a curva é o que impede essa leitura preguiçosa.

### 2.3 Orçamento assimétrico

Hoje toda pergunta recebe os mesmos ~1990 tokens de fatos planos. Sob o atestado:

- **ausente** → contexto quase zero. 22,5% das perguntas ficam quase de graça.
- **composto** → orçamento partido em três: região A, conector, região B.
- **direto** → região concentrada.
- **conflito** → os dois estados, em ordem.

O prompt é escolhido pelo atestado. Quatro prompts, cada um dizendo qual operação de
composição executar. Hoje há um prompt só, com a instrução de abstenção sepultada na
regra 5 de seis regras (`prompts/answer.txt`) — e com `{speaker_a}`/`{speaker_b}` no
cabeçalho, que é estrutura do LoCoMo vazando para dentro do método, exatamente o que
foi corrigido no prompt da L6.

### 2.4 O que isso faz com o segundo gargalo

Com `recall_context = 1.0`, multi-hop trava em f1 ≈ 0,476 contra 0,657 de single-hop
no mesmo subconjunto. Toda a evidência está no contexto e a junção não acontece.
O atestado **composto** ataca isso diretamente: o contexto deixa de ser 58 fatos
planos e vira uma junção explícita com o conector nomeado. É a mesma evidência,
apresentada como a operação que ela pede.

### 2.5 A ponte, consertada

A L6 rodou: `V` foi de 21638 para **21640**. Duas pontes materializadas de 142
candidatos — o LLM rejeitou 98,6%. O culpado não é o sintetizador, é o **gerador de
candidatos**: similaridade de embedding entre episódios recupera semelhança de
**registro conversacional** (despedidas, agradecimentos — "Bye Nate!" a 0,99 de
cosseno), não de conteúdo. O dry run já dizia isso e foi lido como "esperado".

Sob o atestado, o gerador de candidatos passa a ser **dirigido pela estrutura de
slots**: pares de episódios cujas tuplas são **complementares** (um tem
`quem`+`fez-o-quê`, o outro tem `com-o-quê`+`quando` para predicado compatível) e que
não estão ligados a um salto. Uma despedida não tem estrutura complementar; ela some
do topo por construção, sem heurística de cortesia.

A ponte é então exatamente **um atestado composto pré-computado** para pares sem
forma de superfície em comum. Deixa de ser uma condição experimental e vira um
componente do método.

> **Portão numérico obrigatório para qualquer ponte**: multi-hop tem 282 perguntas,
> adversarial tem 446. Um mecanismo que ganha +0,05 em multi-hop (+14 perguntas) e
> perde 0,03 em adversarial (−13) é **líquido zero**. Ponte é afirmação que não está
> no corpus: ela é, por construção, um risco de suporte falso. Nunca avaliar as duas
> colunas separadamente.

---

## 3. O que fazer com o ribbon (e uma ablação que nunca rodou)

`G8-shuffled` é `retrieval.shuffle_context: true` — ele embaralha **os fatos
renderizados no prompt** do grafo G4. É uma ablação de *apresentação*. **A rotação do
grafo de slots tipados nunca foi ablacionada.** A leitura corrente ("a topologia não
carrega sinal") é verdadeira sobre o fatgraph de entidades e não foi testada sobre o
σ da MEST.

E sob o atestado o σ ganha um papel honesto, que não é "ordem carrega narrativa":

> σ é uma **esparsificação de tamanho linear e canônica do espaço de pares de slots**.
> Um episódio com `n` slots tem `n(n−1)/2` pares possíveis; a rotação retém `n` deles
> — os consecutivos — e `SLOT_ORDER` faz com que os retidos sejam justamente
> `(quem, fez-o-quê)`, `(fez-o-quê, com-o-quê)`, `(com-o-quê, quando)`. O teste de
> suporte é local e barato **porque** existe uma rotação.

Isso é uma afirmação de engenharia, verificável, e não depende de a ordem carregar
semântica. A ablação que a testa é barata e nunca foi feita: **permutar `SLOT_ORDER`
e medir a separação do escore de suporte.** Se a separação sobreviver à permutação, o
σ é notação e o texto deve dizer isso; se cair, o ribbon está vivo na forma honesta.

---

## 4. Reforma da avaliação (parte da contribuição, custo zero)

`fgl slots-oracle` passa a reportar um objetivo **de dois lados**, ambos sem LLM:

1. `recall_context` no subconjunto **substantivo** (como hoje);
2. **separação do escore de suporte** entre substantivo e adversarial — AUC e a curva
   de operação.

Um knob que sobe (1) e derruba (2) é uma troca, não uma melhoria. Esse portão teria
pego a regressão de 0,069 micro **antes** de gastar 6,4M tokens.

Complemento obrigatório, que continua faltando e nenhum item acima fecha:
**leave-one-conversation-out** e **um segundo corpus** (LongMemEval, com config
congelado). Sem isso não há afirmação defensável.

---

## 5. Como a contribuição é escrita

> Propomos **MEST-A**, uma memória episódica de slots tipados cujos limiares são
> estimados da distribuição do corpus não anotado e que, antes de gerar, emite um
> **atestado de suporte**: uma classificação estrutural — direto, composto, em
> conflito ou ausente — da forma como a pergunta se projeta na memória, com
> testemunha textual. A geração é condicionada ao atestado, o orçamento de contexto
> é assimétrico por forma de suporte, e a composição multi-hop implícita é
> materializada como junção explícita, pré-computada quando os episódios não
> compartilham forma de superfície. A abstenção deixa de ser uma instrução de prompt
> e passa a ser uma propriedade medida da memória.

Três afirmações verificáveis, nessa ordem de força:

1. **A decisão de responder é o maior bolso de F1 em memória de longo prazo**, e é
   estrutural: 22,5% do LoCoMo mede exatamente isso, e o teto medido é +0,17 micro —
   mais que resolver multi-hop por completo.
2. **Recall sozinho é o objetivo errado** para memória de longo prazo, e otimizá-lo
   degrada o sistema de forma que o instrumento não vê. Temos a regressão medida.
3. **Junções materializadas com proveniência** transformam composição implícita em
   evidência recuperável — condicionada ao portão 282 vs 446.

O que **não** se vende: ribbon como tese (até a ablação de `SLOT_ORDER` existir),
faces como narrativa emergente (F=97 para E=73789, uma face com 15824 meias-arestas),
e "melhor que o estado da arte" antes do segundo corpus.

---

## 6. Ordem de construção (uma coisa, não cinco)

Tudo acima é **um** mecanismo. A ordem só existe porque cada peça torna a seguinte
mensurável:

1. ~~Instrumentar o escore de suporte e reportar sua curva de operação no
   `slots-oracle`.~~ **FEITO (D35).** Zero LLM. Ver §6.1.
2. Ligar o atestado com os quatro prompts e o orçamento assimétrico. Uma rodada.
   **Só depois do portão.**
3. Trocar o gerador de candidatos de ponte por complementaridade de slots; a ponte
   vira atestado composto pré-computado.
4. Permutar `SLOT_ORDER` (a ablação que decide o que o artigo pode dizer sobre σ).
5. Leave-one-conversation-out e LongMemEval com config congelado.

### 6.1 O portão, e o que faz a proposta morrer

```bash
fgl slots-oracle -C L2d -C L5          # zero LLM; o atestado liga sozinho
```

A saída ganhou a seção `support attestation`. Três números decidem tudo:

| leitura | significado |
|---|---|
| **AUC ≈ 0,5** | o escore não separa respondível de não-respondível. A proposta morreu, de graça, e nenhum prompt a salva. |
| **AUC alta, `net Q` ≈ 0** | separa, mas o corte de Otsu cai no lugar errado — investigar o `best_achievable` da curva antes de qualquer outra coisa. |
| **`net Q` > 0 no corte derivado** | há chão. Aí sim vale o passo 2. |

`net Q` já faz a aritmética que as tabelas por categoria convidam a pular:
perguntas ganhas em adversarial menos perguntas destruídas em substantivo,
contra o F1 de referência da `results/L2d-derived`. Um mecanismo é julgado
nas duas colunas ou em nenhuma.

Para medir o efeito de um knob no lado que o oráculo antes não via:

```bash
fgl slots-oracle -C L2d --set slots.concept_link_min=0.75
```

Se `recall_context` subir e a AUC cair, é uma **troca**, não uma melhoria — e é
exatamente essa a troca que custou 0,069 de micro entre as duas rodadas de L2d.

## 7. Verificação pendente antes de confiar em qualquer coisa acima

- `tokens ingest = 0` para a L6 na tabela de custo, mas 2 vértices de ponte
  apareceram no grafo. Ou a etapa 2 rodou e não é instrumentada, ou não rodou. É
  preciso saber qual antes de dizer que "a L6 é no-op".
- As duas rodadas de L2d (0,5375 e 0,469) usam o mesmo nome de condição e o mesmo
  config. É preciso identificar o que mudou no código entre elas — se for uma
  regressão e não a calibração derivada, parte dos 0,069 volta de graça.
