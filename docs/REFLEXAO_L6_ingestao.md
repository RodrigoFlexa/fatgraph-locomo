# Reflexão — para onde mover a ingestão (não só a recuperação), pós-L5

2026-08-20. Gatilho: você pediu para eu refletir sobre o que mudar no método para
dar um salto, com a hipótese de que o gargalo agora está em como organizamos os
dados na ingestão, não em como recuperamos — especialmente para multi-hop.
Isto usa os `metrics.json`/`predictions.jsonl` de `results/L2d-derived` e
`results/L5-conjunction` que você acabou de puxar (rodada com LLM, 1986
perguntas, ainda não registrada em `docs/DECISIONS.md`) e uma leitura do código
de retrieval/render que ainda não tinha sido feita.

## 1. O que a rodada com LLM diz sobre L5 vs L2d

O oracle (D33) tinha deixado em aberto se a L5 (`L2d` + só o canal de conjunção
Steiner, sem o passeio) ganhava da L2d num corpus real com LLM. Agora dá para
responder:

| categoria | L2d f1 | L5 f1 | delta | L2d recall_ctx | L5 recall_ctx | delta recall |
|---|---|---|---|---|---|---|
| single-hop | 0.6046 | 0.6189 | **+0.0143** | 0.910 | 0.916 | +0.006 |
| multi-hop | 0.3777 | 0.3823 | +0.0046 | 0.641 | 0.638 | −0.003 |
| temporal | 0.5413 | 0.5329 | −0.0084 | 0.900 | 0.889 | −0.011 |
| open-domain | 0.2274 | 0.2217 | −0.0057 | 0.565 | 0.554 | −0.011 |
| adversarial | 0.5762 | 0.6166 | **+0.0404** | 0.865 | 0.854 | −0.011 |
| **overall (micro)** | **0.5375** | **0.5517** | **+0.0142** | — | — | — |

A L5 ganha no agregado, mas **não pelo motivo que a hipótese original testava**.
O ganho vem quase todo de abstenção correta em adversarial (+0.040, e ali f1
*é* a taxa de abstenção certa) e de um empurrão em single-hop. Em multi-hop —
a categoria que o canal de conjunção foi desenhado para resolver, e que no
oracle valia +0.014 — o ganho real com LLM é +0.0046, dentro do ruído, e o
`recall_context` de multi-hop não se moveu (0.641 → 0.638, na verdade caiu um
pouco). **O canal Steiner está funcionando como filtro de abstenção
(reduz falsos positivos em adversarial), não como o mecanismo de composição
multi-hop que a D33 registrou como "o único ganho da linha inteira".** Isso já
é uma correção a registrar: o ganho de oracle em multi-hop não sobreviveu à
rodada real na mesma proporção.

Vale rodar `fgl slots-oracle -C L2d -C L5` nas 1986 (não só nas 10) para
confirmar se o oracle de multi-hop também cai fora das 10 conversas de
calibração, ou se é especificamente algo que só aparece com LLM real — as
duas hipóteses pedem correções diferentes.

## 2. O achado novo: dois gargalos, não um

O veredito registrado em `projeto_l_grafos.md` — "ALCANCE ESTÁ SATURADO, o
gargalo é ORDENAÇÃO" — vem do `hop-profile`: 98,6% da evidência que o L2 erra
já está a um salto. Isso é uma afirmação sobre **alcançabilidade no grafo**.
Cruzando `recall_context` por pergunta com `f1` por pergunta nas predições reais,
aparece uma segunda coisa que o hop-profile não mede: **mesmo quando a evidência
está inteira no contexto entregue ao LLM, o f1 de multi-hop não converge para o
de single-hop.**

| categoria | n com recall_context=1 | f1 médio nesse subconjunto |
|---|---|---|
| single-hop | 761 | 0.657 |
| temporal | 279 | 0.577 |
| multi-hop | 108 | **0.476** |
| open-domain | 43 | 0.314 |

Isolado por completo do problema de recuperação — aqui recall já é 1.0, não há
mais o que buscar — multi-hop ainda perde 0.18 de f1 para single-hop. O mesmo
padrão aparece na L5 (0.474 vs 0.668). **Isto não é ordenação nem alcance: é o
que acontece DEPOIS que a evidência certa já está na janela, quando o LLM
precisa combinar duas ou três memórias em vez de extrair uma.**

Uma segunda medição explica por que isso é mais duro do que parece à primeira
vista — não é só "o LLM é pior em multi-hop", é que o **contexto entregue para
multi-hop é proporcionalmente mais ruidoso**, no mesmo orçamento de ~1994
tokens / ~58 fatos que toda pergunta recebe:

| categoria | evidência média (turnos) | fatos por turno de evidência |
|---|---|---|
| single-hop | 1.05 | 55.5 |
| temporal | 1.09 | 49.8 |
| open-domain | 1.23 | 49.2 |
| multi-hop | **2.50** | **23.4** |

Multi-hop precisa achar e ligar ~2,5 turnos de evidência espalhados, com menos
da metade da densidade de sinal por fato que as outras categorias têm para
achar 1. É uma tarefa estruturalmente mais difícil de fazer *dentro do mesmo
contexto*, não só de recuperar.

**Isto é exatamente a intuição que você trouxe, só que localizada com mais
precisão: o problema não é "não organizamos os dados", o pipeline de ingestão
já é bem trabalhado (resolução de entidade em cascata exata→embedding→LLM,
slots tipados, calibração derivada do corpus, dedup/consolidação). O que falta
é organização **na fronteira entre o que a ingestão sabe e o que o texto do
prompt mostra** — a composição multi-hop é calculada na hora da recuperação e
morre lá, sem virar texto legível para quem gera a resposta.**

## 3. Achado de código: o canal que funciona é mudo no prompt

`render_context` (`src/fgl/retrieval/faces.py:1075`) já faz exatamente o que
a D33 diz que falta: quando um grupo de fatos entra no contexto por um vértice
de ligação, o cabeçalho do grupo nomeia essa entidade — `"--- other memories
about {via_entity} ---"`, `"--- chain linking {via_entity} ---"`. O
docstring é explícito sobre a intenção: *"telling the model where two trails
meet is exactly the composition step a multi-hop question asks for, and it is
free"*.

Só que o canal de conjunção Steiner — o único canal que a D33 mediu como ganho
real em multi-hop — não usa esse mecanismo. Em
`src/fgl/retrieval/unified.py:172`:

```python
touch(ep_vid, st.weight * (best / total), "", via=terminals[0], label="steiner")
```

`kind=""` cai no default de `_SOURCE_BY_KIND` (`SOURCE_SLOT_DENSE`), que em
`_SOURCE_PRIORITY` tem a prioridade **mais baixa de todas** (0). Como a
imensa maioria dos episódios tocados pelo Steiner já foi tocada por algum
canal estrutural de prioridade maior (ator, tipo, predicado, conceito), o
rótulo do Steiner é sobrescrito quase sempre — e mesmo nos casos em que
sobrevivesse, `label="steiner"` é uma string literal, não o nome da entidade-
ponte resolvida (deveria ser algo como `self.graph.vertices[terminals[0]].name`).
Ou seja: o sinal "estes dois trechos se conectam por causa de X" existe dentro
do retriever — é literalmente o que o Steiner calcula — mas nunca chega a virar
o cabeçalho `"--- chain linking X ---"` que o próprio código já sabe fazer para
outros canais. **A composição multi-hop é usada para PONTUAR quais episódios
entram no orçamento, mas não é usada para EXPLICAR ao gerador por que eles
estão juntos.**

Isso explica, ao menos em parte, por que o ganho de oracle em multi-hop não
sobreviveu à rodada com LLM (seção 1): o oracle mede se a evidência certa está
no contexto, não se o LLM consegue montar a resposta com o cabeçalho que
recebeu — e o cabeçalho que ele recebeu, na prática, não diz nada sobre a
ligação.

## 4. Propostas incrementais, da mais barata para a mais cara

**4a. Consertar o rótulo do Steiner (custo: minutos, zero re-ingest).**
Passar `kind` real ou dar ao Steiner sua própria prioridade/fonte (ex.:
`SOURCE_STEINER`, prioridade acima de `SOURCE_SLOT_CONCEPT`) e `label` como o
nome resolvido de `terminals[0]` (ou, melhor, dos DOIS terminais quando o
grupo cobre mais de um). Cabeçalho vira algo como `"--- chain linking
Caroline's therapy and the support group ---"`. É a correção mais barata
possível e testa diretamente a hipótese da seção 3: se isolar e nomear a
ligação no texto ajuda a síntese, mesmo sem mudar uma única linha de
recuperação.

**4b. Experimento "de graça": reordenar sem re-recuperar.**
Antes de gastar ingest ou LLM novo: pegue as 108 perguntas multi-hop do L2d/L5
que já têm `recall_context=1.0` (a evidência já está lá) e teste variações de
apresentação — mover os turnos de evidência para o topo do trail, agrupar os
dois trails de evidência adjacentes um ao outro, inserir o cabeçalho da 4a —
sobre o MESMO conjunto de fatos, sem tocar o retriever. Isso isola
apresentação de recuperação por completo, do jeito que `shuffle_seed` em
`render_context` já isola ordem de conteúdo (a mesma ideia que testou se sigma
carrega sinal, aplicada agora ao lado da geração). Custo: reprocessar 108
prompts, nenhum re-ingest.

**4c. Um prompt de resposta específico para multi-hop.**
Vocês já têm o precedente: `answer_open.txt` e `answer_set.txt` existem
porque perguntas de categorias diferentes pedem instruções diferentes. Uma
`answer_multihop.txt` poderia pedir explicitamente, antes da resposta curta,
que o modelo identifique qual par de memórias se conecta e por qual entidade
— não como chain-of-thought solto, mas como um passo estruturado (“BRIDGE:
<entidade> / <memória 1> + <memória 2>”) seguido da resposta extrativa de
sempre. Testável em paralelo à 4a/4b porque ataca o mesmo sintoma (síntese,
não recuperação) por outro ângulo — vale rodar as duas junto e ver se
empilham.

**4d. Reduzir a densidade de ruído no orçamento multi-hop.**
A tabela da seção 2 mostra 23 fatos por turno de evidência em multi-hop contra
~50-55 nas outras categorias, no mesmo teto de tokens. Uma condição que, só
para perguntas detectadas como multi-hop (o parser de pergunta já teria como
inferir isso — mais de um slot específico ligado a atores/conceitos
diferentes), reduza `max_facts_in_prompt` e reaplique o orçamento economizado
como raio maior ao redor dos terminais Steiner, testa diretamente se é
densidade de ruído ou capacidade do modelo que domina o teto de 0.476.

## 5. Mudança de paradigma: materializar a composição na ingestão, não só pontuá-la na recuperação

Todas as propostas da seção 4 mexem em como o texto já recuperado é
apresentado. A mudança de paradigma é sobre o que a ingestão produz como
unidade recuperável.

Hoje uma "unidade" é um fato atômico (uma relação entre duas entidades) ou um
episódio (par de turnos). Multi-hop precisa de DUAS unidades que a recuperação
junta na hora — o "join" é uma operação de leitura (Steiner), nunca vira dado.
A ideia central do L2/L2d — "unidade de índice != unidade de emissão, o
episódio liga, o turno paga" — já reconheceu que a unidade certa para
recuperar não é sempre a unidade certa para o índice. A extensão natural:
**para multi-hop, a unidade certa para a EMISSÃO pode não ser dois fatos
atômicos lado a lado — pode ser um terceiro fato, sintético, que a ingestão já
escreve por extenso.**

Concretamente: quando o ingest liga dois episódios por um vértice de slot
não-hub (o mesmo teste que o Steiner já faz em tempo de leitura — 84.365 pares
de episódios compartilhando 2+ slots não-hub, já contado em `projeto_l_grafos.md`),
materializar ali, offline, uma frase-ponte curta e determinística (sem LLM,
por template, do mesmo jeito que o resto do L2d evita gabarito): "{entidade A}
liga o episódio '{resumo A}' ao episódio '{resumo B}' via {vértice
compartilhado}". Essa frase vira um fato de primeira classe no grafo — tem seu
próprio vértice, pode ser rankeado, pode entrar no orçamento como qualquer
outro — só que em vez de pedir ao LLM para inferir a ligação a partir de dois
fatos dispersos dentro de um contexto com 23 fatos de ruído por evidência, a
ligação já está escrita.

Isso é uma mudança de banco de dados: sair de "join calculado a cada consulta"
para "view materializada para os joins que o corpus efetivamente sustenta" —
só os pares que já compartilham um vértice específico não-hub, o que mantém o
custo limitado (não é todo par de episódios, é a mesma vizinhança que o
Steiner já teria escaneado). O ingest continua zero-LLM. O ganho esperado é
justamente no ponto que a seção 2 isolou: reduzir o trabalho de síntese do
gerador a reconhecer uma frase já pronta em vez de montar a ligação sozinho
dentro de um contexto ruidoso.

Risco simétrico ao que a linha L já mediu duas vezes (L3 do passeio, Steiner
em adversarial): fatos sintéticos de ligação podem, em perguntas adversariais,
sugerir uma conexão que não deveria existir — o mesmo mecanismo que ajuda
multi-hop pode alimentar falsos positivos exatamente onde a L5 acabou de
ganhar (seção 1). Testar com o mesmo protocolo de sempre: oracle nas 10
antes de LLM, hop-profile-style antes de shipar, e olhar adversarial e
multi-hop juntos, nunca um sem o outro.

## 6. Uma segunda direção de paradigma, mais especulativa

Se a materialização de pontes (seção 5) funcionar, o próximo passo natural —
não recomendado ainda, registrado para não perder — é mover parte da extração
de fatos (`extract_facts_topical.txt`) de "um fato = uma relação entre duas
entidades dentro de UMA sessão" para reconhecer, no momento da extração, que
um fato às vezes já é a continuação de outro fato de uma sessão anterior (ex.:
"Caroline foi ao grupo de apoio LGBTQ" numa sessão e "o grupo de apoio ajudou
Caroline a aceitar sua identidade" três sessões depois são o mesmo fio
narrativo). Isso é essencially o que GraphRAG-style community/path
summarization faz — sumarizar comunidades de fatos relacionados em vez de só
extrair pares — mas custa LLM adicional no ingest, o que quebra a propriedade
"zero-LLM" que faz o L2d ser barato e recalibrável. Vale considerar só depois
de a versão determinística (seção 5) mostrar que ligações explícitas ajudam
o suficiente para justificar o custo de uma versão mais rica.

## 7. Ordem recomendada

1. 4a (rótulo do Steiner) — minutos, sem re-ingest, testa a hipótese central
   da seção 3 diretamente.
2. 4b (reordenar sem re-recuperar, nas 108 perguntas com recall=1) — isola
   apresentação de recuperação, reusa infraestrutura de `shuffle_seed` já
   existente.
3. Rodar `fgl slots-oracle -C L2d -C L5` nas 1986 perguntas (não só nas 10)
   para saber se o gap de multi-hop entre oracle e LLM real (seção 1) é do
   corpus de calibração ou é genuíno.
4. 4c (prompt `answer_multihop.txt`) em paralelo a 4a, mesma leva de testes.
5. 4d (orçamento adaptativo por categoria detectada) só se 4a-4c não fecharem
   o gap de 0.18 sozinhos.
6. Seção 5 (pontes materializadas na ingestão) — a mudança de paradigma —
   só depois de 1-5 esgotados, porque se 4a sozinho já mover multi-hop, o
   custo de mudar a ingestão pode não se pagar.

Why: os dois achados novos desta reflexão (o gap recall=1 vs f1, e o Steiner
mudo no render_context) apontam para o mesmo lugar — o gargalo atual não é
"achar" nem "ordenar", é "explicar a ligação depois de achada". Isso muda a
recomendação: o próximo salto mais provável não é mexer em mais uma camada de
retrieval, é fazer a informação que a ingestão/retrieval JÁ TEM sobre como os
fatos se conectam realmente chegar ao texto que o gerador lê.
