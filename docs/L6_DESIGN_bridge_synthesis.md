# L6 — pontes sintetizadas na ingestão com LLM

2026-08-20. Continuação de `docs/REFLEXAO_L6_ingestao.md`, seção 6. Você pediu
para lapidar a ideia especulativa de usar LLM no ingest para conectar pares de
episódios, e levantou a restrição certa: nada aqui pode ser premeditado em
cima do que o LoCoMo especificamente contém — o desenho tem que valer para uma
conversa que ninguém leu ainda. Este documento assume essa restrição como
requisito de design, não como ressalva no fim.

## O que muda de verdade (não é só "chamar LLM mais uma vez")

Vale separar duas coisas que a reflexão anterior deixou próximas demais:

1. **O rótulo mudo do Steiner** (achado 2 do documento anterior) é sobre
   apresentação: a ligação já existe no grafo (dois episódios compartilham um
   slot não-hub), só não vira texto legível.
2. **O que a L6 ataca é outro problema, mais fundo**: o Steiner só consegue
   pontuar uma ligação que JÁ EXISTE no vocabulário tipado — dois episódios
   têm que compartilhar literalmente um ator, predicado, conceito, tipo ou
   tempo. Quando a ligação entre dois episódios é temática ou causal mas não
   passa por nenhuma entidade que os dois textos nomeiam do mesmo jeito
   ("Caroline foi a um grupo de apoio LGBTQ" numa sessão / "Caroline contou
   aos pais" três sessões depois — o mesmo fio narrativo, zero entidade
   literal em comum), **nenhum canal atual consegue nem propor essa aresta**,
   porque nenhum deles lê os dois textos ao mesmo tempo. Steiner, geodésica,
   sigma — todos operam sobre arestas que a extração determinística já
   colocou no grafo. Isso não é a mesma coisa que "alcance saturado": o
   hop-profile mede alcance DENTRO do vocabulário atual; a L6 propõe um tipo
   de aresta que esse vocabulário não consegue representar por construção. É
   por isso que a regra "aumentar alcance não move nada" (D32/D33, provada
   para o passeio) não se aplica automaticamente aqui — mas também não deve
   ser assumida sem teste, e o plano de validação abaixo é desenhado
   justamente para não confiar nisso de graça.

A L6, então, não é "L5 com mais uma passada de LLM" — é a primeira condição
da linha em que a INGESTÃO decide algo que a extração de fatos, sozinha,
estruturalmente não pode decidir: se dois trechos que não compartilham
vocabulário se referem à mesma coisa.

## Desenho em duas etapas: candidatos baratos, LLM só no gargalo

O erro fácil de cometer aqui é perguntar ao LLM "esses dois episódios se
conectam?" para todo par de episódios de uma conversa — O(n²) chamadas, e a
maioria das respostas seria não. O desenho tem que filtrar antes de gastar.

**Etapa 1 — geração de candidatos, zero LLM, reaproveita infraestrutura que já
existe.** O ingest já constrói um índice de embeddings por episódio (é o que
o canal denso usa em `retrieve()`). Para cada episódio, pegar os k vizinhos
mais próximos por similaridade densa que:
(a) NÃO compartilham já nenhum slot não-hub — isso já seria coberto por sigma
e por Steiner, gastar LLM aí é redundante; usar `fgl.evaluation.hops`
(já existe) para confirmar que os dois episódios estão a distância > 1 no
grafo atual antes de considerar o par;
(b) têm similaridade acima de um limiar DERIVADO do próprio corpus daquela
conversa — mesma receita de `concept_link_threshold` em `calibration.py`
(quantil da distribuição de similaridade episódio-episódio observada, não um
número fixo). Isso é o que torna a etapa 1 honesta em relação à restrição que
você levantou: o limiar se ajusta sozinho ao que aquela conversa realmente
contém, não a um número calibrado olhando para o LoCoMo.

Isso troca O(n²) por O(n·k), com k pequeno (5-10). Antes de gastar um
único token de LLM, rodar essa etapa sozinha e CONTAR quantos pares
sobrevivem — o mesmo hábito de "custo medido" que já aparece em D31 (smoke
test com contagem antes de comprometer o orçamento).

**Etapa 2 — julgamento e síntese, um LLM call por candidato sobrevivente.**
O prompt recebe SÓ o texto dos dois episódios candidatos (ou, mais barato e
mais consistente com o resto do pipeline, os `fact_text` já extraídos de cada
um, que já resolveram pronomes) — nunca a pergunta, nunca o gabarito, nunca
outros episódios. Rascunho:

```
# TASK: bridge_synthesis

You are looking at two exchanges from the same long-running conversation
between {speaker_a} and {speaker_b}, taken from different sessions. They were
flagged as topically close, but do not share a named entity in your memory of
this conversation.

EXCHANGE A ({date_a}):
{facts_a}

EXCHANGE B ({date_b}):
{facts_b}

Is there a SPECIFIC, CONCRETE connection between these two exchanges -- the
same underlying person, place, event, plan or feeling continued, referenced,
caused, or contradicted later -- as opposed to just a similar topic in
general? Do not invent a connection that requires information outside what is
shown above.

If NO: respond {{"linked": false}}.
If YES: respond with one SELF-CONTAINED sentence, naming people by their
proper names, stating the connection explicitly, in the same style as the
memories above. Do not restate either exchange in full -- name only what
connects them.

{{"linked": true, "bridge_text": "...", "entity_1": "...", "entity_2": "..."}}
```

A resposta, quando `linked=true`, entra no pipeline exatamente como qualquer
outro fato de `extract_facts_topical.txt` -- passa pelo MESMO
`EntityResolver` (cascata exata→embedding→LLM que já existe em
`entities.py`), então `entity_1`/`entity_2` se ligam a vértices já existentes
em vez de criar duplicatas, e o fato ganha `turn_ids` das duas sessões de
origem. A única coisa nova de fato é a MARCA de proveniência: o fato carrega
`source="ingest_bridge"` (do mesmo jeito que `calibration.py` marca cada
número com `source: derived/fallback`), para que ele possa ser medido,
auditado e desligado por completo sem tocar em mais nada -- o mesmo princípio
de "fallback nunca silencioso" de D30, aplicado a um fato inteiro em vez de a
um limiar.

**Por que isso resolve "como ajuda a conectar pares" de um jeito que o Steiner
sozinho não resolve:** o fato-ponte é uma ARESTA NOVA no grafo, não uma
pontuação melhor sobre arestas antigas. Uma vez que ela existe:
- o canal denso pode achar essa frase diretamente, porque ela foi escrita para
  parecer com o tipo de coisa que uma pergunta multi-hop pergunta;
- o Steiner (ou até um lookup de 1 salto comum) agora tem uma aresta real para
  atravessar entre dois episódios que antes eram ilhas uma da outra no
  vocabulário tipado;
- quando ela entra no contexto, `render_context` (uma vez com o rótulo do
  Steiner consertado, achado 2 do documento anterior) pode dar a ela o
  cabeçalho "--- chain linking X and Y ---" de verdade, porque agora HÁ uma
  entidade nomeada para ancorar o cabeçalho;
- e o mais importante para o gargalo medido na seção 2 do documento anterior:
  o gerador não precisa mais montar a composição sozinho dentro de um
  contexto com 23 fatos de ruído por evidência -- a composição já está escrita
  como UMA frase, do mesmo jeito que uma pergunta single-hop é respondida
  extraindo de uma frase só.

## Por que isso não é premeditação, e onde o risco de premeditação realmente mora

Três decisões de desenho existem especificamente para isso:
1. O limiar de similaridade da etapa 1 é derivado por conversa, não fixo --
   uma conversa nova, de outro domínio, gera seu próprio limiar.
2. O prompt da etapa 2 nunca vê a pergunta nem o gabarito -- só os dois
   episódios. Não há como ele "saber" que uma ponte importa para alguma
   pergunta específica do LoCoMo.
3. A geração de candidatos é auditável e reproduzível sem re-rodar LLM: dado o
   índice de embeddings (que já existe), a lista de pares candidatos é
   determinística.

O risco de premeditação que ISSO NÃO cobre, e que vale registrar honestamente:
o modelo (gpt-5-mini) pode ter visto o LoCoMo no próprio treinamento e
"reconhecer" o benchmark, escrevendo pontes que parecem gerais mas na
prática só funcionam bem porque o modelo já viu esse dataset. É um tipo de
vazamento diferente do que vocês já discutiram em `projeto_l_calibracao.md`
(rótulo de ouro), mais parecido com contaminação de treinamento -- e não dá
para eliminar de dentro do experimento. A mitigação real é a que já está
registrada como o item 2 pendente em `projeto_l_calibracao.md`: rodar a
mesma L6, config congelada, num corpus diferente (LongMemEval) e ver se a
taxa de pontes úteis (abaixo) se mantém. Vale adiantar isso especificamente
para a L6, mais do que para as condições anteriores, porque é a primeira vez
que o julgamento do LLM entra na ingestão em vez de só na extração de fatos
atômicos.

## Como testar sem comprometer o orçamento de LLM (a mesma disciplina de sempre, estendida)

A ordem que vocês já seguem -- hop-profile antes de qualquer passeio, oracle
antes de qualquer rodada com LLM -- se estende naturalmente aqui, com um passo
novo no meio porque, ao contrário de L1-L5, a L6 muda a ingestão e não pode
emprestar os grafos da L2d:

1. **Contagem de candidatos (zero LLM).** Rodar só a etapa 1 nas 10 conversas
   de calibração. Se o número de pares sobreviventes for enorme (o
   equivalente aos 84.365 pares que compartilham slot, citado na reflexão
   anterior), o limiar derivado está solto demais e precisa de um quantil
   mais alto antes de gastar um único token.
2. **Amostra pequena de síntese (LLM, mas barato).** Rodar a etapa 2 numa
   fração dos candidatos de 1-2 conversas. Medir a taxa `linked=true` vs
   `false` -- se quase tudo vier `false`, o filtro da etapa 1 está frouxo
   (candidatos não relacionados demais); se quase tudo vier `true`, o LLM
   pode estar inventando conexão (o prompt pede explicitamente para não
   fazer isso, mas a taxa é o jeito de checar, não a instrução sozinha).
   Auditoria manual de uma dúzia de pontes geradas, como vocês já fazem para
   fatos incongruentes.
3. **hop-profile estendido, ainda zero LLM adicional.** Com as pontes já
   materializadas nas 10 conversas, rodar o equivalente do hop-profile (D32)
   restrito às perguntas multi-hop que a L2d/L5 hoje erram por
   `recall_context < 1` (174 de 282 nas predições que vocês acabaram de
   puxar). Para cada uma, checar: o turno de evidência que faltava passou a
   estar a 1 salto de algum slot ligado à pergunta, agora que a ponte existe?
   Essa é a pergunta que decide se a L6 vale a pena rodar com LLM em pergunta
   -- e ela é respondida sem gastar um único token de resposta, só
   percorrendo o grafo já construído. Se a fração de misses resolvidos for
   perto de zero, a L6 morre aqui, do jeito mais barato possível.
4. **Oracle nas 10 conversas** (`fgl slots-oracle -C L6`), comparando com L2d
   e L5 -- só depois do passo 3 sugerir que há sinal.
5. **Rodada com LLM completa**, e sempre olhando multi-hop E adversarial
   juntos (a L5 ensinou isso: um canal que ajuda uma categoria pode custar
   caro em abstenção -- uma ponte inventada é exatamente o tipo de coisa que
   pode convencer o modelo a responder uma pergunta adversarial que deveria
   abster.)

Cada um desses passos pode matar a L6 mais barato que o próximo. É a mesma
lógica que fez o hop-profile matar a L3 de graça antes de ela rodar.

## O que fica em aberto, de propósito

Quanto contexto dar ao LLM na etapa 2 -- só os `fact_text` dos dois episódios,
ou o texto bruto das duas trocas -- é uma escolha que vale testar empiricamente
em vez de decidir aqui: `fact_text` é mais barato e mais consistente com o
resto do grafo, mas pode já ter perdido nuance que o texto bruto preservaria.
E se vale dar à etapa 2 uma dica do que tornou o par candidato (ex.: "estes
dois episódios foram flagados por proximidade temática em torno de X") ou
deixar o LLM achar a conexão sozinho -- a primeira opção é mais barata e
precisa, a segunda é mais geral e menos enviesada pelo filtro da etapa 1.
Ambas são decisões de implementação, não de arquitetura, e cabem no passo 2
do plano de teste acima.

## Status: implementado (D34)

Código em `fgl.memory.bridges` (`find_bridge_candidates` = etapa 1,
`synthesize_bridges` = etapa 2), condição `configs/conditions/L6_bridges.yaml`,
prompt genérico em `prompts/bridge_synthesis.txt`, testes em
`tests/test_bridges.py`. O rótulo mudo do Steiner (achado 2 acima) foi
corrigido junto -- ver D34 para o detalhe de código dos dois.

Passo 1 do plano de teste acima (contagem de candidatos, zero LLM) já foi
rodado nas 10 conversas de calibração, com o embedder de produção
(`all-MiniLM-L6-v2`), não o `HashingEmbedder` dos testes: **142 candidatos
no total, ~14 por conversa** -- orçamento barato, bem abaixo do teto
`max_candidates=400`. Números completos e uma amostra dos candidatos (bons e
ruins) estão em D34 (`docs/DECISIONS.md`). O achado principal: a etapa 1 sozinha
não é precisa -- a maioria dos pares acima do limiar são despedidas e
agradecimentos genéricos que soam parecidos por registro, não por conteúdo --
mas isso é o esperado e o previsto neste documento (etapa 1 é recall, etapa 2
é precisão), e a etapa 1 também encontra o que foi desenhada para encontrar
(um exemplo real, sobre a mesma prática de pintura da Melanie em duas sessões
sem vocabulário em comum, está em D34).

Passos 2-5 do plano de teste (amostra de síntese com LLM real, hop-profile
estendido, oracle, rodada completa) **não foram executados** -- exigem uma
chamada de LLM de verdade e ficam para a próxima sessão de medição, seguindo
a mesma disciplina de D32/D33.
