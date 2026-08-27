# Passo 1 — seleção por cobertura de consulta: RESULTADO NEGATIVO

**Data:** 2026-08-22 · **Escopo:** L2d-derived, 10 conversas, 1986 perguntas, zero LLM
**Veredito: a hipótese foi rejeitada.** Cobertura não move multi-hop e custa nas outras categorias.

## O que foi construído

`slots.selection = topk | coverage | mmr` em `SlotsConfig`, mais `coverage_decay`,
`coverage_keep_pairs` e `mmr_lambda`. A seleção sai do `_emit` para três políticas:

- **coverage** — guloso submodular. O escore de um turno decompõe em
  `nonslot(ep) + actor_mult(ep)*Σ_v hits[ep][v] + dense(t) + sibling(t)`; só o termo do
  meio é uma afirmação de *cobrir a pergunta*, e só ele é descontado por
  `decay ** n_v`, com `n_v` = nº de episódios distintos já selecionados que cobrem o
  slot `v`. Guloso preguiçoso (CELF). O desconto vale ENTRE episódios, nunca dentro de
  um: um turno cujo episódio já entrou herda o ganho com que o episódio foi admitido
  (`coverage_keep_pairs`), senão a segunda metade do par adjacente seria descontada a
  zero — que é a regressão 0,858 → 0,682 do single-hop entrando pela porta da frente.
- **mmr** — MMR clássico sobre embeddings de turno. É o controle honesto: se MMR mover
  multi-hop tanto quanto a cobertura, a decomposição tipada não é o que paga.

**Garantia de identidade:** `coverage_decay = 1.0` reproduz `topk` turno a turno.
Verificado em 304 perguntas (2 conversas), 0 divergências. O "off" da ablação é o mesmo
caminho de código, não outro.

**Bug encontrado e corrigido durante a implementação:** a condição de reinserção do
guloso preguiçoso estava com o sinal invertido — reempurrava exatamente o item que
deveria aceitar. Custo antes do conserto: >4,5 s/pergunta; depois: 20 ms/pergunta.

## Resultado

n = 1986 perguntas

RECALL_CONTEXT por categoria
braço                 single-ho  multi-hop   temporal  open-doma  adversari      GERAL
topk                     0.9170     0.6533     0.9076     0.5395     0.8711     0.8495
mmr-0.7                  0.9195     0.6328     0.9047     0.5557     0.8823     0.8505
cover-0.7                0.9082     0.6518     0.8951     0.5438     0.8632     0.8420
cover-0.5                0.9047     0.6449     0.8941     0.5483     0.8464     0.8358
cover-0.3                0.8981     0.6409     0.8801     0.5569     0.8307     0.8271
cover-0.0                0.8835     0.6406     0.8738     0.5621     0.8038     0.8140
cover-0.5-nopair         0.8971     0.6457     0.8904     0.5614     0.8229     0.8275

DELTA vs topk
braço                 single-ho  multi-hop   temporal  open-doma  adversari      GERAL
mmr-0.7                 +0.0026    -0.0205    -0.0029    +0.0162    +0.0112    +0.0010
cover-0.7               -0.0087    -0.0015    -0.0125    +0.0043    -0.0078    -0.0075
cover-0.5               -0.0123    -0.0084    -0.0135    +0.0088    -0.0247    -0.0137
cover-0.3               -0.0188    -0.0124    -0.0275    +0.0174    -0.0404    -0.0224
cover-0.0               -0.0335    -0.0127    -0.0337    +0.0226    -0.0673    -0.0355
cover-0.5-nopair        -0.0198    -0.0076    -0.0171    +0.0219    -0.0482    -0.0220

A ASSINATURA: recall do multi-hop por nº de peças de evidência exigidas
braço                        1         2         3         4        5+
(n)                          6       134        57        45        40
topk                     0.833     0.720     0.620     0.511     0.610
mmr-0.7                  0.833     0.705     0.602     0.483     0.572
cover-0.7                0.833     0.720     0.637     0.517     0.568
cover-0.5                0.833     0.713     0.620     0.522     0.563
cover-0.3                0.833     0.705     0.614     0.528     0.562
cover-0.0                0.833     0.705     0.608     0.533     0.562
cover-0.5-nopair         0.833     0.709     0.626     0.533     0.561

Por nº de SESSÕES distintas que a evidência atravessa (todas as categorias)
braço                        1         2        3+
(n)                       1652       204       126
topk                     0.892     0.704     0.528
mmr-0.7                  0.897     0.694     0.492
cover-0.7                0.882     0.708     0.528
cover-0.5                0.876     0.700     0.524
cover-0.3                0.865     0.697     0.528

## Leitura

1. **Multi-hop não se move.** O melhor braço de cobertura fica em −0,0015 (ruído);
   apertar o desconto piora. MMR fica em −0,0205.
2. **A assinatura prevista não apareceu.** A previsão era achatar a queda do recall
   conforme a pergunta exige mais peças. O que houve foi ganho minúsculo em 3–4 peças
   (+0,017 / +0,022, n=57 e n=45) e perda clara em 5+ (−0,04). Não é achatamento.
3. **Custa nas outras categorias**, sobretudo adversarial (−0,067 no desconto máximo),
   que precisa de concentração e não de espalhamento.
4. **O único sobrevivente é open-domain:** +0,023, e monótono na força do desconto
   (0,5395 → 0,5621). n=96, então é indício, não resultado — mas a monotonicidade
   através de cinco braços não parece ruído.
5. Orçamento preservado: ~1985 tokens em todos os braços; a cobertura até compra mais
   unidades (62,7 contra 59,7), então não é artefato de gastar menos.

## Por que falhou — o diagnóstico que sobra

Medido sobre a evidência multi-hop que a L2d perde (347 turnos):

| onde estava a evidência perdida | |
|---|---|
| com um vizinho (±2 turnos) já no contexto | 120 (34,6%) |
| **em região não alcançada** | **227 (65,4%)** |

E o contexto já toca ~17 das ~33 sessões da conversa. **Amplitude não era o gargalo.**

Isso corrige uma leitura minha anterior: eu havia medido "67% da evidência perdida está
em sessão JÁ TOCADA" e li aquilo como "é só reordenar". Na granularidade que importa —
a região local, não a sessão — 65% está a mais de dois turnos de qualquer coisa
recuperada. A sessão tinha sido tocada em OUTRO lugar.

Daí o motivo estrutural da falha: **a seleção só redistribui o orçamento entre os
candidatos que já foram pontuados.** Se o episódio da evidência que falta não recebeu
escore, nenhuma política de seleção o alcança. Cobertura era a camada errada.

## Para onde isso aponta

O padrão do multi-hop é `q → e1 → e2`, onde `e2` se liga a `e1`, não a `q`. Nossa
pontuação é de um salto **a partir da pergunta**, então `e2` frequentemente não recebe
escore algum. A L3 tentou dois saltos a partir da PERGUNTA e perdeu, porque isso é
alcance não direcionado (ruído).

O que não foi tentado: expansão **condicionada à evidência já selecionada** — depois do
primeiro passe, semear um segundo passe com os slots dos episódios escolhidos, gastando
uma fração limitada do orçamento. É exatamente o `evidence closure` do Zero-Mem
(`C(q) = Dedup(M(q) ∪ N_g(M(q)) ∪ N_h(M(q)))`), que nós não temos e eles têm — e a
diferença de multi-hop entre os dois métodos é 41,61 contra 37,77.
