# Runbook — rodar a linha L no servidor

> **Atualizado 2026-08-19, depois do primeiro oracle completo.** O veredito
> mudou: **rode a L2d**, não a L3 nem a L4. Ver `docs/DECISIONS.md` D32 para os
> números. A L3 está medida como negativa (o `hop-profile` avisou antes) e a L4
> perde 0.012 de recall geral para a L2d. A pergunta em aberto é a **L5**
> (`L2d + só o canal de conexão`), que custa 3 minutos: `fgl slots-oracle -C L2d -C L5`.

Escrito para ser seguido de cima para baixo. **As três primeiras etapas não
gastam um único token de LLM** e decidem se as duas últimas valem a pena — se
alguma delas disser não, pare ali e me diga o que apareceu, é informação e não
fracasso.

Tempo total estimado: ~15 min de diagnóstico grátis, depois ~1h50 por condição
com LLM (a L2 levou 6742 s).

---

## 0. Pré-requisitos

Nada novo para instalar. As implementações usam só `numpy` e a `heapq` da
biblioteca padrão — nada de `scipy`, nada de `networkx`.

```bash
cd ~/fatgraph-locomo
source .venv/bin/activate
git pull                     # ou aplique os arquivos que enviei
pytest -q                    # 500 testes, ~1 min. Tem que estar verde.
fgl config show L3 --paths   # sanity: as condições novas carregam
fgl config show L4 --paths
```

Se `pytest` falhar, **pare** — nada abaixo é confiável.

---

## 1. O portão: onde a evidência está, em saltos  ⏱ ~3 min, custo zero

Esta é a etapa que decide a L3. Um passeio mais longo só encontra evidência que
esteja *alcançável naquele número de saltos*; este comando mede onde ela está.

```bash
fgl hop-profile -C L2 --out artifacts/hops_L2.json
```

**O que olhar**, na tabela `evidence this condition MISSED (the headroom)`:

| leitura | significado | o que fazer |
|---|---|---|
| coluna **hop 2** ≥ 0.10 | há junção real para achar | siga para a etapa 2, a L3 tem alvo |
| coluna **hop 2** ≈ 0 e **hop 1** alto | a evidência já era alcançável; o problema é ranking | a L3 não vai mover; o ganho está no scorer, não no alcance |
| coluna **unreach** alta | não existe caminho no grafo | é problema de **ingestão**, não de leitura — nenhum passeio resolve |

O comando imprime esse veredito sozinho, em verde ou amarelo, no fim.

Ele também imprime os números de ribbon structure. Espere ver algo próximo de
`faces used: 108 of a bipartite ceiling of 34355 (0.31%)` e
`genus 23822 against a floor of 6699 (3.56x)` — é a versão quantitativa de "as
faces não carregam informação", e o `episode pairs sharing 2+ non-hub slots` é
a contagem direta dos 4-ciclos que um mergulho quadrangular viraria faces.

> **Se quiser me mandar uma coisa só desta semana, mande este output.** É o que
> diz se a direção está certa.

---

## 2. Comparar as leituras sem gastar nada  ⏱ ~10 min, custo zero

```bash
fgl slots-oracle -C L1 -C L2 -C L2d -C L3 -C L4 --out artifacts/oracle_L.json
```

A L3 **não paga ingestão** — ela lê os grafos da L2 (`graphs_condition:
L2-slots`), então precisa que a L2 já tenha rodado. Se os grafos não existirem
o comando diz exatamente qual condição construir primeiro. A L2d e a L4
constroem os seus (tempo multirresolução muda o conjunto de vértices), ainda com
zero chamadas de LLM.

**O que olhar:**

1. **`recall_context` por categoria.** A L3 tem que subir em *multi-hop* — é
   para isso que ela existe. Se subir em multi-hop e cair em single-hop, o
   passeio está trazendo ruído: baixe `propagation.decay`.
2. **A linha `how the graph is read`.** Confira `bridgeable=` — é a fração de
   slots que o passeio pode atravessar. Se estiver perto de 0, o passeio não tem
   por onde andar e a L3 virou a L2 em silêncio, independentemente de
   `hops=2`. Se estiver perto de 1, o corte de hub não está filtrando nada.
3. **`mean tokens` / `mean units`.** Têm que estar próximos entre as condições.
   Se a L4 gastar muito menos, a abstenção está disparando demais — veja o item
   4.
4. **A linha de abstenção.** A L4 reporta os motivos separados
   (`dead_terminal`, `disconnected`, `far_apart`). Regra de bolso: com
   adversarial em 446 e substantivas em 1540, a abstenção só vale a pena se
   `caught / 446 > 2.5 × (false_positives / 1540)`. Se não valer, afrouxe:

   ```bash
   fgl slots-oracle -C L4 --set steiner.abstain_quantile=0.99
   ```

---

## 3. A curva que passa pelo número publicado  ⏱ ~5 min, custo zero

```bash
fgl slots-sweep -C L3 -k propagation.hops \
  --out artifacts/sweep_hops.json --html artifacts/sweep_hops.html
```

`hops=1` com `normalization=none` reproduz a L2 **exatamente** (turno por turno,
score por score — está no teste). Então o ponto mais à esquerda desta curva é o
número publicado da L2, e a subida daí é a contribuição do passeio, isolada.

Vale a pena rodar mais dois de uma vez:

```bash
fgl slots-sweep -C L4 \
  -k propagation.hops -k propagation.decay -k propagation.dense_seed \
  -k steiner.weight -k steiner.abstain_quantile \
  --out artifacts/sweep_L4.json --html artifacts/sweep_L4.html
```

Leia a coluna **verdict**: `flat` significa que o número não é resultado
nenhum (ótimo, é defesa); `peaked` significa que é, e precisa ser declarado.

Duas ablações que valem cada uma um minuto e respondem a uma objeção de revisor:

```bash
# "um hub é filtro, nunca ponte" — a regra que sustenta as três leituras
fgl slots-oracle -C L3 --set propagation.bridge_hubs=true

# o passeio não-retornante — sem ele o salto 2 é a semente refletida
fgl slots-oracle -C L3 --set propagation.non_backtracking=false
```

As duas **devem piorar**. Se não piorarem, a regra não está fazendo o que eu
disse que faz, e isso é mais importante saber do que o número final.

---

## 4. Os runs com LLM  ⏱ ~1h50 cada

Só depois que a etapa 2 mostrar ganho de `recall_context`. Rode nesta ordem e
pare no meio se quiser — cada um é independente:

```bash
fgl run L3 2>&1 | tee artifacts/run_L3.log
fgl run L4 2>&1 | tee artifacts/run_L4.log
fgl report
```

Se o tempo for curto e couber só um: **rode a L4.** Ela contém a L3 e é a única
que ataca a regressão de adversarial, que é onde 45% do ganho da última rodada
foi embora.

---

## 5. Como ler o `fgl report` final

Não olhe o micro primeiro. Olhe nesta ordem:

1. **adversarial.** Se não voltar na direção de 0.666, a abstenção por conexão
   não está funcionando e é o primeiro lugar para mexer. Ela sozinha vale
   ~0.013 de micro.
2. **multi-hop.** É o alvo direto do passeio e do canal de conexão. 0.376 é a
   base.
3. **open-domain.** É a categoria que vive no canal denso; `dense_seed: 0.5` na
   L4 é a única coisa que a ataca. 0.249 é a base.
4. **micro**, por último, e comparado com `recall_context` da etapa 2. **Se o
   recall subir muito e o micro pouco de novo, o gargalo é a geração, não a
   recuperação** — e aí a próxima semana é sobre prompt e não sobre grafo.

Para responder essa última pergunta de forma direta, o número que vale mais que
o micro:

```bash
fgl diagnose L4 --show 8
```

F1 condicionado a `recall_context = 1`. Se ele caiu de L2 para L4, mais
recuperação está virando distração e nenhuma topologia resolve isso.

---

## Referência rápida das condições

| condição | o que muda | grafos | custo de ingestão |
|---|---|---|---|
| `L2` | slots tipados, 1 salto | próprios | zero LLM |
| `L2d` | L2 com todo limiar derivado do corpus | próprios | zero LLM |
| `L3` | L2 + passeio de 2 saltos não-retornante | **empresta da L2** | nenhum |
| `L4` | L2d + passeio + conexão Steiner + abstenção derivada | próprios | zero LLM |

`L3` é isolamento (muda uma coisa, o delta é atribuível). `L4` é síntese (a
pergunta que um isolamento não responde: as peças compõem?).
