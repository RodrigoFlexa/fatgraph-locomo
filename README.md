# fgl — fatgraph memory for LLM agents

Memória de longo prazo baseada em **fatgraphs (ribbon graphs)**, avaliada no
benchmark **LoCoMo** com o protocolo e o scorer oficiais.

Vértices são entidades canônicas, arestas são memórias atômicas (pares de
meias-arestas), **α** é a involução sem ponto fixo que define a conectividade,
**σ** é a ordem cíclica em cada vértice, e as **faces** — órbitas de `φ = σ∘α` —
são trilhas fechadas de memórias que *emergem* da topologia em vez de serem
declaradas. A recuperação percorre faces (`walk_face`) em vez de fazer k-NN
sobre fatos soltos.

> **Leia primeiro [`docs/COERENCIA.md`](docs/COERENCIA.md).** É a auditoria da
> especificação original: quatro erros que quebrariam a execução ou os números,
> mais um risco estrutural que só aparece rodando.
> [`docs/DECISIONS.md`](docs/DECISIONS.md) registra cada escolha de implementação.

---

## Início rápido (Linux)

A partir da pasta `fatgraph-locomo`:

```bash
python3 -m venv .venv
source .venv/bin/activate

# OBRIGATÓRIO antes do install -e: o venv do Ubuntu 22.04 vem com
# setuptools 59, anterior ao PEP 660. Sem este upgrade o `pip install -e .`
# COPIA o pacote em vez de linkar, e suas edições em src/ são ignoradas.
pip install --upgrade pip setuptools wheel

pip install -e ".[all]"
fgl setup                          # cria .env e busca o dataset (idempotente)
$EDITOR .env                       # preencha AZURE_OPENAI_ENDPOINT e API_KEY
fgl info                           # confere tudo antes de gastar
fgl run-all --dry-run -n 1 -q 10   # offline, sem custo, ~2 s
fgl run G1 -n 1                    # primeira corrida real, 1 conversa
```

`fgl info` mostra `package: editable → .../src/fgl` quando o install está são.
Se aparecer `COPY, not editable`, refaça o upgrade acima e reinstale.

Pacotes de sistema, se faltarem: `sudo apt install python3-venv git`.

**macOS**: idem. **Windows (PowerShell)**: `py -3.11 -m venv .venv` e
`.\.venv\Scripts\Activate.ps1`; se o PowerShell recusar, rode uma vez
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`.

### Rodar o estudo completo em segundo plano

O `run-all` leva horas. A saída redirecionada sai limpa (o Rich desliga a
animação quando não está num terminal), então dá para logar direto:

```bash
tmux new -s fgl                     # ou: screen
source .venv/bin/activate
fgl run-all 2>&1 | tee results/run.log
# Ctrl-B D para destacar; tmux attach -t fgl para voltar
```

Sem tmux:

```bash
nohup fgl run-all > results/run.log 2>&1 &
tail -f results/run.log
```

Pode matar e retomar quando quiser: grafos são persistidos e toda chamada de LLM
fica em cache por hash de prompt, então reexecutar pula o que já foi feito.

### Notebooks numa máquina remota

```bash
jupyter lab --no-browser --port 8888 notebooks/
# na sua máquina: ssh -N -L 8888:localhost:8888 usuario@servidor
```

Sem interface gráfica nenhuma, os notebooks continuam executáveis em lote:

```bash
pip install nbclient
jupyter execute notebooks/01_results_overview.ipynb
```

### Quanto custa e quanto demora

| escopo | chamadas de LLM | tempo aproximado |
|---|---|---|
| `--dry-run` (qualquer escopo) | 0 | segundos — não gasta nada |
| `fgl run G1 -n 1` | ~250 | 5–10 min |
| `fgl run-all -n 1` | ~1 500 | 30–60 min |
| `fgl run-all` (o estudo completo) | ~15 000 | 4–8 h, ~60 M tokens |

Com `gpt-4o-mini` o estudo completo fica na casa de uma dezena de dólares
(confira o preço vigente). O cache por hash de prompt é agressivo: reexecutar
uma corrida interrompida custa quase nada, então dá para parar com `Ctrl-C` e
retomar depois.

### Instalação mais leve

`[all]` traz `sentence-transformers`, que puxa PyTorch (~2 GB). Para evitar:

```bash
pip install -e ".[azure,metrics,notebooks]"
```

e no `.env`: `FGL_EMBEDDING_PROVIDER=azure` — os embeddings passam a vir do seu
deployment `text-embedding-3-small`, sem nenhum modelo local.

Só para explorar a CLI e o `--dry-run`, o núcleo basta (`pip install -e .`):
numpy, PyYAML, typer e rich, ~20 MB. As 6 condições em modo offline rodam em
~2 s nessa instalação mínima.

A condição L1 precisa do extra `bipartite` **e** do download do modelo — o
pacote pip sozinho não traz o modelo:

```bash
pip install -e ".[bipartite]"
python -m spacy download en_core_web_sm
```

### Se algo der errado

| sintoma | causa | solução |
|---|---|---|
| `fgl: command not found` | venv não ativado | `source .venv/bin/activate` |
| editar `src/` não muda nada | install copiado (setuptools < 64) | `pip install -U pip setuptools wheel && pip install -e .` — confirme com `fgl info` |
| exit 3, "Azure não configurado" | `.env` em branco ou com o valor de exemplo | edite o `.env`; a mensagem diz qual variável |
| exit 4, dataset ausente | LoCoMo não baixado | `fgl setup` |
| `No module named venv` | Ubuntu sem o pacote | `sudo apt install python3-venv` |
| tudo dá F1 = 0 | você está em `--dry-run` (LLM falso) | tire o `-d` |
| **todas as condições com o MESMO F1 e adversarial = 1.000** | o backend devolveu respostas vazias; toda pergunta virou abstenção | `fgl doctor` |
| `fgl doctor` mostra resposta vazia com `finish_reason='length'` | deployment de *reasoning*: o orçamento de tokens acabou no raciocínio interno | `--set retrieval.answer_max_tokens=3000 --set llm.max_tokens=8000` |

### Gateway corporativo (credenciais num `.ini`, CA própria)

Se o seu acesso é por gateway — credenciais num `config.ini`, certificado
privado, `base_url` em vez de `azure_endpoint` — não é preciso editar código.
Três variáveis no `.env`:

```bash
FGL_AZURE_CONFIG_INI=config-v1.x.ini     # ConfigParser, seção [OPENAI]
FGL_CA_BUNDLE=petrobras-ca-root.pem      # resolvido a partir da raiz do projeto
FGL_LLM_DEPLOYMENT=gpt-5-mini-petrobras
```

O `.ini` é lido com `ExtendedInterpolation` e aceita as chaves
`OPENAI_API_KEY`, `OPENAI_API_VERSION` e `AZURE_OPENAI_BASE_URL`. Um endpoint
que já contém caminho (`.../openai/v1`) é passado como `base_url`
automaticamente. `fgl info` mostra o que foi reconhecido; `fgl doctor` mostra a
requisição exata antes de enviá-la.

### Deployments de reasoning (gpt-5, o1, o3, o4-mini)

Detectados pelo nome do deployment e configurados sozinhos:

| | chat (gpt-4o) | reasoning (gpt-5, o*) |
|---|---|---|
| limite de tokens | `max_tokens` | `max_completion_tokens`, piso de 4000 |
| `temperature` | enviada | **não enviada** (só aceitam o padrão) |
| `seed` | enviada | não enviada |
| `reasoning_effort` | — | `low` (respostas do LoCoMo são curtas) |

O piso importa: esses modelos gastam o orçamento **pensando antes de emitir
qualquer coisa**. Um limite de 64 tokens — suficiente para uma resposta
extractiva — é consumido inteiro pelo raciocínio e a API devolve `content=""`.

Ajuste em `configs/base.yaml` ou pela linha de comando:

```bash
fgl run G1 --set llm.reasoning_min_tokens=8000 --set llm.reasoning_effort=minimal
fgl run G1 --set llm.reasoning_min_tokens=0     # não envia limite algum
fgl run G1 --set llm.api_style=chat             # força o formato antigo
```

Se o gateway rejeitar algum parâmetro opcional (`seed`, `reasoning_effort`,
`response_format`…), ele é descartado e a chamada refeita automaticamente — uma
vez por parâmetro, e a lição fica registrada para as chamadas seguintes.

### Se o LLM devolver respostas vazias

Um deployment de reasoning (o1/o3/o4-mini, família gpt-5) trata
`max_completion_tokens` como um orçamento **compartilhado** entre o raciocínio
interno e a resposta visível. O padrão de 64 tokens para responder uma pergunta
é consumido inteiro pelo raciocínio, e a API devolve `content=""`.

O framework agora **aborta na primeira resposta vazia** em vez de transformá-la
em abstenção — porque isso produziria uma tabela completa e sem significado, com
adversarial exatamente 1.000 e todas as condições idênticas.

```bash
fgl doctor                                    # mostra a resposta crua e o finish_reason
fgl run G1 -n 1 --set retrieval.answer_max_tokens=3000 --set llm.max_tokens=8000
```

Cada `metrics.json` traz um bloco `sanity` com esses diagnósticos, e o
`fgl report` estampa um aviso no topo quando a corrida é suspeita.


## Configuração

Três camadas, sem sobreposição de responsabilidade:

| onde | o quê | exemplo |
|---|---|---|
| `.env` | segredos e escolha de modelo | `AZURE_OPENAI_API_KEY`, `FGL_LLM_DEPLOYMENT` |
| `configs/base.yaml` | tudo que é comum às condições | thresholds, orçamentos, caminhos |
| `configs/conditions/*.yaml` | só o que difere entre condições | `sigma_policy`, `curation` |

Precedência, da mais baixa para a mais alta:

```
configs/base.yaml → configs/conditions/X.yaml → .env → --dry-run → --set
```

`--set` é aplicado por último de propósito: um override explícito nunca é
descartado em silêncio, nem por `--dry-run`.

Nenhum segredo entra em YAML, e o manifesto de cada resultado guarda apenas
impressões digitais (`"azure_api_key": "<set:9f3c1a2b>"`).

## CLI

```bash
fgl info                       # versões, dependências, credenciais, dataset, condições
fgl setup                      # busca o LoCoMo
fgl config list                # condições disponíveis
fgl config show G1             # configuração resolvida (YAML colorido)
fgl config show G1 --json | jq # saída limpa, sem decoração
fgl config keys retrieval      # tudo que --set aceita, com tipo e default
fgl config diff G2 G3          # exatamente o que muda entre duas condições
fgl config validate            # valida todas as condições, exit != 0 se alguma falhar

fgl ingest G1                  # constrói a memória (só lê diálogos)
fgl qa G1                      # responde e pontua (só lê a memória)
fgl run G1                     # os dois
fgl run-all                    # todas as condições + relatório comparativo
fgl report                     # regenera as tabelas a partir de results/
```

Varredura de hiperparâmetros direto do shell, sem tocar em YAML:

```bash
for m in 3 5 8 12; do
  fgl qa G1 --set retrieval.top_m_anchors=$m \
            --set paths.results_dir=results/sweep-m$m
done
fgl report -r results/sweep-m8
```

Opções comuns: `-d/--dry-run`, `-n/--limit-conversations`,
`-q/--limit-questions`, `-c/--conversation conv-26`, `-s/--set`.

Códigos de saída: `0` ok · `2` erro de configuração · `3` credenciais ausentes ·
`4` dataset ausente.

## Estrutura

```
fatgraph-locomo/
├── .env.example              # template versionado (o .env real é gitignored)
├── pyproject.toml            # pacote instalável, entrypoint `fgl`
├── Makefile
├── configs/
│   ├── base.yaml
│   └── conditions/           # B1 B2 B3 G1..G11 T1 L1 + test_offline
├── data/external/locomo/     # dataset oficial (fgl setup)
├── docs/
│   ├── COERENCIA.md          # auditoria da especificação
│   └── DECISIONS.md          # decisões de implementação
├── notebooks/
│   ├── nbutils.py            # carga + estilo + plots (mesmo código do `fgl report`)
│   ├── 01_results_overview.ipynb
│   ├── 02_graph_topology.ipynb
│   └── 03_retrieval_and_cost.ipynb
├── prompts/                  # 7 prompts versionados (hash vai no manifesto)
├── src/fgl/
│   ├── cli.py                # Typer + Rich
│   ├── settings.py           # .env → Settings, com redação de segredos
│   ├── paths.py              # resolução da raiz do projeto
│   ├── config.py             # YAML → dataclasses, overrides, validação
│   ├── pipeline.py           # orquestração
│   ├── core/                 # α, σ, φ, faces, Euler, curadoria topológica
│   ├── memory/               # extração, entidades, políticas de σ, curadoria
│   ├── retrieval/            # embedders, índice vetorial, walk_face, QA
│   ├── llm/                  # cliente Azure/fake + biblioteca de prompts
│   ├── data/                 # loader do LoCoMo
│   ├── evaluation/           # scorer oficial + geração de relatórios
│   └── baselines/            # B1, B2, B3
├── tests/                    # 108 testes offline
├── artifacts/                # grafos, logs JSONL, cache de fatos (gitignored)
├── results/                  # metrics.json, predictions.jsonl, report.md
└── .cache/                   # cache de LLM e de embeddings (gitignored)
```

## Notebooks

`from nbutils import *; ctx = setup()` carrega tudo e devolve DataFrames prontos
(`ctx.f1`, `ctx.graph`, `ctx.faces`, `ctx.growth`, `ctx.recall`, `ctx.cost`) mais
funções de plot com paleta estável por condição. A carga usa exatamente o mesmo
código do `fgl report`, então um número num gráfico e o mesmo número no terminal
não podem divergir.

| notebook | o que responde |
|---|---|
| `01_results_overview` | F1 por categoria, as três comparações-chave, taxa de abstenção, e as perguntas em que as condições discordam |
| `02_graph_topology` | verificação da fórmula de Euler linha a linha, distribuição de comprimento de faces (C9), folhas vs bígonos (C3), crescimento por sessão |
| `03_retrieval_and_cost` | recall@k, F1 condicionado ao recall, custo por fase e por propósito, F1 por 1k tokens |

Para inspecionar um smoke run offline: `setup(dry=True)`.

## Condições

| id | condição | descrição |
|----|----------|-----------|
| B1 | `full-context` | conversa inteira no prompt |
| B2 | `rag-turns` | k-NN sobre turnos brutos, top-10 |
| B3 | `rag-facts` | k-NN sobre **os mesmos fatos** de G1, sem grafo |
| G1 | `fatgraph-min` | `sigma-time`, sem curadoria nem consolidação |
| G2 | `fatgraph-cur` | `sigma-time` + curadoria + consolidação |
| G3 | `fatgraph-agent` | `sigma-agent` + curadoria + consolidação |
| G4 | `fatgraph-sigma` | G1 + expansão pela órbita de σ na recuperação |
| G5 | `fatgraph-coverage` | G1 + face escolhida por cobertura das entidades da pergunta |
| G6 | `fatgraph-join` | G4 + G5, os dois mecanismos de multi-hop juntos |
| L1 | `bipartite` | grafo turno×entidade, ingestão sem LLM, recuperação sensível a grau |

G1/G2/G3 são os `F1`/`F2`/`F3` da especificação, renomeados para não colidir com
a métrica F1 (`COERENCIA.md` C5).

**G4 ataca o multi-hop.** Uma pergunta multi-hop é, no fatgraph, duas memórias na
mesma órbita de σ — duas arestas incidentes no mesmo vértice. `φ = σ∘α` sai do
vértice a cada passo, então a face só reencontra a entidade-ponte depois de uma
volta na superfície, em geral além do `budget_tokens`. A G4 percorre σ direto, a
partir das **duas** entidades da memória-âncora. Custo: zero chamadas de LLM.

**G5 inverte a unidade de recuperação.** Em vez de `argmax` sobre meia-aresta
com a face vindo de brinde, a **face** é o que se recupera, pontuada por
similaridade agregada **mais a cobertura das entidades que a pergunta nomeia**.
Cobertura é sinal estrutural: uma face que passa pelos dois vértices é candidata
a ponte mesmo que nenhum fato dela pareça com a pergunta — o que o cosseno não
consegue expressar. Sem face cobrindo duas entidades, cai para a geodésica entre
elas.

**G6 = G4 + G5.** São ortogonais: σ expande a partir de um âncora certo ao qual
falta o segundo salto; a cobertura escolhe qual trilha recuperar quando o âncora
é irrelevante.

```bash
fgl ingest G1                        # obrigatório: G4/G5/G6 REUTILIZAM os grafos da G1
fgl run G4 && fgl run G5 && fgl run G6
fgl report                           # tabelas comparativas + auditoria
```

Cada uma difere da G1 apenas no bloco de recuperação (`fgl config diff G1 G5`) e
lê os grafos da G1 byte a byte — então o delta isola a recuperação, e não a
extração nem a resolução de entidades. Com os dois flags em `false` (o default) o
caminho de código é o antigo, de modo que resultados já guardados de G1–G3
continuam reproduzíveis.

**B3 é a ablação crítica**: B3 e G1 consomem fatos byte-a-byte idênticos, lidos
do mesmo cache de extração — que não depende da condição, e há um teste que trava
isso. A única diferença entre os dois é a topologia.

Comparações: **B3 → G1** (valor das faces), **G1 → G2** (curadoria/consolidação),
**G2 → G3** (o agente escolhendo σ), **G1 → G4** (o salto por σ, mesmo grafo),
**G1 → G5** (cobertura de entidades), **G4 → G6** e **G5 → G6** (cada mecanismo
adicionado ao outro — é o par que mostra se eles somam ou se se sobrepõem).

**L1 muda a ingestão, não a recuperação sobre o mesmo grafo.** G4–G6 herdam o
diagnóstico da estrela (81% dos vértices em grau 1, os dois falantes como os
vértices de maior grau) porque leem os grafos da G1 byte a byte — o extrator de
triplas, generativo, raramente reusa frase para a "outra" entidade, então quase
nada recorre para a resolução de entidade fundir. L1 não extrai triplas: um
vértice por turno, um vértice por entidade canônica (spaCy NER + noun-chunks,
determinístico, `fgl.memory.ner`), uma aresta por menção observada — zero
chamadas de LLM na ingestão. `sigma` sai de graça da ordem de processamento nos
dois vértices (leitura no turno, cronologia na entidade), sem precisar de
`SigmaPolicy`. O falante nunca é vértice — decidido contando o LoCoMo real
(79% do multi-hop é uma pessoa só, sem ponte nenhuma; só 18% precisa da ponte
entre dois falantes) — e a recuperação (`fgl.retrieval.bipartite`) classifica
cada entidade ligada pelo grau: grau 1 é acerto direto, grau intermediário tem
a órbita inteira enumerada, hub vira só filtro/bônus, nunca caminhado. Medido
em `fgl ingest L1 -n 10`: `degree_1_frac` cai de 81% para 51,3%, `hub_share` de
~45% para 2,09%. Detalhes, degenerescências encontradas ao medir e o que essa
condição ainda não faz (sem detecção de incongruência, pouca cobertura de
adversarial) em `docs/DECISIONS.md` D24.

```bash
fgl ingest L1 -n 10    # constrói SEUS PRÓPRIOS grafos, não reaproveita a G1
fgl run L1
```

## O que sai de cada corrida

`results/<condição>/metrics.json`:

* **F1 oficial** por categoria e agregados macro/micro — todas as 1986 perguntas,
  nada filtrado nem subamostrado;
* **recall@k** (k ∈ {5,10}) contra as evidências anotadas, mais `recall_context`;
* **estatísticas do grafo** por conversa e por sessão: V, E, F, C, gênero,
  histograma de comprimento de faces, bígonos vs folhas, colapsos, consolidações,
  incongruências;
* **custo** em tokens separado entre ingestão e QA, e por propósito de chamada;
* **manifesto**: config completa, hash de cada prompt, ambiente redigido, versão
  do Python, commit do git, e qual stemmer o scorer usou;
* **auditoria da cobertura** (só quando ligada): `coverage_link_rate` (a
  pré-condição de tudo — sem entidade ligada não há sinal e a condição vira G1),
  `coverage_bridge_rate` (faces cobrindo 2+ entidades), `coverage_evidence_rate`,
  `geodesic_rate` e `recall_context_no_coverage`;
* **auditoria da expansão por σ** (só quando ligada): `sigma_use_rate`,
  `sigma_evidence_rate` (fração de perguntas em que σ alcançou uma evidência que
  nenhuma face alcançou), `recall_context_no_sigma` — o mesmo recall calculado
  como se a expansão não tivesse rodado — e `sigma_dup_rate`, que distingue
  "a face já cobria a órbita" de "as órbitas estão vazias". Corridas antigas e
  condições com o flag desligado simplesmente não têm esse bloco.

`results/<condição>/predictions.jsonl` tem uma linha por pergunta.
Decisões do agente (posição de inserção, justificativa, julgamentos de curadoria
e incongruência) vão para `artifacts/logs/<condição>/<conversa>.jsonl`.

## Notas de protocolo

* A memória é construída **só** a partir dos diálogos; a fase de QA lê **só** a
  memória (o grafo é recarregado do disco). B1/B2 são as exceções, por definição.
* Perguntas temporais recebem o sufixo do pipeline oficial — sem ele o F1 da
  categoria 2 despenca por incompatibilidade de formato.
* Adversariais são respondidas em **formato livre**, não em múltipla escolha como
  no pipeline oficial. A regra de pontuação é a mesma, mas o cenário é mais
  difícil: os números de categoria 5 **não são comparáveis** aos publicados.
  Detalhes em `COERENCIA.md` C8.

## Desenvolvimento

```bash
make test        # 108 testes, offline, ~5s
make lint        # ruff
make smoke       # 6 condições ponta a ponta, offline
make clean-dry   # remove só os artefatos de dry-run
```

A suíte roda sem rede, sem credenciais e sem download de modelo: o backend
`FakeLLM` é determinístico e o `HashingEmbedder` não tem dependências. Testes que
precisam do dataset são pulados automaticamente até você rodar `fgl setup`.

## Fase 2

`whitehead_flip` está implementado e testado (preserva C, gênero e |F| num grafo
teta), desligado por `curation.whitehead_flip: false`.
