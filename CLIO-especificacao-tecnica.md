# CLIO: Chronologically Layered Interval Ontology

**Especificação técnica de implementação, v1.0**

Memória de longo prazo para agentes conversacionais baseada em grafo bitemporal
dobrado, com uma álgebra de acesso composicional em vez de roteamento por tipo de
pergunta.

Este documento é autocontido e serve como especificação de implementação. Toda
decisão de projeto está justificada para que o implementador saiba o que pode e o
que não pode alterar.

---

## Sumário

1. [Princípios invioláveis](#1-princípios-invioláveis)
2. [Visão geral da arquitetura](#2-visão-geral-da-arquitetura)
3. [Modelo de dados](#3-modelo-de-dados)
4. [O catálogo de relações (Σ)](#4-o-catálogo-de-relações-σ)
5. [Normalização temporal](#5-normalização-temporal)
6. [Pipeline síncrono: ingestão e extração](#6-pipeline-síncrono-ingestão-e-extração)
7. [Pipeline assíncrono: consolidação](#7-pipeline-assíncrono-consolidação)
8. [Dobra e resolução de entidades](#8-dobra-e-resolução-de-entidades)
9. [A álgebra de acesso](#9-a-álgebra-de-acesso)
10. [Interface com o agente](#10-interface-com-o-agente)
11. [Geração da resposta](#11-geração-da-resposta)
12. [Persistência e índices](#12-persistência-e-índices)
13. [Complexidade e orçamentos](#13-complexidade-e-orçamentos)
14. [Configuração](#14-configuração)
15. [Estrutura de arquivos](#15-estrutura-de-arquivos)
16. [Plano de implementação](#16-plano-de-implementação)
17. [Testes](#17-testes)
18. [Avaliação](#18-avaliação)
19. [Modos de falha e não-objetivos](#19-modos-de-falha-e-não-objetivos)

---

## 1. Princípios invioláveis

Estes cinco princípios determinam o comportamento do sistema. Uma implementação
que viole qualquer um deles não é CLIO.

**P1. O log episódico é a verdade. O grafo é índice.**
Nenhum episódio é editado ou removido. Todo o resto é derivado e reconstruível a
partir do log. Isso é o que torna reversível qualquer erro de extração ou de
fusão de entidades.

**P2. Nada é apagado. Intervalos são fechados.**
Atualizar um fato é escrever uma data de fim, nunca deletar uma linha. Isso é o
que permite responder "onde ela mora", "onde ela morava em maio" e "quando você
soube que ela mudou" a partir das mesmas arestas.

**P3. O LLM propõe, o código decide.**
O LLM é usado em exatamente dois pontos: extração de proposições candidatas
(obrigatório) e escrita de sumários (opcional). Ele nunca decide onde escrever,
nunca calcula datas, nunca funde entidades, nunca atribui confiança numérica.
Todas as decisões de atualização são código determinístico parametrizado pelo
catálogo.

**P4. Toda janela é a interseção do caminho.**
Uma trilha de acesso carrega a interseção dos intervalos de todas as arestas que
percorreu. Interseção vazia mata a trilha. Coerência temporal de caminho é uma
propriedade da estrutura de dados, não uma verificação opcional.

**P5. A resposta é gerada a partir do episódio, não da proposição.**
Proposições servem para localizar episódios com precisão temporal. Quem redige a
resposta lê o texto original. Isso impede que um erro de extração vire
automaticamente um erro de resposta.

---

## 2. Visão geral da arquitetura

```
                         ESCRITA
  turno ──► ingestão ──► extração (LLM) ──► staging
                                              │
                              (assíncrono)    ▼
                            normalização temporal
                            endereçamento
                            cardinalidade e dependências
                            dobra
                            promoção do staging
                                              │
                                              ▼
                                   GRAFO BITEMPORAL DOBRADO
                                              │
                         LEITURA              ▼
  agente ──► movimentos da álgebra ──► trilhas ──► evidência ──► log
```

Três camadas de armazenamento:

| Camada | Conteúdo | Mutabilidade |
|---|---|---|
| `log` | episódios, menções | append-only |
| `staging` | proposições extraídas, ainda não consolidadas | append + promoção |
| `graph` | arestas bitemporais dobradas | intervalos fechados, nunca deletados |

Dois eixos de tempo em toda aresta:

- `t_valid`: quando o fato é verdade no mundo.
- `t_tx`: quando o agente acreditou no fato.

Fechar `t_valid` significa "deixou de ser verdade". Fechar `t_tx` significa
"nunca foi verdade, eu estava errado". São operações distintas com efeitos
observáveis distintos.

---

## 3. Modelo de dados

### 3.1 Entidades centrais

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Literal

# Um intervalo semiaberto [start, end). end=None significa aberto à direita.
@dataclass(frozen=True)
class Interval:
    start: Optional[datetime]   # None = desde sempre
    end: Optional[datetime]     # None = até agora

    def intersect(self, other: "Interval") -> Optional["Interval"]:
        s = max_opt(self.start, other.start, default_min=True)
        e = min_opt(self.end, other.end, default_max=True)
        if s is not None and e is not None and s >= e:
            return None
        return Interval(s, e)

    def contains(self, t: datetime) -> bool: ...
    def overlaps(self, other: "Interval") -> bool: ...
    def is_open(self) -> bool: return self.end is None
```

Convenção obrigatória: intervalos são **semiabertos** `[start, end)`. Isso torna
o fechamento funcional exato, sem sobreposição de um instante.

```python
@dataclass
class Episode:
    id: str                  # "ep_000173"
    session_id: str
    speaker: str
    text: str
    ts_ingest: datetime      # carimbo de ingestão, base de toda âncora temporal
    seq: int                 # ordem total no log
    meta: dict = field(default_factory=dict)


@dataclass
class Entity:
    id: str                  # "ent_00042"
    canonical_name: str
    type: str                # "Person", "Organization", "Place", "Activity", ...
    aliases: list[str] = field(default_factory=list)
    created_from: str = ""   # episode id
    merged_into: Optional[str] = None   # se != None, este id é um alias de outro


class EvidenceKind(str, Enum):
    LITERAL = "literal"                 # trecho verbatim
    COREFERENCE = "coreference"         # literal, mas com pronome resolvido
    IMPLICATURE = "implicature"         # "também", "de novo", "ainda"
    CONTEXTUAL = "contextual"           # inferida de contexto amplo


class Operation(str, Enum):
    ASSERT = "assert"       # afirma um fato
    REASSERT = "reassert"   # reafirma um fato já conhecido
    CLOSE = "close"         # o fato deixou de ser verdade  -> fecha t_valid
    RETRACT = "retract"     # o fato nunca foi verdade      -> fecha t_tx


@dataclass
class Proposition:
    id: str
    subject_id: str
    relation: str
    object_id: str
    operation: Operation
    polarity: bool                      # False = negação explícita
    time_expression: Optional[str]      # TRECHO LITERAL, não data
    t_valid: Optional[Interval]         # preenchido pelo normalizador
    t_tx: Interval                      # [ts do episódio, None)
    evidence_kind: EvidenceKind
    confidence: float                   # derivado de evidence_kind, não do LLM
    span: str                           # trecho exato do texto
    episode_id: str
    status: Literal["staged", "promoted", "rejected"] = "staged"


@dataclass
class Edge:
    id: str
    src_id: str
    label: str                          # em Σ, ou label+"⁻¹"
    dst_id: str
    t_valid: Interval
    t_tx: Interval
    provenance: list[str]               # proposition ids
    reinforcement: int = 1              # número de confirmações independentes
    last_confirmed: Optional[datetime] = None
    confidence: float = 0.0             # max das proposições de origem
    conflict_flag: bool = False


@dataclass
class Mention:
    """Registro cru para contagem. Não passa por consolidação."""
    id: str
    episode_id: str
    entity_id: Optional[str]
    surface: str
    ts: datetime
```

### 3.2 Por que `Mention` existe separadamente

A dobra funde ocorrências repetidas em uma aresta canônica. É exatamente o que ela
deve fazer, e é exatamente o que destrói multiplicidade. Perguntas de contagem
("quantas vezes ela mencionou escalada") não podem ser respondidas no grafo. O
movimento `count` consulta `Mention` e `Episode`, nunca `Edge`.

### 3.3 Arestas inversas

Para relações declaradas `invertible: true`, o consolidador materializa a aresta
inversa com label `label⁻¹`. Isso permite que `follow` caminhe nos dois sentidos
sem lógica especial. Arestas inversas compartilham `provenance`, `t_valid` e
`t_tx` com a direta e são atualizadas em conjunto (mesma transação).

Relações não inversíveis não geram inversa. Tentar `follow("label⁻¹")` sobre elas
retorna erro de rótulo inexistente.

---

## 4. O catálogo de relações (Σ)

O catálogo é a peça mais importante do sistema. Toda a inteligência temporal está
aqui, não no LLM. É um arquivo versionado, carregado na inicialização.

### 4.1 Esquema

```yaml
relations:
  - name: works_at
    signature: [Person, Organization]
    cardinality: functional        # functional | multi
    volatility: slow               # static | slow | fast
    invertible: true
    inverse_name: employs
    default_duration: null         # se null, intervalo aberto à direita
    closes_on_new: true            # novo valor fecha o anterior
    dependents: [managed_by, works_with]   # fechar isto fecha os dependentes
    aliases_surface: ["trabalha em", "trabalha na", "empregado em"]
```

### 4.2 Semântica de cada campo

| Campo | Efeito na consolidação |
|---|---|
| `signature` | Proíbe dobra entre destinos de tipos incompatíveis. Filtra o catálogo enviado ao extrator. |
| `cardinality` | `functional`: no máximo uma aresta viva por instante; valor novo fecha o anterior. `multi`: arestas coexistem. |
| `volatility` | Define `default_duration` quando não há expressão temporal. `static`: intervalo infinito. `slow`: aberto à direita. `fast`: janela curta (default 1 dia). |
| `invertible` | Materializa aresta inversa. |
| `closes_on_new` | Se `false`, um valor novo não fecha o anterior mesmo sendo funcional (raro; use para relações onde a sobreposição é um conflito a sinalizar, não a resolver). |
| `dependents` | Ao fechar `t_valid` desta relação, fecha os dependentes na mesma data. Grafo de dependências deve ser acíclico. |

### 4.3 Catálogo inicial para diálogo pessoal

Implementar como `catalog/personal_dialogue.yaml`. Este é o conjunto mínimo
viável; expandir conforme a Seção 4.4.

```yaml
types: [Person, Organization, Place, Activity, Object, Event, Topic]

relations:
  # --- Residência e localização ---
  - name: lives_in
    signature: [Person, Place]
    cardinality: functional
    volatility: slow
    invertible: false
    dependents: []

  - name: born_in
    signature: [Person, Place]
    cardinality: functional
    volatility: static
    invertible: false

  # --- Trabalho ---
  - name: works_at
    signature: [Person, Organization]
    cardinality: functional
    volatility: slow
    invertible: true
    inverse_name: employs
    dependents: [managed_by, works_with, has_role]

  - name: has_role
    signature: [Person, Topic]
    cardinality: functional
    volatility: slow
    invertible: false

  - name: managed_by
    signature: [Person, Person]
    cardinality: functional
    volatility: slow
    invertible: true
    inverse_name: manages

  - name: works_with
    signature: [Person, Person]
    cardinality: multi
    volatility: slow
    invertible: true
    inverse_name: works_with

  - name: hired
    signature: [Person, Person]
    cardinality: multi
    volatility: static
    invertible: true
    inverse_name: hired_by

  # --- Relações pessoais ---
  - name: family_of
    signature: [Person, Person]
    cardinality: multi
    volatility: static
    invertible: true
    inverse_name: family_of

  - name: partner_of
    signature: [Person, Person]
    cardinality: functional
    volatility: slow
    invertible: true
    inverse_name: partner_of

  - name: friend_of
    signature: [Person, Person]
    cardinality: multi
    volatility: slow
    invertible: true
    inverse_name: friend_of

  # --- Gostos e atividades ---
  - name: practices
    signature: [Person, Activity]
    cardinality: multi
    volatility: slow
    invertible: false

  - name: likes
    signature: [Person, Topic]
    cardinality: multi
    volatility: slow
    invertible: false

  - name: dislikes
    signature: [Person, Topic]
    cardinality: multi
    volatility: slow
    invertible: false

  # --- Posses ---
  - name: owns
    signature: [Person, Object]
    cardinality: multi
    volatility: slow
    invertible: true
    inverse_name: owned_by

  # --- Planos e eventos ---
  - name: attended
    signature: [Person, Event]
    cardinality: multi
    volatility: fast
    invertible: true
    inverse_name: attended_by

  - name: plans_to
    signature: [Person, Event]
    cardinality: multi
    volatility: fast
    invertible: false

  # --- Estudo ---
  - name: studies_at
    signature: [Person, Organization]
    cardinality: functional
    volatility: slow
    invertible: true
    inverse_name: enrolls
    dependents: [studies_subject]

  - name: studies_subject
    signature: [Person, Topic]
    cardinality: multi
    volatility: slow
    invertible: false
```

### 4.4 Crescimento do catálogo

O extrator opera com vocabulário fechado, porque rótulos livres nunca dobram. Mas
o vocabulário não é fixo para sempre.

Quando nenhum rótulo permitido serve, o extrator emite:

```json
{ "operation": "assert", "relation": "UNMAPPED",
  "suggested_relation": "adopted_pet",
  "subject_id": "ent_001", "object_surface": "gato",
  "span": "adotei um gato semana passada" }
```

Isso vai para a tabela `unmapped_queue`, nunca para o grafo. Um processo
periódico (`clio.catalog.mine_unmapped`) agrupa sugestões por similaridade,
e emite um relatório de candidatos com contagem de ocorrências. A promoção a tipo
do catálogo é uma ação humana, versionada. Depois da promoção, `clio.rebuild`
reprocessa o log e as ocorrências antigas entram no grafo.

---

## 5. Normalização temporal

Módulo determinístico. **O LLM nunca calcula datas.** Ele devolve o trecho
literal, e este módulo o converte.

### 5.1 Assinatura

```python
def resolve_time(
    expression: Optional[str],
    anchor: datetime,             # ts_ingest do episódio
    relation: RelationSpec,
    locale: str = "pt_BR",
) -> tuple[Optional[Interval], float]:
    """Retorna (intervalo, confiança_da_resolução). None se irresolúvel."""
```

### 5.2 Regras de resolução

Aplicar nesta ordem, primeira que casar vence:

| Padrão | Resultado | Confiança |
|---|---|---|
| Data absoluta ("14 de janeiro de 2023", "em 2019") | intervalo exato ou ano inteiro | 1.00 |
| Deítico de dia ("ontem", "hoje", "anteontem") | `anchor ± n dias`, granularidade dia | 0.95 |
| Deítico de semana ("semana passada", "essa semana") | semana ISO relativa ao anchor | 0.85 |
| Deítico de mês ("mês passado", "em maio") | mês relativo; se só o nome do mês, o mais recente antes do anchor | 0.85 |
| Deítico de ano ("ano passado") | ano relativo | 0.90 |
| Duração retroativa ("há dois anos", "faz três meses") | `[anchor - d, None)` | 0.80 |
| Marcador de início ("comecei", "desde") | `[resolvido, None)` | herda |
| Marcador de fim ("saí", "parei", "até") | fecha em `resolvido` | herda |
| Vago ("recentemente", "faz um tempo") | `None` | 0.0 |
| Ausente | ver 5.3 | - |

### 5.3 Quando não há expressão temporal

Consultar `volatility` da relação:

```
static  -> Interval(None, None)
slow    -> Interval(anchor, None)        # aberto à direita
fast    -> Interval(anchor, anchor + config.fast_window)
```

### 5.4 Quando a resolução falha

`t_valid = None`. A proposição fica no staging com flag `unanchored`. Ela ainda é
consultável, mas apenas pela **ordem parcial do log**: o sistema sabe que o fato
foi afirmado depois do episódio N, mesmo sem saber quando é verdade. O movimento
`restrict` sobre validade não a alcança; o movimento sobre transação alcança.

Poder responder "não sei quando, mas foi depois de X" é uma capacidade, não um
fracasso. Não invente datas.

### 5.5 Granularidade

Todo intervalo carrega granularidade implícita pela sua duração. Um intervalo de
um mês inteiro representa "algum momento em maio", não "todo o mês de maio".
Consequência prática para `intersect`: usar interseção de intervalos é uma
aproximação otimista (assume sobreposição possível). Isso é intencional. A
alternativa pessimista mataria trilhas válidas.

Registrar a granularidade em `Interval.granularity: Literal["day","month","year"]`
para que a geração da resposta possa hedgear ("por volta de maio de 2023").

---

## 6. Pipeline síncrono: ingestão e extração

Executa por turno. Orçamento: uma chamada de LLM. Nunca bloqueia por
consolidação.

### 6.1 Passos

```python
def ingest_turn(text: str, speaker: str, session_id: str, ts: datetime) -> IngestResult:
    # 1. grava o episódio (append-only)
    ep = log.append(Episode(...))

    # 2. monta o contexto de extração (CÓDIGO, não LLM)
    ctx = build_extraction_context(ep)

    # 3. chama o extrator (LLM)
    raw = llm_extract(ctx)

    # 4. valida a saída contra o schema e contra Σ
    props = validate_and_bind(raw, ep)

    # 5. normaliza tempo (CÓDIGO)
    for p in props:
        p.t_valid, tconf = resolve_time(p.time_expression, ep.ts_ingest, sigma[p.relation])

    # 6. atribui confiança por tabela fixa (CÓDIGO)
    for p in props:
        p.confidence = CONFIDENCE_TABLE[p.evidence_kind] * tconf_factor(tconf)

    # 7. grava menções para contagem
    log.append_mentions(extract_mentions(ep, props))

    # 8. escreve tudo no staging
    staging.insert(props)

    return IngestResult(episode=ep, propositions=props,
                        consolidation_debt=staging.pending_count())
```

### 6.2 Construção do contexto (passo 2)

Este é o passo que faz a extração funcionar. Sem ele o extrator produz strings
soltas que não se ligam a nada.

```python
def build_extraction_context(ep: Episode) -> ExtractionContext:
    # a. janela de turnos anteriores para resolver correferência
    prev = log.previous_turns(ep, n=config.coref_window)  # default 3

    # b. candidatos de entidade: busca híbrida sobre nomes já no grafo
    surfaces = extract_noun_phrases(ep.text)   # spaCy ou similar
    candidates = []
    for s in surfaces:
        candidates += entity_index.search_lexical(s, k=3)   # BM25 sobre nome+aliases
        candidates += entity_index.search_dense(s, k=3)     # embedding
    candidates = dedupe_and_rank(candidates)[:config.max_candidates]  # default 20

    # c. catálogo filtrado pelos tipos dos candidatos
    types_present = {c.type for c in candidates} | {"Person"}  # falante sempre
    relations = sigma.filter_by_types(types_present)

    return ExtractionContext(ep, prev, candidates, relations)
```

### 6.3 Prompt do extrator

Template fixo. Variáveis entre `{}`.

```
Você extrai afirmações estruturadas de um turno de diálogo.

DATA DO TURNO: {ep.ts_ingest:%Y-%m-%d}
FALANTE: {ep.speaker} ({speaker_entity_id})

TURNOS ANTERIORES (apenas para resolver pronomes, não extraia deles):
{prev_turns}

TURNO ATUAL:
"{ep.text}"

ENTIDADES CONHECIDAS. Use o id. Se a entidade não estiver na lista, use
"new:<nome canônico>":
{candidates_table}

RELAÇÕES PERMITIDAS. Não invente rótulos. Se nada servir, use "UNMAPPED" e
preencha suggested_relation:
{relations_table}

REGRAS OBRIGATÓRIAS:
1. NÃO calcule datas. Copie o TRECHO LITERAL que indica tempo em
   time_expression, ou null. Exemplos: "mês passado", "em 2019", "ontem".
2. Classifique a operação:
   - assert   : afirma algo novo
   - reassert : repete algo já sabido, sem mudança
   - close    : o fato deixou de valer ("saí da empresa", "não moro mais lá")
   - retract  : o fato nunca foi verdade ("na verdade não", "eu tinha me
                enganado", "nunca foi")
3. Classifique a evidência:
   - literal      : o texto diz explicitamente
   - coreference  : literal, mas você resolveu um pronome
   - implicature  : implicado por "também", "de novo", "ainda", "voltei a"
   - contextual   : inferido do contexto, não dito
4. polarity = false apenas para negação explícita ("não gosto de X").
5. span deve ser um trecho VERBATIM do turno atual.
6. Se não houver nada a extrair, devolva [].

Devolva APENAS um array JSON, sem markdown, no schema:
[{
  "operation": "assert|reassert|close|retract",
  "subject_id": "ent_xxx | new:Nome",
  "relation": "nome_da_relacao | UNMAPPED",
  "suggested_relation": "string | null",
  "object_id": "ent_xxx | new:Nome",
  "polarity": true,
  "time_expression": "string | null",
  "evidence_kind": "literal|coreference|implicature|contextual",
  "span": "trecho verbatim"
}]
```

### 6.4 Tabela de confiança (passo 6)

```python
CONFIDENCE_TABLE = {
    EvidenceKind.LITERAL:     0.90,
    EvidenceKind.COREFERENCE: 0.80,
    EvidenceKind.IMPLICATURE: 0.55,
    EvidenceKind.CONTEXTUAL:  0.40,
}

def tconf_factor(tconf: float) -> float:
    """Resolução temporal ruim reduz a confiança da proposição."""
    return 1.0 if tconf >= 0.85 else 0.9 if tconf > 0 else 0.85
```

**Nunca peça um número de confiança ao LLM.** Ele produz valores mal calibrados e
não reprodutíveis. Peça a classificação linguística, que ele faz bem, e converta
por tabela. Assim o limiar τ é auditável e você consegue explicar por que uma
proposição foi promovida.

### 6.5 Validação (passo 4)

Rejeitar e logar, nunca aceitar silenciosamente:

- `relation` fora de Σ e diferente de `UNMAPPED` → rejeita.
- Tipos de `subject`/`object` violam `signature` → rejeita.
- `span` não é substring literal de `ep.text` (após normalização de espaço) →
  rebaixa `evidence_kind` para `contextual`.
- `operation` em `{close, retract}` sem aresta alvo existente → vai para staging
  como pendente órfã, reprocessada na próxima consolidação.

---

## 7. Pipeline assíncrono: consolidação

Executa por gatilho: fim de sessão, `staging.pending_count() > N`, ou ociosidade.
Idempotente e retomável. Interromper no meio não corrompe estado.

### 7.1 Ordem das fases

A ordem importa. Não reordenar.

```python
def consolidate(scope: ConsolidationScope) -> ConsolidationReport:
    props = staging.pending(scope)

    phase_1_resolve_entities(props)      # new:Nome -> ent_id
    phase_2_address(props)               # (subject, relation) -> endereço
    phase_3_apply_operations(props)      # assert/reassert/close/retract
    phase_4_cardinality(touched)         # fecha valores anteriores
    phase_5_propagate_dependents(closed) # ponto fixo sobre Σ.dependents
    phase_6_fold(touched)                # dobra + jornal
    phase_7_promote_staged()             # acúmulo de evidência
    phase_8_detect_conflicts(touched)
    phase_9_summarize(scope)             # opcional, LLM

    return report
```

### 7.2 Fase 1: resolução de entidades novas

`new:Salvador` vira um vértice. Antes de criar, tentar casar novamente com o
índice (o grafo pode ter mudado desde a extração). Se casar acima de
`config.tau_entity`, reusar. Senão criar com `provisional=True`.

Vértices provisórios são candidatos preferenciais a dobra na fase 6.

### 7.3 Fase 2: endereçamento

**O endereço de escrita é o par `(subject_id, relation)`.** Não é busca
semântica, não é decisão do LLM, é uma chave. Isso é o que torna a atualização
determinística.

```python
def address(p: Proposition) -> EdgeAddress:
    return EdgeAddress(src=p.subject_id, label=p.relation)
```

`graph.edges_at(address)` devolve todas as arestas naquele endereço,
independente de intervalo.

### 7.4 Fase 3: aplicação das operações

```python
def apply(p: Proposition) -> list[Edge]:
    addr = address(p)
    existing = graph.edges_at(addr)

    if p.operation == Operation.ASSERT:
        if p.confidence < config.tau_promote:
            staging.keep(p)                     # fica no staging
            return []
        return [graph.create_edge(
            src=p.subject_id, label=p.relation, dst=p.object_id,
            t_valid=p.t_valid, t_tx=Interval(p.t_tx.start, None),
            provenance=[p.id], confidence=p.confidence)]

    if p.operation == Operation.REASSERT:
        e = find_live_edge(existing, dst=p.object_id, at=p.t_tx.start)
        if e is None:
            p.operation = Operation.ASSERT
            return apply(p)
        e.reinforcement += 1
        e.last_confirmed = p.t_tx.start
        e.provenance.append(p.id)
        e.confidence = max(e.confidence, p.confidence)
        return [e]

    if p.operation == Operation.CLOSE:
        e = find_live_edge(existing, dst=p.object_id, at=p.t_tx.start)
        if e is None:
            staging.orphan(p); return []
        close_at = p.t_valid.start if p.t_valid else p.t_tx.start
        e.t_valid = Interval(e.t_valid.start, close_at)
        mark_closed(e, cause="explicit", by=p.id)
        return [e]

    if p.operation == Operation.RETRACT:
        e = find_edge_any(existing, dst=p.object_id)
        if e is None:
            staging.orphan(p); return []
        e.t_tx = Interval(e.t_tx.start, p.t_tx.start)   # FECHA TRANSAÇÃO
        mark_retracted(e, by=p.id)
        return [e]
```

A distinção `CLOSE` vs `RETRACT` é o ponto mais importante desta fase.
`CLOSE` mexe em `t_valid`, `RETRACT` mexe em `t_tx`. Campos diferentes, efeitos
observáveis diferentes, zero ambiguidade.

### 7.5 Fase 4: cardinalidade

```python
def phase_4_cardinality(touched: set[EdgeAddress]):
    for addr in touched:
        spec = sigma[addr.label]
        if spec.cardinality != "functional" or not spec.closes_on_new:
            continue
        edges = sorted(graph.live_edges_at(addr), key=lambda e: e.t_valid.start)
        for a, b in zip(edges, edges[1:]):
            if a.dst_id == b.dst_id:
                continue                       # mesmo valor, não é conflito
            if a.t_valid.overlaps(b.t_valid):
                if a.t_valid.start < b.t_valid.start:
                    a.t_valid = Interval(a.t_valid.start, b.t_valid.start)
                    mark_closed(a, cause="functional_supersede", by=b.id)
                else:
                    b.conflict_flag = True     # início idêntico, indecidível
                    a.conflict_flag = True
```

**A data de fechamento vem da validade do fato novo, não da data do episódio.**
Se em junho a pessoa diz que se mudou em maio, a mudança fecha em maio. Usar a
data do episódio faria "onde ela morava em maio" responder errado. Este é o
motivo pelo qual a ancoragem temporal síncrona é obrigatória: ela alimenta uma
decisão tomada depois.

### 7.6 Fase 5: propagação de dependentes

```python
def phase_5_propagate_dependents(closed: list[Edge]):
    queue = deque(closed)
    seen = set()
    while queue:
        e = queue.popleft()
        if e.id in seen: continue
        seen.add(e.id)
        for dep_label in sigma[e.label].dependents:
            for d in graph.live_edges_at(EdgeAddress(e.src_id, dep_label)):
                if d.t_valid.end is None or d.t_valid.end > e.t_valid.end:
                    d.t_valid = Interval(d.t_valid.start, e.t_valid.end)
                    mark_closed(d, cause=f"dependent_of:{e.label}", by=e.id)
                    queue.append(d)
```

Ninguém "sabe" que trocar de emprego encerra a relação de gerência. Está escrito
em uma linha do catálogo (`works_at.dependents: [managed_by]`) e o
comportamento é reprodutível. Σ.dependents deve formar um DAG; validar na
inicialização.

### 7.7 Fase 7: promoção por acúmulo

Proposições no staging abaixo de `tau_promote` esperam confirmação independente.

```python
def phase_7_promote_staged():
    for group in staging.group_by(lambda p: (p.subject_id, p.relation, p.object_id)):
        if len(group) < 2: continue
        if len({p.episode_id for p in group}) < 2: continue   # independência
        combined = combine_confidence([p.confidence for p in group])
        if combined >= config.tau_promote:
            earliest = min(p.t_valid.start for p in group if p.t_valid)
            graph.create_edge(..., t_valid=Interval(earliest, None),
                              provenance=[p.id for p in group],
                              confidence=combined)
            staging.mark_promoted(group)

def combine_confidence(cs: list[float]) -> float:
    """Noisy-OR. Duas evidências fracas independentes valem mais que uma."""
    prod = 1.0
    for c in cs: prod *= (1 - c)
    return 1 - prod
```

O intervalo herda o **início da evidência mais antiga**. Uma implicatura fraca de
março pode ficar dormente meses no staging e ser confirmada em novembro; quando
promovida, a validade começa em março. Nenhuma decisão síncrona conseguiria fazer
isso.

### 7.8 Fase 8: detecção de conflitos

Marcar, não resolver, quando a resolução for indecidível:

| Situação | Ação |
|---|---|
| Funcional, dois valores, intervalos sobrepostos, mesmo início | `conflict_flag = True` em ambas |
| Polaridade oposta no mesmo endereço e intervalo | `conflict_flag = True` |
| Dobra forçaria fundir vértices com atributos incompatíveis | aborta a dobra, marca |

Arestas com `conflict_flag` continuam vivas e são retornadas pelos movimentos com
o flag exposto. O agente decide o que fazer, e pode perguntar ao usuário.

---

## 8. Dobra e resolução de entidades

Resolução de entidades não é um módulo separado. É consequência da dobra.

### 8.1 Condição de dobra

Duas arestas `e1 = (u, r, v1, I1)` e `e2 = (u, r, v2, I2)` podem ser dobradas se
**todas** valerem:

| # | Condição | Verificação |
|---|---|---|
| C1 | Mesma origem e mesmo rótulo | `e1.src == e2.src and e1.label == e2.label` |
| C2 | Tipos dos destinos compatíveis | `type(v1) == type(v2)` e ambos em `signature[1]` |
| C3 | Intervalos compatíveis | `I1.overlaps(I2)` ou adjacentes dentro de `config.fold_slack` |
| C4 | Confiança de identidade acima do limiar | `identity_score(v1, v2) >= config.tau_fold` |

C4 é o único componente não trivial.

### 8.2 Escore de identidade

```python
def identity_score(a: Entity, b: Entity) -> float:
    if a.type != b.type: return 0.0
    s = 0.0
    s += W_NAME      * name_similarity(a, b)          # Jaro-Winkler + alias exato
    s += W_CONTAINED * contains_as_token(a, b)        # "Rui" ⊂ "Rui Sampaio"
    s += W_STRUCT    * neighbor_overlap(a, b)         # Jaccard das vizinhanças
    s += W_ROLE      * same_role_context(a, b)        # mesmo (src, label) de entrada
    s += W_TEMPORAL  * temporal_compatibility(a, b)   # intervalos não conflitam
    s -= P_DISTINCT  * explicit_distinction(a, b)     # "o outro Rui", "não aquele"
    return clamp(s, 0.0, 1.0)

# defaults; expor em config
W_NAME, W_CONTAINED, W_STRUCT, W_ROLE, W_TEMPORAL = 0.35, 0.20, 0.20, 0.15, 0.10
P_DISTINCT = 0.9
```

`explicit_distinction` procura no log marcadores de desambiguação explícita. Um
único "o outro Rui" deve ser suficiente para bloquear a dobra.

### 8.3 Algoritmo

```python
def fold(scope: set[str]) -> list[FoldRecord]:
    uf = UnionFind(graph.vertices())
    queue = PriorityQueue()          # maior escore primeiro

    for addr in scope:
        edges = graph.edges_at(addr)
        for e1, e2 in combinations(edges, 2):
            if not (C1(e1,e2) and C2(e1,e2) and C3(e1,e2)): continue
            sc = identity_score(vertex(e1.dst), vertex(e2.dst))
            if sc >= config.tau_fold:
                queue.push(sc, (e1, e2))

    records = []
    while queue:
        sc, (e1, e2) = queue.pop()
        v1, v2 = uf.find(e1.dst), uf.find(e2.dst)
        if v1 == v2: continue
        if not attributes_compatible(v1, v2):
            mark_conflict(v1, v2); continue

        rec = FoldRecord(
            id=next_fold_id(), kept=v1, absorbed=v2, score=sc,
            trigger=current_scope_episode(),
            migrated_edges=graph.edges_incident(v2),
            snapshot=snapshot_vertex(v2))
        journal.append(rec)

        uf.union(v1, v2)
        graph.migrate_edges(v2, v1)
        graph.mark_alias(v2, merged_into=v1)
        records.append(rec)

        # a fusão de destinos cria novos pares dobráveis: reenfileirar
        for addr in graph.addresses_touching(v1):
            enqueue_candidates(addr, queue)

    return records
```

A dobra é iterada até ponto fixo. Fundir dois destinos cria novos pares
candidatos, porque tudo que estava pendurado em cada vértice migra para o
vértice fundido. É isso que cria caminhos nunca observados: depois de fundir
`Rui` e `Rui Sampaio`, a aresta `hired` passa a sair do mesmo nó que
`managed_by`, e a pergunta "quem me contratou era meu chefe?" passa a ter
resposta.

### 8.4 Jornal e reversão

```python
@dataclass
class FoldRecord:
    id: str
    kept: str
    absorbed: str
    score: float
    trigger: str                 # episode id
    migrated_edges: list[str]
    snapshot: dict               # estado completo do vértice absorvido
    reverted: bool = False

def unfold(fold_id: str) -> None:
    """Reverte uma fusão. Reverte também fusões posteriores que dependam dela."""
    rec = journal.get(fold_id)
    dependents = journal.folds_after(fold_id, touching=rec.kept)
    for d in reversed(dependents):
        _revert_single(d)
    _revert_single(rec)
    reprocess_from_log(scope=affected_addresses(rec))
```

Gatilhos de reversão:

1. Marcador explícito de distinção em episódio posterior.
2. Contradições em cascata na fase 8 acima de `config.max_conflicts_per_vertex`.
3. Comando manual de operador.

Sem o jornal, um "Rui" errado contamina permanentemente uma região do grafo,
porque dobra é congruência e propaga.

---

## 9. A álgebra de acesso

Não existe roteador. A memória não expõe intenções, expõe movimentos. O agente
compõe.

### 9.1 Estado

```python
@dataclass(frozen=True)
class Trail:
    vertex_id: str
    window: Interval           # interseção de TODOS os intervalos do caminho
    path: tuple[str, ...]      # proposition ids percorridos
    labels: tuple[str, ...]    # rótulos percorridos, para explicação

@dataclass
class AccessState:
    trails: list[Trail]
    evidence: list[str]                # episode ids materializados
    tx_point: datetime                 # ponto de vista de transação; default now
    dead_count: int = 0
    death_cause: Optional[str] = None
    budget_used: int = 0
```

**Invariante I1.** Para toda `Trail`, `window == intersect(I_e for e in path)` e
`window is not None`.

Trilhas cuja janela esvaziaria não são criadas. A coerência temporal de caminho é
uma propriedade da estrutura, não uma checagem que alguém pode esquecer.

### 9.2 Os oito movimentos

| Movimento | Assinatura | Semântica |
|---|---|---|
| `anchor(text)` | `→ State` | Ponto de entrada. Busca híbrida sobre entidades e episódios. |
| `follow(label)` | `State → State` | Aplica a letra, estreita a janela, poda vazias. |
| `restrict(axis, interval)` | `State → State` | `axis ∈ {valid, tx}`. Estreita ou muda o ponto de vista. |
| `filter(predicate)` | `State → State` | Filtra trilhas por vértice, tipo ou atributo. |
| `expand(k)` | `State → State` | Ativação por espalhamento a k passos. Para quando o rótulo não é nomeável. |
| `history(label)` | `State → Series` | Todas as arestas do rótulo ao longo do tempo, sem colapso funcional. |
| `evidence()` | `State → Episodes` | Materializa o texto bruto da proveniência. |
| `count(predicate)` | `→ int` | Vai ao log. Preserva multiplicidade. Ignora o grafo. |

### 9.3 Implementação de `follow`

O núcleo do sistema.

```python
def follow(state: AccessState, label: str) -> AccessState:
    if label not in sigma and not is_inverse(label):
        raise UnknownLabel(label)

    new_trails, dead, cause = [], 0, None
    for t in state.trails:
        edges = graph.out_edges(t.vertex_id, label)
        matched = False
        for e in edges:
            if not e.t_tx.contains(state.tx_point):
                continue                                  # retratada nesta visão
            w = t.window.intersect(e.t_valid)
            if w is None:
                continue                                  # incoerência temporal
            new_trails.append(Trail(
                vertex_id=e.dst_id, window=w,
                path=t.path + tuple(e.provenance),
                labels=t.labels + (label,)))
            matched = True
        if not matched:
            dead += 1
            cause = classify_death(t, edges, state.tx_point)

    return AccessState(trails=new_trails, evidence=state.evidence,
                       tx_point=state.tx_point,
                       dead_count=dead, death_cause=cause,
                       budget_used=state.budget_used + 1)


def classify_death(t, edges, tx_point) -> str:
    if not edges:
        return "no_edge_with_label"
    if all(not e.t_tx.contains(tx_point) for e in edges):
        return "all_edges_retracted"
    if all(t.window.intersect(e.t_valid) is None for e in edges):
        return "empty_temporal_window"
    return "unknown"
```

`death_cause` é o que permite ao agente reagir de forma inteligente: alargar a
janela, mudar o ponto de vista de transação, tentar `expand`, ou concluir que a
premissa da pergunta é falsa.

### 9.4 Três propriedades que caem de graça

**Composição.** `follow(r1) ∘ follow(r2) == follow(r1·r2)` sobre o esqueleto
atemporal, módulo estreitamento monótono da janela. Isso permite otimizar
sequências antes de executar, desde que a otimização preserve a interseção.

**Redução livre.** `follow(r) ∘ follow(r⁻¹)` retorna ao vértice de origem com
janela possivelmente estreitada. Útil para consultas do tipo "quem mais mora onde
ela mora".

**Cardinalidade funcional resolve single-hop.** Se a relação é funcional, `follow`
depois de `restrict(valid, [t,t])` produz no máximo uma trilha. A resposta é
única por construção, sem desempate por recência de embedding.

### 9.5 `expand`, o motor associativo

Necessário quando a pergunta não nomeia a relação.

```python
def expand(state: AccessState, k: int = 2) -> AccessState:
    seeds = {t.vertex_id: 1.0 / len(state.trails) for t in state.trails}
    scores = personalized_pagerank(
        graph.restrict_to_live(state.tx_point), seeds,
        alpha=config.ppr_alpha, max_hops=k)
    top = topk(scores, config.expand_k)
    # trilhas de expansão têm janela herdada da semente e path parcial
    return AccessState(trails=[Trail(v, seed_window_for(v, state), ...) for v in top], ...)
```

Regra de uso: `expand` escolhe pontos de entrada, `follow` verifica caminhos.
Nunca responda apenas com `expand`, porque ele não garante coerência temporal.

---

## 10. Interface com o agente

Os movimentos são expostos como ferramentas. O retorno é compacto para não
estourar contexto.

### 10.1 Schema das ferramentas

```json
[
 {"name": "memory_anchor",
  "description": "Encontra pontos de entrada na memória a partir de um texto.",
  "input_schema": {"type":"object","properties":{
    "text":{"type":"string"}},"required":["text"]}},

 {"name": "memory_follow",
  "description": "Segue uma relação a partir do estado atual. Estreita a janela temporal.",
  "input_schema": {"type":"object","properties":{
    "label":{"type":"string"}},"required":["label"]}},

 {"name": "memory_restrict",
  "description": "Restringe o estado a um período. axis=valid para quando o fato era verdade; axis=tx para o que o agente acreditava naquela data.",
  "input_schema": {"type":"object","properties":{
    "axis":{"enum":["valid","tx"]},
    "start":{"type":"string","format":"date"},
    "end":{"type":"string","format":"date"}},"required":["axis"]}},

 {"name": "memory_filter",
  "description": "Mantém apenas trilhas cujo vértice casa com o nome ou tipo dado.",
  "input_schema": {"type":"object","properties":{
    "name":{"type":"string"},"type":{"type":"string"}}}},

 {"name": "memory_expand",
  "description": "Expande por associação quando o nome da relação é desconhecido.",
  "input_schema": {"type":"object","properties":{
    "hops":{"type":"integer","default":2}}}},

 {"name": "memory_history",
  "description": "Devolve toda a série temporal de uma relação, sem colapso.",
  "input_schema": {"type":"object","properties":{
    "label":{"type":"string"}},"required":["label"]}},

 {"name": "memory_evidence",
  "description": "Devolve o texto original dos episódios que sustentam as trilhas vivas.",
  "input_schema": {"type":"object","properties":{}}},

 {"name": "memory_count",
  "description": "Conta ocorrências no log. Use para perguntas de quantidade e frequência.",
  "input_schema": {"type":"object","properties":{
    "entity":{"type":"string"},"topic":{"type":"string"},
    "start":{"type":"string"},"end":{"type":"string"}}}}
]
```

### 10.2 Formato do retorno

```json
{
  "live_trails": 1,
  "sample": [
    {"vertex": "Rui Sampaio", "type": "Person",
     "window": "2023-09-05..", "hops": 2,
     "labels": ["works_at", "managed_by"],
     "conflict": false}
  ],
  "dead_trails": 1,
  "death_cause": "empty_temporal_window",
  "available_labels": ["works_at", "lives_in", "practices", "managed_by"],
  "consolidation_debt": 0,
  "budget_used": 3,
  "budget_left": 5
}
```

`available_labels` é o campo que dispensa qualquer classificação prévia da
pergunta. O agente vê o que pode fazer a partir de onde está.

### 10.3 Laço do agente

```
1. o agente lê a pergunta
2. chama memory_anchor
3. lê available_labels e death_cause, escolhe o próximo movimento
4. repete até:
   - trilhas vivas com evidência suficiente, ou
   - todas as trilhas mortas com causa identificada, ou
   - orçamento esgotado (config.movement_budget, default 8)
5. chama memory_evidence
6. redige a resposta a partir dos episódios
```

Empiricamente, 2 movimentos bastam para fato corrente, 4 a 6 para composição
temporal. Orçamento 8 é folgado.

---

## 11. Geração da resposta

Prompt final recebe três blocos, nesta ordem de prioridade:

1. **Episódios de evidência**, texto bruto, com data.
2. **Fatos estruturados** das trilhas vivas, com janela e flag de conflito.
3. **Diagnóstico**, quando não há trilha viva: causa da morte e em que passo.

Regras de redação a instruir no prompt:

- Priorize o texto do episódio sobre o fato estruturado quando divergirem. O
  fato serviu para localizar o episódio; o episódio é a fonte.
- Se `death_cause == "empty_temporal_window"`, a premissa da pergunta é falsa.
  Diga isso explicitamente, com o período que causou o conflito.
- Se `death_cause == "all_edges_retracted"`, a informação foi retratada. Diga que
  não há registro válido, e ofereça o que se acreditava antes se for útil.
- Se `conflict == true`, apresente ambas as versões com suas datas.
- Se a granularidade do intervalo for mês ou ano, hedgeie ("por volta de maio").
- Se não houver evidência, diga que não sabe. Nunca preencha.

---

## 12. Persistência e índices

### 12.1 Esquema relacional (SQLite ou Postgres)

```sql
CREATE TABLE episodes (
  id TEXT PRIMARY KEY, session_id TEXT NOT NULL, speaker TEXT NOT NULL,
  text TEXT NOT NULL, ts_ingest TIMESTAMP NOT NULL, seq INTEGER NOT NULL,
  meta JSON);
CREATE INDEX idx_ep_session ON episodes(session_id, seq);
CREATE INDEX idx_ep_ts ON episodes(ts_ingest);

CREATE TABLE entities (
  id TEXT PRIMARY KEY, canonical_name TEXT NOT NULL, type TEXT NOT NULL,
  aliases JSON, created_from TEXT, merged_into TEXT, provisional BOOLEAN);
CREATE INDEX idx_ent_name ON entities(canonical_name);
CREATE INDEX idx_ent_merged ON entities(merged_into);

CREATE TABLE propositions (
  id TEXT PRIMARY KEY, subject_id TEXT, relation TEXT, object_id TEXT,
  operation TEXT, polarity BOOLEAN, time_expression TEXT,
  t_valid_start TIMESTAMP, t_valid_end TIMESTAMP, t_valid_gran TEXT,
  t_tx_start TIMESTAMP, t_tx_end TIMESTAMP,
  evidence_kind TEXT, confidence REAL, span TEXT,
  episode_id TEXT, status TEXT);
CREATE INDEX idx_prop_addr ON propositions(subject_id, relation);
CREATE INDEX idx_prop_status ON propositions(status);

CREATE TABLE edges (
  id TEXT PRIMARY KEY, src_id TEXT, label TEXT, dst_id TEXT,
  t_valid_start TIMESTAMP, t_valid_end TIMESTAMP, t_valid_gran TEXT,
  t_tx_start TIMESTAMP, t_tx_end TIMESTAMP,
  provenance JSON, reinforcement INTEGER, last_confirmed TIMESTAMP,
  confidence REAL, conflict_flag BOOLEAN);
CREATE INDEX idx_edge_addr ON edges(src_id, label);
CREATE INDEX idx_edge_dst ON edges(dst_id, label);
CREATE INDEX idx_edge_valid ON edges(t_valid_start, t_valid_end);

CREATE TABLE mentions (
  id TEXT PRIMARY KEY, episode_id TEXT, entity_id TEXT,
  surface TEXT, ts TIMESTAMP);
CREATE INDEX idx_mention_ent ON mentions(entity_id, ts);

CREATE TABLE fold_journal (
  id TEXT PRIMARY KEY, kept TEXT, absorbed TEXT, score REAL,
  trigger_episode TEXT, migrated_edges JSON, snapshot JSON,
  reverted BOOLEAN, created_at TIMESTAMP);

CREATE TABLE unmapped_queue (
  id TEXT PRIMARY KEY, suggested_relation TEXT, subject_surface TEXT,
  object_surface TEXT, span TEXT, episode_id TEXT, count INTEGER);
```

### 12.2 Índices de busca

- **Lexical**: BM25 sobre `entities.canonical_name + aliases` e sobre
  `episodes.text`. SQLite FTS5 basta.
- **Denso**: embeddings de nomes de entidade e de episódios. Qualquer store
  vetorial; o volume é pequeno.
- **Grafo em memória**: listas de adjacência carregadas na inicialização. O grafo
  consolidado de um usuário cabe em poucos megabytes, porque a dobra impede
  crescimento linear no número de turnos. `follow` nunca toca disco.
- **Disco**: apenas `evidence()` e `count()`.

### 12.3 Reconstrução

```
clio.rebuild --from-log --scope all
```

Descarta `entities`, `propositions`, `edges`, `fold_journal` e reprocessa o log
inteiro. Deve ser determinístico dado o mesmo catálogo e a mesma seed do
extrator. Este comando é o que torna P1 operacional, e é um teste em si.

---

## 13. Complexidade e orçamentos

| Operação | Custo | Onde |
|---|---|---|
| Ingestão por turno | 1 chamada de LLM | bloqueante, ~1s |
| Normalização temporal | O(1) | síncrono |
| Endereçamento | O(1) hash | assíncrono |
| Cardinalidade | O(d log d), d = arestas no endereço | assíncrono |
| Propagação de dependentes | O(V·D), D = profundidade do DAG de Σ | assíncrono |
| Dobra | quase linear com union-find | assíncrono |
| `follow` | O(grau do vértice) | memória |
| `expand` | O(iterações · arestas) do PPR | memória |
| `evidence` | 1 leitura em disco | fim da consulta |
| `count` | O(log n + k) com índice | disco |

O gargalo real é a extração, em custo e em qualidade. Todo o resto é
infraestrutura para conter os erros dela.

---

## 14. Configuração

```yaml
clio:
  catalog_path: catalog/personal_dialogue.yaml

  extraction:
    model: <modelo do projeto>
    coref_window: 3
    max_candidates: 20
    temperature: 0.0

  temporal:
    fast_window_days: 1
    locale: pt_BR

  thresholds:
    tau_promote: 0.70      # confiança mínima para ir ao grafo
    tau_fold: 0.80         # confiança mínima de identidade para dobrar
    tau_entity: 0.85       # reuso de entidade na fase 1
    fold_slack_days: 7     # tolerância de adjacência de intervalos

  consolidation:
    trigger_on_session_end: true
    trigger_pending_count: 20
    trigger_idle_seconds: 120
    max_conflicts_per_vertex: 3

  access:
    movement_budget: 8
    expand_k: 10
    ppr_alpha: 0.15
    max_trails: 50         # poda por escore quando estourar

  retrieval:
    hybrid_weights: {lexical: 0.4, dense: 0.6}
```

Todos os limiares devem ser expostos. Eles são o objeto das ablações.

---

## 15. Estrutura de arquivos

```
clio/
├── __init__.py
├── config.py
├── types.py                  # Interval, Episode, Entity, Proposition, Edge, Mention
├── catalog/
│   ├── __init__.py
│   ├── loader.py             # carrega e valida Σ, checa DAG de dependentes
│   ├── spec.py               # RelationSpec
│   └── personal_dialogue.yaml
├── log/
│   ├── store.py              # append-only, ordem total
│   └── mentions.py
├── ingest/
│   ├── pipeline.py           # ingest_turn
│   ├── context.py            # build_extraction_context
│   ├── extractor.py          # prompt, chamada, parsing
│   ├── validate.py           # validação contra schema e Σ
│   └── prompts/extract.txt
├── temporal/
│   ├── resolver.py           # resolve_time
│   ├── intervals.py          # álgebra de intervalos, Allen
│   └── patterns_pt_br.py
├── consolidate/
│   ├── pipeline.py           # as 9 fases
│   ├── entities.py           # fase 1
│   ├── operations.py         # fase 3
│   ├── cardinality.py        # fase 4
│   ├── dependents.py         # fase 5
│   ├── fold.py               # fase 6, union-find
│   ├── journal.py            # FoldRecord, unfold
│   ├── promote.py            # fase 7
│   └── conflicts.py          # fase 8
├── graph/
│   ├── store.py              # persistência
│   ├── adjacency.py          # grafo em memória
│   └── queries.py            # edges_at, live_edges_at, out_edges
├── access/
│   ├── state.py              # Trail, AccessState
│   ├── movements.py          # os oito movimentos
│   ├── ppr.py                # expand
│   └── tools.py              # schemas para o agente
├── answer/
│   ├── generator.py
│   └── prompts/answer.txt
├── index/
│   ├── lexical.py
│   └── dense.py
└── cli.py                    # ingest, consolidate, query, rebuild, inspect
```

---

## 16. Plano de implementação

Ordem sugerida. Cada marco é testável isoladamente.

**M1. Fundação.** `types.py`, álgebra de `Interval` com testes exaustivos, log
append-only, esquema do banco. Sem LLM.

**M2. Catálogo.** Loader, validação de DAG, catálogo inicial. Sem LLM.

**M3. Temporal.** `resolve_time` completo com a tabela da Seção 5.2 e suite de
testes com pelo menos 40 expressões em pt-BR. Sem LLM. **Não avance sem este
módulo sólido**, porque erros aqui contaminam tudo silenciosamente.

**M4. Consolidação sem dobra.** Fases 1 a 5 e 7 a 8. Alimentar com proposições
escritas à mão. Verificar fechamento funcional e propagação de dependentes com o
corpus da Seção 17.2. Sem LLM.

**M5. Extração.** Contexto, prompt, validação. Primeiro ponto onde o LLM entra.
Medir precisão e cobertura contra proposições anotadas à mão.

**M6. Dobra.** Fase 6, union-find, jornal, `unfold`. Testar com o caso
`Rui`/`Rui Sampaio` e com o caso negativo `o outro Rui`.

**M7. Álgebra de acesso.** Os oito movimentos sobre o grafo já consolidado.
Testar com os traços da Seção 17.3, sem agente.

**M8. Integração com o agente.** Schemas de ferramenta, laço, geração de resposta.

**M9. Avaliação.** LoCoMo, invariância de ordem, ablações.

---

## 17. Testes

### 17.1 Unitários obrigatórios

- `Interval.intersect`: bordas coincidentes, semiabertura, `None` em cada lado,
  interseção vazia por adjacência exata.
- `resolve_time`: cada linha da tabela 5.2, mais casos de fuso e virada de ano.
- Cardinalidade: valor novo fecha anterior na data de **validade**, não na do
  episódio.
- Dependentes: fechamento em cascata com profundidade 2.
- `CLOSE` mexe só em `t_valid`; `RETRACT` mexe só em `t_tx`. Teste explícito de
  que o outro campo não mudou.
- `follow`: janela resultante é a interseção; trilha morre por janela vazia;
  trilha morre por transação encerrada; `death_cause` correto em cada caso.
- Dobra: as quatro condições, cada uma isoladamente bloqueando.
- `unfold`: reverte fusões dependentes na ordem correta.
- `count`: retorna multiplicidade, não cardinalidade do grafo.

### 17.2 Corpus de integração

Fixture obrigatória, `tests/fixtures/melanie.yaml`. Sete episódios que exercitam
todos os mecanismos.

| Id | Data | Texto |
|---|---|---|
| E1 | 2023-01-14 | "Comecei na Vertex essa semana, estou morando em Recife" |
| E2 | 2023-03-02 | "Minha gerente aqui é a Bia, ela também curte escalada" |
| E3 | 2023-06-20 | "Me mudei pra Salvador mês passado, sigo na Vertex remoto" |
| E4 | 2023-09-05 | "Saí da Vertex, entrei na Kaia. Meu chefe agora é o Rui" |
| E5 | 2023-11-11 | "Fui escalar de novo no fim de semana" |
| E6 | 2023-12-01 | "Na verdade a Bia nunca foi minha gerente, ela era de outra equipe" |
| E7 | 2024-01-20 | "O Rui, meu chefe, o Rui Sampaio, foi quem me contratou na Kaia" |

Estado esperado após consolidação completa:

```
Melanie -works_at->     Vertex    valid[2023-01-08, 2023-09-05)  tx[E1, ∞)
Melanie -works_at->     Kaia      valid[2023-09-05, ∞)           tx[E4, ∞)
Melanie -lives_in->     Recife    valid[2023-01-14, 2023-05-01)  tx[E1, ∞)
Melanie -lives_in->     Salvador  valid[2023-05-01, ∞)           tx[E3, ∞)
Melanie -managed_by->   Bia       valid[2023-03-02, 2023-09-05)  tx[E2, E6)
Melanie -managed_by->   Rui       valid[2023-09-05, ∞)           tx[E4, ∞)
Melanie -practices->    escalada  valid[2023-03-02, ∞)           tx[E5, ∞)
Rui     -hired->        Melanie   valid[2023-09-05, ∞)           tx[E7, ∞)
Bia     -practices->    escalada  valid[2023-03-02, ∞)           tx[E2, ∞)

entities: "Rui Sampaio" merged_into "Rui" (fold #1)
mentions: escalada = 2 (E2, E5)
```

Cinco asserções que este corpus verifica e que nenhum outro teste verifica:

1. `lives_in Recife` fecha em **maio** (validade de E3), não em junho (episódio).
2. `managed_by Bia` fecha `t_valid` em setembro **por dependência** de
   `works_at`, sem menção explícita.
3. `managed_by Bia` fecha `t_tx` em E6 sem alterar `t_valid`.
4. `practices escalada` começa em **março** apesar de ter sido promovida em
   novembro, herdando o início da implicatura de E2.
5. `Rui Sampaio` funde com `Rui` e `hired` migra para o vértice fundido.

### 17.3 Traços de acesso esperados

Testar a álgebra sem agente, chamando os movimentos diretamente.

**T1. Fato corrente.** "Onde a Melanie trabalha?"
```
anchor("Melanie") ; restrict(valid, [hoje,hoje]) ; follow("works_at")
→ 1 trilha, vertex=Kaia, evidence=[E4]
```

**T2. Fato passado.** "Onde ela trabalhava em fevereiro de 2023?"
```
anchor("Melanie") ; restrict(valid, 2023-02) ; follow("works_at")
→ 1 trilha, vertex=Vertex
```
Mesma composição de T1, argumento diferente. Não existe motor temporal separado.

**T3. Multi-hop com retratação.** "Quem era o chefe dela quando morava em Recife?"
```
anchor("Melanie") ; follow("lives_in") ; filter(name="Recife")
                  ; follow("lives_in⁻¹") ; follow("managed_by")
→ 0 trilhas, death_cause="all_edges_retracted"
```
A aresta da Bia tem validade compatível, mas transação encerrada. Sem o eixo de
transação, o sistema responderia "Bia" com alta confiança.

**T4. Crença histórica.** Continuando de T3:
```
restrict(tx, 2023-11-01) ; follow("managed_by")
→ 1 trilha, vertex=Bia, window=[2023-03-02, 2023-05-01)
```
A janela é a **interseção** de `[mar-02, set-05)` com `[jan-14, mai-01)`, não
qualquer uma das duas. Este é o invariante I1 em ação.

**T5. Premissa falsa.** "Quem era o chefe dela quando morava em Recife e
trabalhava na Kaia?"
```
... follow("works_at") ; filter(name="Kaia")
→ 0 trilhas, death_cause="empty_temporal_window"
```
`[jan-14, mai-01) ∩ [set-05, ∞) = ∅`. A resposta correta é que a premissa é
falsa, e o sistema sabe qual interseção esvaziou. Perguntas adversariais não
exigem detector de adversarialidade.

**T6. Contagem.** "Quantas vezes ela mencionou escalada?"
```
count(topic="escalada") → 2
```
No grafo dobrado a resposta seria 1, porque dobra funde ocorrências. Por isso
`count` ignora o grafo.

**T7. Evolução.** "Como a situação profissional dela mudou?"
```
anchor("Melanie") ; history("works_at")
→ [(Vertex, jan-08..set-05), (Kaia, set-05..)]
```

**T8. Caminho criado por dobra.** "Quem me contratou era meu chefe?"
```
anchor("Melanie") ; follow("hired⁻¹") ; follow("manages") ; filter(name="Melanie")
→ 1 trilha
```
Este caminho nunca foi observado em nenhum episódio. Ele existe porque a dobra
identificou `Rui` e `Rui Sampaio`.

### 17.4 Teste de canonicidade

O mais importante para publicação.

```python
def test_order_invariance():
    reference = build_memory(EPISODES)
    for _ in range(20):
        shuffled = shuffle_sessions(EPISODES)      # embaralha SESSÕES, mantém
                                                   # ordem interna dos turnos
        m = build_memory(shuffled)
        assert canonical_form(m) == canonical_form(reference)
```

`canonical_form` normaliza ids e serializa arestas ordenadas. A igualdade deve
valer para o grafo consolidado. Ela **não** vale para o staging, porque promoção
parcial é sensível à ordem. Documente isso: canonicidade existe no limite da
consolidação completa.

---

## 18. Avaliação

### 18.1 Benchmarks

- **LoCoMo** como principal. Atenção: parte relevante das perguntas é single-hop
  respondível de uma sessão só, e modelos de contexto longo já pontuam alto. Ele
  sozinho não distingue a contribuição.
- **LongMemEval** como complemento, cobre melhor atualização de conhecimento e
  raciocínio temporal.
- **Benchmark diagnóstico próprio**, gerado sinteticamente, com cinco famílias:
  retratação explícita, cadeia composicional com épocas distintas, premissa
  temporalmente impossível, contagem de recorrência, e homônimos. Esta suite é o
  que blinda o trabalho contra a crítica de que o ganho veio do LLM e não do
  mecanismo.

### 18.2 Baselines

Contexto completo, BM25, RAG denso, Mem0, Zep/Graphiti, A-Mem, HippoRAG.

### 18.3 Métricas

Além de F1 e acurácia por categoria:

| Métrica | Definição | O que mede |
|---|---|---|
| Variância sob permutação | desvio das respostas em 20 embaralhamentos | canonicidade |
| Taxa de compressão | arestas vivas / proposições ingeridas | eficiência da dobra |
| Precisão temporal | fração de respostas com janela correta | fibra temporal |
| Detecção de premissa falsa | recall de `empty_temporal_window` | coerência de caminho |
| Movimentos por pergunta | média por categoria observada | custo de acesso |
| Fidelidade da retratação | acurácia em consultas sobre `t_tx` | eixo de transação |

### 18.4 Ablações

Cada uma remove um mecanismo e deve degradar uma métrica específica:

| Ablação | Métrica que deve cair |
|---|---|
| Sem eixo de transação | fidelidade da retratação |
| Sem interseção de janela em `follow` | detecção de premissa falsa |
| Sem dependentes no catálogo | precisão temporal |
| Sem dobra | taxa de compressão, T8 falha |
| Sem staging (`tau_promote = 0`) | precisão geral |
| Sem `count` separado | perguntas de contagem |
| `tau_fold` variando 0.6 a 0.95 | curva precisão vs cobertura |

Se uma ablação não degrada nada, o mecanismo é supérfluo e deve sair do artigo e
do código.

---

## 19. Modos de falha e não-objetivos

### 19.1 Falhas esperadas e suas defesas

| Erro | Onde aparece | Defesa |
|---|---|---|
| Extrator vincula à entidade errada | aresta em vértice errado | dobra gera contradição, jornal reverte |
| Extrator inventa fato | aresta espúria | `contextual` fica no staging até confirmação |
| Extrator perde fato | silêncio | `rebuild` após melhorar o prompt |
| Data relativa ambígua | intervalo errado | `t_valid = None`, ordem parcial do log |
| Rótulo errado da lista fechada | endereço errado | `span` de proveniência permite auditar |
| Sobre-dobra | perde distinção episódica | `tau_fold` alto, `explicit_distinction` |
| Homônimos fundidos | contaminação em cascata | jornal + `unfold` + reprocessamento |

Defesa final, que é P5: a resposta é redigida a partir do episódio. A proposição
serve para localizar o episódio certo com precisão temporal. Se a extração
gravou algo errado, quem redige lê o texto e tem chance de corrigir.

### 19.2 Não-objetivos

Fora do escopo desta versão. Não implementar, não prometer:

- **Negação como inverso.** `label⁻¹` é direção, não negação. Negação é
  `polarity=false`. Confundir os dois quebra a semântica.
- **Axiomas de domínio gerais.** "pai de pai é avô" é um quociente do grupo
  livre, e quocientes gerais têm problema da palavra indecidível. Se for
  necessário, colocar em uma camada de reescrita separada, mantida confluente,
  nunca dentro da dobra.
- **Raciocínio probabilístico.** Confiança é filtro de promoção, não distribuição
  sobre mundos.
- **Memória compartilhada entre agentes.** A interseção de duas memórias como
  produto fibrado é extensão natural, mas não está nesta versão.
- **Aprendizado do catálogo.** A mineração da `unmapped_queue` produz um
  relatório. A promoção é humana e versionada.

---

## Apêndice A: glossário rápido

| Termo | Significado |
|---|---|
| `t_valid` | quando o fato é verdade no mundo |
| `t_tx` | quando o agente acreditou no fato |
| endereço | o par `(subject_id, relation)`, chave determinística de escrita |
| trilha | `(vértice, janela, proveniência)`, unidade do estado de acesso |
| janela | interseção dos intervalos de todas as arestas percorridas |
| dobra | identificação de arestas compatíveis; resolve entidades como efeito |
| staging | proposições extraídas ainda não promovidas ao grafo |
| dívida de consolidação | número de proposições no staging aguardando |
| CLOSE | fecha `t_valid`: deixou de ser verdade |
| RETRACT | fecha `t_tx`: nunca foi verdade |
