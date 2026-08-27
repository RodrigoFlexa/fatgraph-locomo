# MECA — Memória Episódica de Compreensão Amortizada

> Desenho. Duas condições irmãs, **M1 (flat)** e **M2 (ribbon)**, que compartilham
> ingestão e consolidação byte a byte e diferem **só no leitor** — que é a única
> forma de a comparação "o ribbon graph paga?" significar alguma coisa.

---

## 1. O problema, dito sem o LoCoMo

Dado um corpus grande, crescente e datado de textos com autor (turnos de conversa,
atas, tíquetes, e-mails, prontuários), responder perguntas que exigem:

1. um fato dito uma vez e nunca repetido;
2. um fato que **mudou** ao longo do tempo;
3. a **composição** de dois fatos ditos em lugares diferentes;
4. saber que o corpus **não contém** a resposta.

Nada acima menciona falante, sessão, diálogo de duas pessoas ou categoria de
pergunta. Essa é a régua: se uma peça do método só faz sentido com uma dessas
coisas, ela não entra.

## 2. A tese: a estrutura de custo está invertida

No RAG usual a ingestão é barata e burra (fatiar, embeddar) e responder é caro e
difícil — e paga o preço **N vezes**, uma por pergunta. Cada consulta refaz, sob
ruído e orçamento apertado, a mesma compreensão: quem é "ela", quando foi "mês
passado", se aquilo aconteceu ou era só um plano.

MECA inverte:

> **Leia uma vez, profundamente. Responda muitas vezes, barato.**

A ingestão acontece uma vez por documento e pode ser arbitrariamente cara. É lá
que o LLM deve estar. O que se guarda não é um ponteiro para o texto — é o
**resultado da compreensão**.

Isso torna a proposta falseável em custo, não só em qualidade: existe um número
de perguntas **N\*** a partir do qual MECA fica mais barata que jogar o corpus
inteiro no contexto. `N* = custo_ingestão / (economia por consulta)`. Reportar N\*
é um resultado, e é genérico.

## 3. A unidade: a proposição atestada

O objeto atômico da memória deixa de ser o turno (ou o episódio) e passa a ser:

```
Proposição p
  sujeito      EntityRef          resolvido, com aliases
  predicado    forma canônica + embedding
  objeto       EntityRef | Literal | ∅
  qualificadores  {papel → EntityRef|Literal}     (aberto)
  válido_de    TimePoint | ∅
  válido_até   TimePoint | ∅ | ABERTO
  afirmado_em  TimePoint                          (quando foi DITO)
  modalidade   afirmado | pretendido | desejado | hipotético | perguntado | relatado
  polaridade   + | −
  evidência    [(doc_id, span)]                   obrigatória, ≥1
  derivada_de  [prop_id]                          ∅ se foi dita explicitamente
  substituída_por  prop_id | ∅
```

Cada campo tem de se defender. Por que estes:

**`sujeito`/`objeto` resolvidos.** É o que troca busca por similaridade por
**consulta**. Sem resolução não existe "procure a proposição cujo sujeito é X".

**`qualificadores` como dicionário aberto.** Responde à sua preocupação de
"muitos slots": o esquema é pequeno e fixo, o conteúdo é aberto. Lugar,
instrumento, quantidade, companhia — tudo entra como papel rotulado, nenhum vira
tipo de vértice novo. Nenhuma ontologia por domínio.

**Dois relógios: `válido_de` e `afirmado_em`.** "Mês passado eu pedi demissão",
dito em 2023-06, é `afirmado_em=2023-06` e `válido_de≈2023-05`. Quase todo KG-RAG
funde os dois. Separá-los é o que permite responder tanto "quando ela saiu?"
quanto "quando ela te contou?", e é o que faz supersessão funcionar.

**`modalidade` e `polaridade`.** Um plano não é um fato; uma negação não é uma
ausência. Em texto conversacional isso é a maior fonte de alucinação: recuperar
"vou me mudar para Lisboa" e responder "mora em Lisboa". Nenhum store de triplas
carrega isso.

**`evidência` obrigatória.** Nada existe na memória sem um span que a sustente.
É o que torna a ingestão com LLM defensável em vez de temerária.

**`derivada_de`.** Proposição implícita é marcada, nunca confundida com dita.
Ablação de uma linha.

**`substituída_por`.** O store guarda **estado**, não um saco de fatos.

## 4. Camada 1 — Compreensão (ingestão, LLM pesado, amortizado)

### 4.1 Segmentação

Fonte = sequência de *enunciados* `(id, autor?, texto, timestamp?)`. Corta-se em
**passagens** por queda de coesão, com o corte no **quantil da distribuição da
própria fonte** (não um limiar absoluto), e limites estruturais declarados
(mín./máx. de enunciados) que são premissa, não knob varrido.

Ponto arquitetural que vale registrar: **em MECA a segmentação importa muito
menos do que no MEST.** Lá a passagem era a unidade indexada e emitida, então uma
fronteira errada custava recuperação. Aqui a unidade guardada é a proposição; a
passagem é só a janela que o extrator lê. Fronteira errada custa um pouco de
contexto para o extrator, e nada depois disso. É um argumento a favor do desenho,
e é por isso que a ideia de segmentar por surpresa deixa de ser crítica.

### 4.2 Extração em duas passagens separadas

**Passagem A — o que foi dito.** O LLM recebe uma passagem, o autor (opcional) e a
data (opcional), e devolve as proposições **explícitas**, cada uma com o span
exato que a sustenta. Resolve anáfora ("ela" → quem), resolve tempo relativo
("mês passado" → data, dada a data da passagem) e marca modalidade e polaridade.

**Passagem B — o que se segue.** Chamada separada, que vê a passagem e a saída de
A, e propõe as proposições **implícitas** que um leitor competente tiraria: "vou
me mudar em abril" (dito em março) → "mora no novo lugar a partir de abril". Elas
saem marcadas `derivada_de`. Separar A de B é o que faz `implicatura: on|off` ser
uma ablação limpa em vez de uma reescrita de prompt.

Restrição inegociável, herdada da L6: **nenhum prompt vê pergunta, categoria ou
gabarito, e nenhum menciona falante, sessão ou diálogo de duas pessoas.** A
entrada é "uma passagem de texto".

### 4.3 Verificação por acarretamento

Toda proposição — dita ou derivada — passa por um verificador que pergunta uma
coisa só: **isto é acarretado pelo span citado?** O que não passa não entra.

É aqui que a fraqueza do LLM (inventar) é checada pela força do LLM (julgar
acarretamento, tarefa muito mais fácil). Uma chamada em lote por passagem, com
todas as proposições dela de uma vez — custo marginal pequeno.

`Verifier` é um protocolo: `LLMVerifier` (padrão) e `NullVerifier` (aceita tudo).
A diferença entre os dois **é a medição do que a verificação compra**, e é uma das
poucas coisas neste projeto que ninguém publicou.

## 5. Camada 2 — Consolidação (determinística; LLM só nos empates)

1. **Resolução de entidades.** Blocking por embedding + alias exato; LLM só
   decide empates acima de um limiar **derivado do corpus** (reusa
   `fgl.memory.calibration` — a parte defensável do trabalho já feito).

2. **Deduplicação de proposições.** Mesmo sujeito, predicado compatível (cosseno
   acima de quantil derivado), mesmo objeto → funde. Evidências se unem,
   `afirmado_em` fica com a mais antiga. **Repetição no corpus vira confiança em
   vez de ruído** — hoje ela vira duplicata competindo por orçamento.

3. **Funcionalidade do predicado, estimada do corpus.** Para supersessão é
   preciso saber se um predicado admite um valor por vez ("mora em") ou vários
   ("leu"). Sem ontologia: mede-se, na própria memória, a fração de sujeitos que
   têm exatamente um objeto para aquele predicado; acima de um quantil derivado,
   o predicado é funcional. Sem gabarito, sem lista escrita à mão. Fallback
   seguro: **nunca substituir**.

4. **Linha do tempo.** Para predicado funcional, uma proposição posterior fecha a
   anterior (`válido_até` = `válido_de` da nova; `substituída_por` apontado). Uma
   consulta sem tempo devolve o estado **vigente**; com tempo, o estado naquele
   instante.

5. **Vínculos tipados entre proposições.** `elabora` (mesmo sujeito+predicado,
   acrescenta qualificadores), `atualiza` (supersessão), `contradiz` (mesmo
   sujeito+predicado, objetos incompatíveis, validades sobrepostas e nenhuma
   substitui — **as duas ficam**, e o conflito é mostrado na resposta em vez de
   escondido). Causalidade só entra se **dita** e verificada; MECA não infere
   causa, que é onde a alucinação mora.

## 6. Camada 3 — Resposta (plano de consulta, barato)

A pergunta vira uma **proposição-alvo com um buraco**:

```
?  sujeito=Melanie   predicado≈"pintar"   objeto=?   em=?   modalidade=afirmado
```

Execução, estrutura primeiro e similaridade depois:

1. resolver as entidades nomeadas contra o store de entidades;
2. casar o predicado por embedding contra o vocabulário de predicados;
3. buscar proposições que satisfaçam os argumentos ligados;
4. se nenhuma → checar se as **entidades** sequer existem. Aqui a distinção é
   real e é uma consulta, não um palpite: *"não conheço essa pessoa"* é diferente
   de *"conheço, mas nunca disse isso sobre ela"*;
5. se o buraco exige junção — a pergunta liga A e pergunta sobre B, e nenhuma
   proposição tem os dois — o plano dá **um segundo passo**: os objetos e
   qualificadores das proposições do passo 1 viram ligações novas, e consulta-se
   de novo. **Limitado a 2 passos.** É o *evidence closure*, mas sobre
   proposições, onde é exato, em vez de sobre um grafo de similaridade, onde a L3
   provou que é ruído;
6. emitir as proposições **com seus spans**, agrupadas pelo passo do plano.

### 6.1 O que o gerador vê

Não uma pilha de 58 fatos. Uma junção apresentada como junção:

```
--- o que a memória afirma sobre Melanie e pintura ---
[2023-03, afirmado] Melanie retomou a pintura.
    "finalmente peguei os pincéis de novo no mês passado"   (D5:3, 12 abr 2023)

--- ligado por: a exposição ---
[2023-06, afirmado] Caroline foi a uma exposição do trabalho de Melanie.
    "fui ver a mostra dela no centro"                        (D9:1, 20 jun 2023)
```

A proposição é a afirmação resolvida; o span é a prova. O gerador não precisa
re-derivar nada — só ler, compor e citar.

### 6.2 A abstenção sai de graça

"Existe proposição vigente com este sujeito e este predicado?" é uma consulta com
resposta, não um escore com limiar. Foi exatamente isso que faltou ao atestado da
D35/D36 — e o motivo pelo qual ele não podia funcionar é que o MEST guarda formas
de superfície, não proposições. Aqui a propriedade vem da representação, e não é
uma categoria de benchmark: é a coisa que torna qualquer memória segura de usar.

## 7. As duas condições: M1 (flat) × M2 (ribbon)

Regra de ouro, herdada do que a L5 fez certo (`reduces_to_l2()` provado por
teste): **um único store, dois leitores.** A ingestão e a consolidação são as
mesmas linhas de código nas duas condições, então um delta medido não pode ser
outra coisa senão o leitor.

O store **é sempre um fatgraph**: vértices de proposição, entidade, predicado,
tempo e literal; arestas = incidências proposição–argumento rotuladas pelo papel.
A rotação é estrutura *adicional* sobre a mesma incidência, e o leitor flat
simplesmente a ignora.

| | **M1 — flat** | **M2 — ribbon** |
|---|---|---|
| enumerar a linha do tempo de um sujeito | índice invertido + ordenação | **órbita de σ** (cronológica por construção) |
| checar o buraco | teste de campo na proposição | **canto** no vértice de proposição |
| achar a junção | reconsulta ao índice (2 passos) | **caminhada de face (φ)** entre as duas entidades |
| ordem de emissão sob truncagem | por escore | **por posição na órbita/face** |

Essas quatro linhas são a única diferença, e as duas últimas são os únicos lugares
onde a rotação pode realmente pagar:

- **truncagem.** Quando não cabe tudo, a órbita dá uma ordem *principiada*
  (cronológica, local) onde o índice dá ordem por escore. A orçamento apertado
  elas divergem.
- **descoberta da junção.** A caminhada de face acha uma cadeia que a reconsulta
  perde quando a ligação intermediária não está no top-k.

Teste de identidade obrigatório: com `ribbon.order=score` e `ribbon.join=index`, o
`RibbonReader` tem de reproduzir o `FlatReader` **fato a fato**. Sem esse teste a
comparação não vale nada.

Declaro a expectativa antes de medir, para não ter como ajustar a narrativa
depois: **espero que M1 e M2 fiquem próximos**, porque os dois são exatos sobre o
mesmo store. Se ficarem, o resultado é publicável e é o que a linha L nunca teve —
a ablação honesta da topologia. Se M2 ganhar, ganha nas duas linhas acima e o
mecanismo estará nomeado antes do experimento.

## 8. O que mata a proposta (portões, em ordem)

1. **Fidelidade da extração** (quase grátis, sem gabarito de resposta): que fração
   dos turnos de evidência anotados tem ao menos uma proposição cujo span cai
   dentro? Se as proposições não cobrem a evidência, nada a jusante funciona.
   É o análogo do `hop-profile`, e mata a tese antes de qualquer F1.
2. **Composição**: no subconjunto em que toda a evidência está no contexto,
   multi-hop sai de 0,476? É a aposta central. Se não sair, o gargalo não era
   representação.
3. **Ribbon**: M1 × M2 ao mesmo orçamento, mesmo store.
4. **Custo**: reportar `tokens_ingest` e `tokens_QA` separados (a tabela já tem as
   colunas, e a L6 mostrou que `tokens ingest` não está instrumentado — corrigir)
   e calcular N\*.
5. **Portabilidade**: config congelado num segundo corpus.

E a régua que a linha L nunca passou: **B1-full-context = 0,546.** Se MECA não
bater isso, ela é uma contribuição de custo, não de conhecimento — e o artigo tem
de dizer isso com essas palavras.

## 9. O que o MECA reaproveita, e o que joga fora

**Reaproveita:** `fgl.core.FatGraph` (o store), `fgl.memory.calibration` (todo
limiar derivado do corpus, incluindo os novos), `fgl.memory.entities` (resolução),
`fgl.memory.temporal` (verificação determinística das datas que o LLM propõe),
`fgl.retrieval.embeddings`, a `Runner`/`_INGESTORS`/`_RETRIEVERS`, e a avaliação
inteira.

**Joga fora:** a extração não-generativa por spaCy+WordNet como *fonte da
memória*, o episódio como unidade de emissão, os cinco canais tipados com pesos, o
prior de ator, `enumerate_sets`, e o prompt de resposta com `{speaker_a}` —
tudo específico do formato do LoCoMo ou do seu gerador de perguntas.

## 10. Contra o que a proposta se defende

- **GraphRAG**: extrai entidades e relações, agrupa em comunidades, resume, e
  responde por map-reduce sobre resumos. Sem tempo de validade, sem modalidade,
  sem verificação por acarretamento, e o resumo apaga a proveniência.
- **KG-RAG / stores de triplas**: (s,p,o) sem intervalo de validade, sem
  modalidade, sem span obrigatório.
- **Grafos temporais centrados em eventos**: o parente mais próximo — evento,
  participantes, tempo. Falta modalidade/polaridade, falta verificação, e a
  resposta continua sendo recuperação por similaridade em vez de plano.
- **Memórias de agente (MemGPT, Zero-Mem e afins)**: guardam resumos e fatos,
  recuperam por similaridade; o *evidence closure* do Zero-Mem é o parente do
  passo 2 do plano, mas sobre texto, não sobre proposições.

A novidade defensável é a **combinação com verificação**: uma memória cuja unidade
é uma proposição *verificada, datada por intervalo, com modalidade e span
obrigatório*, respondida por *plano* em vez de similaridade — e cuja abstenção é
propriedade da representação, não mecanismo pendurado na frente.
