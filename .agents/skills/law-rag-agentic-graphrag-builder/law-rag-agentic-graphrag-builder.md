---
name: law-rag-agentic-graphrag-builder
description: build, debug, evaluate, and maintain a legal RAG chatbot using agentic workflows, GraphRAG or legal knowledge graph retrieval, hybrid search, citation validation, version-aware indexing, and Robot Framework source monitoring for newly released acts and regulations.
---

# Law RAG Agentic GraphRAG Builder

Use this skill when working on a legal chatbot, legal retrieval system, statute QA system, regulation update pipeline, or compliance assistant that relies on retrieved evidence from official legal sources.

## Mission
Build and maintain a law-domain retrieval and answer system that is:
- citation-grounded
- version-aware
- graph-aware
- updateable through automated monitoring
- conservative under uncertainty

## Scope
This skill covers:
- legal document ingestion
- legal hierarchy parsing
- GraphRAG or legal graph construction
- hybrid retrieval
- agent workflow routing
- citation validation
- amendment/version handling
- Robot Framework source monitoring and ingestion triggering
- regression and evaluation design

This skill does not justify giving unsupported legal advice.

## High-level rules
1. Evidence before generation.
2. Legal structure before chunking.
3. Version metadata is mandatory.
4. Use hybrid retrieval rather than vector-only retrieval.
5. Prefer deterministic extraction for legal structure and references.
6. Keep fixes minimal when debugging.
7. Preserve provenance for every answerable unit.

## System assumptions
Unless the user states otherwise, assume:
- official legal sources are the corpus of record
- the chatbot should answer with citations
- the system should distinguish enactment, amendment, consolidation, and repeal states
- update automation should trigger ingestion, not replace the ingestion logic

## Working method

### 1. Restate the task precisely
Identify whether the task is mainly about:
- ingestion
- parsing
- graph design
- retrieval quality
- answer generation
- versioning
- update automation
- evaluation

### 2. Localize the change
Prefer the smallest layer that should change.
Examples:
- parser bug -> fix parser, not agent logic
- citation mismatch -> fix grounding or source mapping, not the whole retriever
- update detection failure -> fix Robot Framework selectors or source adapter, not GraphRAG

### 3. Separate legal truth from model behavior
Always distinguish:
- what the source text says
- what the parser extracted
- what the retriever found
- what the model inferred

### 4. Preserve auditability
Any change that affects answers should preserve or improve:
- source traceability
- version traceability
- section mapping
- reproducibility

## Legal parsing rules
- Parse document hierarchy before semantic chunking.
- Keep section paths stable.
- Normalize references like section, subsection, paragraph, part, schedule.
- Store original text spans and source metadata.
- Treat definitions, obligations, exceptions, and penalties as first-class retrieval targets.

## Graph construction rules
Recommended nodes:
- Document
- Act
- Regulation
- Part
- Chapter
- Section
- Subsection
- Paragraph
- Definition
- Obligation
- Exception
- Penalty
- Version
- LegalConcept

Recommended edges:
- CONTAINS
- DEFINES
- REFERS_TO
- APPLIES_TO
- IMPOSES
- EXCEPTS
- HAS_PENALTY
- AMENDS
- REPEALS
- SUPERSEDES
- EFFECTIVE_FROM
- CITED_IN

Prefer this extraction order:
1. structural edges
2. explicit references
3. deterministic legal semantics
4. constrained LLM-assisted semantic edges only if needed

## Retrieval rules
Use a hybrid retrieval mindset:
- exact metadata match when possible
- lexical retrieval for explicit section references
- vector retrieval for semantic paraphrases
- graph traversal for definitions, obligations, exceptions, and cross-reference paths
- reranking before final evidence selection

### Query routing
- direct lookup -> exact + lexical first
- definition -> graph + lexical
- obligation -> graph + vector + rerank
- exception -> graph traversal + support chunks
- amendment/timeline -> version graph + diff records
- broad summary -> hybrid retrieval with strict evidence sufficiency checks

## Answer rules
Answers should preferably include:
1. direct answer
2. legal basis
3. exceptions or limits
4. version/date caveat if relevant
5. citations
6. uncertainty note when necessary

Never present an unsupported inference as settled legal fact.

## Versioning rules
Always check whether the issue involves:
- original text
- amendment
- consolidated text
- repeal or supersession
- effective date boundaries

Do not mix versions silently.

## Robot Framework automation rules
Use Robot Framework for:
- opening official publication pages
- detecting new act or regulation releases
- extracting title/date/link
- downloading files or pages
- triggering ingestion APIs
- collecting screenshots and failure logs

Do not use Robot Framework for:
- core legal parsing
- graph indexing
- embeddings
- answer synthesis

### Update pipeline expectation
1. detect candidate release
2. compare against registry
3. download and checksum
4. register document
5. trigger ingestion API
6. parse and index
7. run regression
8. promote if validation passes

## Evaluation rules
Prefer research-quality evidence over anecdotal impressions.
Track:
- section hit rate
- citation precision
- unsupported claim rate
- hallucinated citation rate
- version correctness
- exception coverage
- gold-set pass rate

## Debugging rules
When diagnosing a bug:
- isolate the failing layer
- inspect evidence paths first
- verify parser and source mapping before changing prompting
- keep unrelated code untouched
- explain what changed, why, and what remains uncertain

## Deliverable style
When asked to help with implementation, prefer to produce:
- architecture decisions
- code patches with minimal scope
- explicit assumptions
- regression plan
- risk notes
- next steps sorted by priority

## Preferred folder structure
```text
apps/
automation/robot/
src/ingestion/
src/parsing/
src/graph/
src/retrieval/
src/agents/
src/versioning/
src/validation/
data/raw/
data/parsed/
data/diffs/
data/gold/
```

## Completion standard
A task is not complete merely because code runs.
It is complete when:
- evidence mapping is intact
- version handling is sane
- legal structure is preserved
- relevant regression checks were considered
- the change is explainable and auditable

