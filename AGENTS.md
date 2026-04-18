# AGENTS.md

## Project Identity
This repository implements a Legal RAG system for Malaysian law.

Primary goals:
1. Minimize hallucination.
2. Use citation-grounded answers only.
3. Preserve legal structure during parsing and chunking.
4. Improve retrieval precision through ranking/reranking.
5. Enforce RBAC and secure handling of protected operations.
6. Keep the code modular, testable, and easy to debug.

## Working Style
For non-trivial tasks, plan before coding.

Default workflow:
1. Understand the task
2. Identify assumptions and unknowns
3. Propose a minimal implementation plan
4. Make the smallest safe change
5. Add or update tests
6. Summarize trade-offs and risks

## Mandatory Output Format
Use this structure for substantial tasks:

Task
Assumptions
Plan
Changes
Tests
Risks

## Non-Negotiable Rules
- Do not invent laws, sections, citations, or legal claims.
- If a legal answer is not supported by retrieved context, abstain.
- Do not silently change public interfaces without explaining why.
- Prefer small, reviewable patches over broad refactors.
- Any retrieval or generation change must include or update tests.
- Any RBAC or security change must include positive and negative access tests.
- Preserve section/clause integrity where possible during chunking.
- Keep metadata flat and serializable for vector store ingestion.
- Avoid notebook-only logic in production modules.
- Do not add placeholder functionality and present it as complete.

## Repository Priorities
Priority order:
1. ingestion
2. chunking
3. embeddings / vector store
4. retrieval
5. generation / refusal policy
6. evaluation
7. security / RBAC
8. app layer

If the retrieval pipeline is unstable, do not prioritize UI work.