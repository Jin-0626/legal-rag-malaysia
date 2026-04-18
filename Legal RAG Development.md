# Research Guide: Legal RAG Development (Malaysia)

**Prepared for:** Electronic Systems Engineering Student / AI Researcher

**Tech Stack:** Python, VS Code Copilot, LangChain, ChromaDB, Ollama

---

## ## 1. Project Overview

Building a **Retrieval-Augmented Generation (RAG)** system specifically for Malaysian Law. The goal is to create a "Personal Law Consultant" that can interpret statutes (Acts) and provide citations with high fidelity, reducing the hallucination risk common in general-purpose LLMs.

---

## ## 2. Data Acquisition Strategy

The "Ground Truth" must be sourced from official Malaysian portals.

### ### Sources to Scrape:

- **Federal Legislation:** [LOM Portal](https://lom.agc.gov.my/) (The primary source for Acts of Parliament).
    
- **Judgments:** [Kehakiman Portal](https://www.kehakiman.gov.my/) (For Grounds of Judgment).
    
- **Tech-Specific:** Communications and Multimedia Act 1998, Computer Crimes Act 1997.
    

### ### Copilot Prompt for Scraper:

> "Write a Python script using `requests` and `BeautifulSoup` to scrape PDF links from the AGC Laws of Malaysia (LOM) index. The script should download PDFs into folders organized by year and create a `metadata.json` mapping the Act Number to the filename."

---

## ## 3. The RAG Pipeline

### ### Step 1: Document Processing (PDF to Markdown)

- **Reasoning:** Markdown preserves structural headers (Section 1, Section 2) better than plain text, which is vital for legal citations.
    
- **Tool:** `PyMuPDF` or `Marker`.
    

### ### Step 2: Semantic & Structural Chunking

Legal text requires **Section-Aware Chunking**. Do not split in the middle of a clause.

- **Copilot Prompt:** > "Using LangChain's `RecursiveCharacterTextSplitter`, create a function that splits Malaysian legal text. Prioritize splitting at 'Section' headers using regex `r'Section\s+\d+'`. Ensure a chunk size of 1000 tokens with 10% overlap."
    

### ### Step 3: Vector & Relationship Mapping (The GNN Approach)

Since you are familiar with **Graph Neural Networks (GNN)**, implement **GraphRAG**:

- **Nodes:** Specific Sections of an Act.
    
- **Edges:** "Amends," "Refers To," or "Overrules."
    
- **Vector DB:** ChromaDB (for semantic similarity).
    
- **Graph DB:** Neo4j (to store the relationships between Acts).
    

---

## ## 4. Prompt Engineering & Guardrails

To prevent the AI from "inventing" laws, use a restricted system prompt.

### ### System Prompt Template:

Plaintext

```
You are a Malaysian Legal Research Assistant. 
1. Use ONLY the provided context to answer.
2. Every claim MUST include a citation (e.g., 'According to Section 4 of the PDPA 2010...').
3. If the answer is not in the context, state that you cannot find the relevant statute.
4. Do not offer legal advice, only legal information.
```