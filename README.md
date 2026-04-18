# Legal RAG System - Malaysian Law

A Retrieval-Augmented Generation (RAG) system designed for Malaysian legal documents with high-fidelity citations and reduced hallucination.

**Stack:** Python, LangChain, ChromaDB, Ollama, BeautifulSoup4, VS Code

---

## 📋 Project Structure

```
legal-rag/
├── data/
│   ├── raw_documents/          # Original PDFs and legal docs
│   ├── processed/              # Processed & chunked documents
│   └── embeddings/             # ChromaDB vector store
├── src/
│   ├── document_processor.py   # PDF → Markdown extraction & chunking
│   ├── rag_pipeline.py         # Main RAG retrieval & generation
│   └── web_scraper.py          # BeautifulSoup4-based web scraper
├── notebooks/
│   ├── 01_setup_and_test.ipynb # Setup verification & testing
│   └── 02_web_scraping_beautifulsoup.ipynb # Web scraping demo
├── config/
│   └── settings.py             # Configuration & paths
├── scripts/
│   └── setup_ollama.ps1        # Ollama model setup
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
└── README.md
```

---

## 🚀 Quick Start

### 1. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Set up Ollama**

Ollama runs a local LLM without needing OpenAI/cloud APIs.

#### Install Ollama:
- **Windows**: Download from [ollama.ai](https://ollama.ai)
- **macOS/Linux**: `curl https://ollama.ai/install.sh | sh`

#### Start Ollama:
```bash
ollama serve
```

#### Pull a model (in another terminal):
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

Verify models:
```bash
ollama list
```

### 3. **Configure Environment**

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Adjust settings if needed:
```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

### 4. **Run Setup Notebook**

Open `notebooks/01_setup_and_test.ipynb` in VS Code and run all cells to verify installation.

---

## 📚 Pipeline Overview

### Step 1: **Web Scraping with BeautifulSoup4**
- Parse HTML from LOM and Kehakiman portals
- Extract PDF links and metadata from legal documents
- Download Acts with proper naming and organization
- Handle requests efficiently with session management

### Step 2: **Document Processing**
- Extract text from PDFs (PyMuPDF/`fitz`)
- Detect legal sections (Section 1, 2, etc.) with regex
- Create overlapping chunks (1000 token size, 100 token overlap)
- Save metadata: Act number, section name, source

### Step 3: **Semantic Chunking**
- Use LangChain's `RecursiveCharacterTextSplitter`
- Prioritize section boundaries
- Preserve legal structure for accurate citations

### Step 4: **Vector Embeddings**
- Use Ollama's `nomic-embed-text` model
- Store in ChromaDB for fast retrieval
- Cosine similarity for section search

### Step 5: **RAG Query**
- Retrieve top-K relevant sections (default: 5)
- Generate answer with retrieved context
- System prompt enforces citation requirements

---

## 🤖 Usage Examples

### Web Scraping with BeautifulSoup4

```python
from src.web_scraper import LawScraper

# Initialize scraper
scraper = LawScraper(download_dir="data/raw_documents/downloaded")

# Scrape LOM Portal
documents = scraper.scrape_lom_acts(max_documents=10)

# Download PDFs
downloaded_files = scraper.download_multiple_pdfs(documents)

print(f"Downloaded {len(downloaded_files)} files")
```

### RAG Query

```python
from src.rag_pipeline import LegalRAG

# Initialize
rag = LegalRAG()
rag.create_collection()

# Add documents (from JSON chunks)
with open('data/processed/pdpa_chunks.json') as f:
    chunks = json.load(f)
rag.add_documents(chunks)

# Query
result = rag.query("What is the definition of personal data under PDPA?")
print(result['answer'])
print(result['sources'])
```

---

## 🛠️ Current Status

- [x] **Web Scraping**: BeautifulSoup4-based scraper for LOM/Kehakiman portals
- [x] **Document Processing**: PDF extraction and section-aware chunking
- [x] **RAG Pipeline**: LangChain + ChromaDB integration
- [ ] **Download & Process**: Batch download and process Acts
- [ ] **Neo4j Integration**: Graph relationships between Acts
- [ ] **Advanced Chunking**: ML-based section detection
- [ ] **Web UI**: Simple Flask interface for queries
- [ ] **Testing Suite**: Unit tests for all modules

---

## 📖 Next Steps

1. **Sample Data Testing**: Use `sample_pdpa_2010.md` to verify the pipeline works
2. **Web Scraping**: Implement scraper to download Acts from LOM
3. **Document Testing**: Run full pipeline on real Malaysian legislation
4. **Graph Setup**: Add Neo4j for tracking legal relationships
5. **UI Development**: Create a simple interface for queries

---

## ⚠️ Important Notes

- **Local LLMs Only** (for now): Using Ollama to avoid API costs and external dependencies
- **Citation Accuracy**: System prompt enforces mandatory citations to prevent hallucinations
- **Privacy**: All processing happens locally; no data sent to cloud services
- **Malaysian Context**: Documents and prompts tailored for Malaysian legal system

---

## 🤝 Contributing

1. Create a new branch for your feature
2. Add unit tests for new modules
3. Test against sample PDPA document
4. Submit PR with description

---

## 📝 License

This project is for educational and research purposes.

---

## 📧 Support

For issues or questions, check the documentation or create an issue in the repository.
