#  Agentic RAG System

## What is Agentic RAG?

Agentic RAG (Retrieval-Augmented Generation) is an advanced RAG architecture where an **autonomous agent** orchestrates the entire retrieval and generation pipeline. Unlike traditional RAG, the agent actively decides *how* to retrieve, *whether* the results are sufficient, and *iteratively refines* the answer through self-correction loops.

The agent can break down complex queries, plan multi-step retrieval strategies, evaluate its own outputs, and improve them — all autonomously.

---

##  Applications

- **Enterprise Q&A systems** — answering complex multi-part business questions over large document corpora
- **Legal document analysis** — reasoning across contracts, case law, and regulations
- **Research assistants** — synthesizing findings across academic papers
- **Customer support bots** — handling nuanced, multi-intent queries with high accuracy
- **Medical knowledge bases** — multi-hop reasoning over clinical guidelines

---

## Core Components

| File | Role |
|------|------|
| `config.py` | API keys, model settings, vector DB config, agent hyperparameters |
| `models.py` | Pydantic schemas: `QueryClassification`, `QueryDecomposition`, `RetrievalEvaluation`, `GenerationEvaluation` |
| `prompts.py` | Prompt templates for classification, decomposition, retrieval eval, generation, refinement |
| `processor.py` | PDF loading, text chunking, vector store creation/loading (Chroma / FAISS) |
| `system.py` | Core `AgenticRAG` class — orchestrates the full agentic pipeline |
| `main.py` | Entry point: system setup, example query runner, interactive CLI |
| `requirements.txt` | Python dependencies |

### Pipeline Steps (inside `system.py → process_query()`)

1. **Query Classification** — Determines query type (factual, multi-hop, comparison, etc.) and complexity
2. **Query Decomposition** *(optional)* — Breaks complex queries into ordered sub-queries
3. **Document Retrieval** — Similarity search + cross-encoder re-ranking
4. **Retrieval Evaluation** — LLM judges whether retrieved docs are sufficient
5. **Answer Generation** — Generates answer grounded in retrieved documents
6. **Self-Correction Loop** — Iteratively evaluates and refines answer until quality threshold is met

---

## ⚙️ Configuration (`config.py`)

```python
LLM_MODEL = "stepfun/step-3.5-flash:free"   # via OpenRouter
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSSENCODER = "BAAI/bge-base-en-v1.5"
VECTOR_DB_TYPE = "chroma"                    # or "faiss"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_DOCUMENTS = 5
MAX_ITERATIONS = 5                           # self-correction rounds
ENABLE_SELF_CORRECTION = True
ENABLE_QUERY_DECOMPOSITION = True
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your PDF(s)

Place your PDF file(s) in the project directory. Update the list in `main.py`:

```python
pdf_files = ["./your_document.pdf"]
```

### 3. First-Time Setup (Build Vector DB)

```bash
# In main.py, set rebuild_db=True
rag_system, doc_processor = setup_system(pdf_files, rebuild_db=True)
```

Then run:

```bash
python main.py
```

### 4. Subsequent Runs (Load Existing DB)

```bash
# In main.py, set rebuild_db=False (default)
python main.py
```

### 5. Interactive Mode

After startup, you'll enter interactive mode:

```
🔍 Your question: What are the main skills listed in the document?
```

**Commands:**
- `verbose on` / `verbose off` — toggle step-by-step output
- `help` — show all commands
- `exit` — quit

### 6. Use as a Library

```python
from config import RAGConfig
from processor import DocumentProcessor
from system import AgenticRAG

config = RAGConfig()
processor = DocumentProcessor(config)
processor.load_existing_vector_store()

rag = AgenticRAG(config, processor.vector_store)
result = rag.process_query("What is the candidate's experience?", verbose=True)
print(result["final_answer"])
```

---

## 📦 Dependencies

- `langchain`, `langchain-community`, `langchain-openai`, `langchain-core`
- `chromadb` — vector store
- `sentence-transformers` — embeddings + cross-encoder reranking
- `pypdf` — PDF loading
- `pydantic` — data models

---

## 🗂️ Project Structure

```
Agentic_RAG/
├── config.py          # Configuration
├── models.py          # Pydantic data models
├── prompts.py         # All prompt templates
├── processor.py       # Document loading & vector store
├── system.py          # AgenticRAG core logic
├── main.py            # Entry point & CLI
├── requirements.txt   # Dependencies
└── README.md          # This file
```
