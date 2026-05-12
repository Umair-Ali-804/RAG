# 📄 Naive RAG System

## What is Naive RAG?

Naive RAG is the **simplest and most foundational** form of Retrieval-Augmented Generation. It follows a straightforward three-step pipeline: **Retrieve → Read → Generate**. Documents are chunked, embedded into a vector store, and at query time the top-K most similar chunks are fetched and passed directly to an LLM to produce an answer.

There is no query planning, no re-ranking, no self-correction — just pure, clean retrieval-augmented generation. It serves as the baseline against which all advanced RAG systems are measured.

---

## Applications

- **Document Q&A** — answering questions over a fixed set of PDFs or text files
- **Internal knowledge bases** — employee handbooks, policy documents, FAQs
- **Chatbots with context** — adding domain knowledge to a general LLM
- **Rapid prototyping** — fastest way to stand up a RAG system before adding complexity
- **Educational demos** — teaching the fundamentals of RAG

---

## Core Components

| File | Role |
|------|------|
| `config.py` | API keys, model names, chunking and retrieval parameters |
| `processor.py` | PDF loading, text splitting, vector store creation/loading |
| `system.py` | `NaiveRAG` class — retrieve + generate in one simple pass |
| `main.py` | Entry point and interactive CLI |
| `requirements.txt` | Python dependencies |

### Pipeline

```
User Query
    │
    ▼
[Vector Store] ──similarity_search──► Top-K Chunks
    │
    ▼
[LLM] ──(query + chunks)──► Answer
```

---

## Configuration (`config.py`)

```python
LLM_MODEL      = "stepfun/step-3.5-flash:free"   # via OpenRouter
EMBEDDING_MODEL= "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DB_TYPE = "chroma"   # or "faiss"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 200
TOP_K_DOCUMENTS= 5
```

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your PDF and update `main.py`

```python
pdf_files = ["./your_document.pdf"]
```

### 3. First run — build vector store

```python
# In main.py
rebuild_db = True
```

```bash
python main.py
```

### 4. Subsequent runs

```python
rebuild_db = False
```

```bash
python main.py
```

### 5. Use as a library

```python
from config import RAGConfig
from processor import DocumentProcessor
from system import NaiveRAG

config = RAGConfig()
processor = DocumentProcessor(config)
processor.load_vector_store()

rag = NaiveRAG(config, processor.vector_store)
result = rag.query("What is the document about?")
print(result["answer"])
```

---

## Project Structure

```
Naive_RAG/
├── config.py          # Configuration
├── processor.py       # PDF loading & vector store
├── system.py          # NaiveRAG core logic
├── main.py            # Entry point & CLI
├── requirements.txt   # Dependencies
└── README.md          # This file
```

---

## Limitations

Naive RAG is simple by design. Known limitations:

- No query understanding or decomposition
- Top-K retrieval may miss relevant chunks
- No answer quality checks
- Struggles with multi-hop or complex reasoning questions

For these cases, consider Agentic RAG, Self-Reflective RAG, or Multistage RAG.
