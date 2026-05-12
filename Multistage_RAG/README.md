# 🔭 Multistage RAG System

## What is Multistage RAG?

Multistage RAG implements a **multi-stage retrieval funnel** that progressively narrows down documents from a large candidate pool to the most relevant final set. Each stage applies a different technique — vector search, cross-encoder re-ranking, and LLM-based relevance scoring — at increasing cost and precision.

It also includes **query expansion**: generating alternative phrasings of the user's question to maximise recall in Stage 1.

---

## Applications

- **Large document corpora** — when the vector store contains thousands of chunks and coarse similarity is insufficient
- **Precision-critical retrieval** — legal, medical, financial contexts where the top 3-4 chunks must be exactly right
- **Search engines over private data** — enterprise search with high recall and high precision requirements
- **Multi-document synthesis** — combining information from many sources with careful ranking
- **RAG benchmarking** — establishing a strong retrieval baseline to evaluate generation quality

---

## Core Components

| File | Role |
|------|------|
| `config.py` | API keys, stage K values (`STAGE1_TOP_K`, `STAGE2_TOP_K`, `STAGE3_TOP_K`) |
| `processor.py` | PDF loading, chunking, vector store creation/loading |
| `system.py` | `MultistageRAG` class — 3-stage funnel + query expansion + answer generation |
| `main.py` | Entry point and interactive CLI |
| `requirements.txt` | Python dependencies |

### Pipeline (3-Stage Funnel)

```
User Query
    │
    ├──► Query Expansion (LLM generates 3 alt queries)
    │
    ▼
Stage 1: Vector Similarity Search  ──► Top 20 candidates
    │                               (broad, high recall)
    ▼
Stage 2: Cross-Encoder Re-ranking  ──► Top 8 candidates
    │                               (semantic precision)
    ▼
Stage 3: LLM Relevance Scoring     ──► Top 4 final docs
    │                               (highest precision)
    ▼
[LLM Generation] ──► Final Answer
```

---

## Configuration (`config.py`)

```python
LLM_MODEL      = "stepfun/step-3.5-flash:free"
EMBEDDING_MODEL= "sentence-transformers/all-MiniLM-L6-v2"
CROSSENCODER   = "BAAI/bge-base-en-v1.5"   # cross-encoder for Stage 2
STAGE1_TOP_K   = 20    # broad retrieval
STAGE2_TOP_K   = 8     # after re-ranking
STAGE3_TOP_K   = 4     # final LLM-scored selection
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
from system import MultistageRAG

config = RAGConfig()
processor = DocumentProcessor(config)
processor.load_vector_store()

rag = MultistageRAG(config, processor.vector_store)
result = rag.query("What projects has the candidate worked on?")
print(result["answer"])
```

---

## Project Structure

```
Multistage_RAG/
├── config.py          # Configuration & stage K values
├── processor.py       # PDF loading & vector store
├── system.py          # MultistageRAG 3-stage pipeline
├── main.py            # Entry point & CLI
├── requirements.txt   # Dependencies
└── README.md          # This file
```

---

## Stage Breakdown

| Stage | Method | Input | Output | Purpose |
|-------|--------|-------|--------|---------|
| 1 | Bi-encoder (vector) + query expansion | Raw query + 3 alt queries | 20 candidates | High recall |
| 2 | Cross-encoder scoring | 20 candidates | Top 8 | Semantic precision |
| 3 | LLM relevance scoring | Top 8 | Top 4 | Final high-precision selection |
| Gen | LLM generation | Top 4 docs | Answer | Answer synthesis |
