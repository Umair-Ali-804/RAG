#  Graph RAG System

## What is Graph RAG?

Graph RAG augments traditional retrieval with a **knowledge graph** built from your documents. Instead of treating each chunk as an isolated piece of text, it extracts **entities** (people, concepts, organizations) and **relationships** between them, stores them in a graph structure, and uses graph traversal alongside vector search to answer questions.

This allows the system to answer questions that require **multi-hop reasoning** — e.g. "What is the relationship between X and Y?" — which flat vector search cannot handle well.

---

## Applications

- **Research paper analysis** — tracing citations, authors, concepts, and findings across papers
- **Legal document networks** — mapping relationships between parties, clauses, and cases
- **Enterprise knowledge graphs** — connecting people, projects, products, and teams
- **Medical knowledge bases** — disease → symptom → treatment relationship mapping
- **Competitive intelligence** — mapping company relationships, acquisitions, partnerships

---

## Core Components

| File | Role |
|------|------|
| `config.py` | API keys, model settings, graph traversal depth |
| `models.py` | Pydantic schemas: `Entity`, `Relation`, `KnowledgeGraph` |
| `graph_builder.py` | PDF loading, LLM-based entity/relation extraction, NetworkX graph construction |
| `system.py` | `GraphRAG` class — graph traversal + vector retrieval + answer generation |
| `main.py` | Entry point and interactive CLI |
| `requirements.txt` | Python dependencies |

### Pipeline

```
PDFs
 │
 ▼
[LLM Entity Extraction] ──► Entities + Relations
 │
 ▼
[NetworkX Graph] ◄──────────────────────────────┐
 │                                               │
 ▼                                               │
User Query ──► Entity Matching ──► Subgraph Traversal (depth=N)
 │
 ▼
[LLM] ──(graph context + text chunks)──► Answer
```

---

## Configuration (`config.py`)

```python
LLM_MODEL      = "stepfun/step-3.5-flash:free"
EMBEDDING_MODEL= "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE     = 800
GRAPH_DEPTH    = 2     # how many relationship hops to traverse
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

### 3. Run (graph is built fresh each run)

```bash
python main.py
```

> **Note:** Graph extraction uses LLM calls per chunk — for large documents this may take a few minutes on the first run. Consider caching the graph with `pickle` for repeated use.

### 4. Use as a library

```python
from config import RAGConfig
from graph_builder import GraphBuilder
from system import GraphRAG

config = RAGConfig()
gb = GraphBuilder(config)
chunks = gb.load_and_chunk(["./doc.pdf"])
gb.extract_graph(chunks)

rag = GraphRAG(config, gb)
result = rag.query("What is the relationship between X and Y?")
print(result["answer"])
```

---

## Project Structure

```
Graph_RAG/
├── config.py          # Configuration
├── models.py          # Entity, Relation, KnowledgeGraph schemas
├── graph_builder.py   # PDF loading + graph extraction (NetworkX)
├── system.py          # GraphRAG core logic
├── main.py            # Entry point & CLI
├── requirements.txt   # Dependencies
└── README.md          # This file
```

---

## Key Concepts

- **Entity**: A named concept extracted from text (person, org, technology, etc.)
- **Relation**: A directed edge between two entities with a relationship type
- **Subgraph**: The neighbourhood of nodes reachable within `GRAPH_DEPTH` hops from a matched entity
- **Graph context**: A textual representation of the subgraph edges, passed to the LLM alongside raw text chunks
