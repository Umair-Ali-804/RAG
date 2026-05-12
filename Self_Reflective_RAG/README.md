# 🔁 Self-Reflective RAG System

## What is Self-Reflective RAG?

Self-Reflective RAG adds a **critique-and-refine loop** on top of standard RAG. After generating an initial answer, the system uses a second LLM call to critically evaluate the answer — checking accuracy, completeness, and grounding in the source documents. If the answer falls short, the system retrieves additional documents (using a refined query) and rewrites the answer, iterating until a confidence threshold is met.

This is inspired by the **Self-RAG** paper (Asai et al., 2023) and implements the core idea of using reflection tokens and iterative refinement.

---

## Applications

- **High-stakes Q&A** — medical, legal, financial queries where accuracy is critical
- **Long-form report generation** — ensuring completeness before delivering results
- **Fact-checking systems** — iteratively verifying claims against source documents
- **Research assistants** — producing thorough, well-grounded summaries
- **Customer support** — ensuring answers are fully grounded before delivery

---

## Core Components

| File | Role |
|------|------|
| `config.py` | API keys, reflection rounds, confidence threshold |
| `models.py` | `ReflectionResult` Pydantic schema |
| `prompts.py` | Prompt templates for generation, reflection, and refinement |
| `processor.py` | PDF loading, chunking, vector store management |
| `system.py` | `SelfReflectiveRAG` class — generate → reflect → refine loop |
| `main.py` | Entry point and interactive CLI |
| `requirements.txt` | Python dependencies |

### Pipeline

```
User Query
    │
    ▼
[Vector Store] ──► Top-K Docs ──► [LLM] ──► Initial Answer
                                                │
                                                ▼
                                    [Reflection LLM] ──► ReflectionResult
                                                │
                               ┌────────────────┴──────────────────┐
                          Sufficient?                          Not sufficient
                               │                                    │
                               ▼                          Retrieve more docs
                         Final Answer               (refined query) + Refine Answer
                                                            │
                                                    [Loop up to MAX_ROUNDS]
```

---

## Configuration (`config.py`)

```python
LLM_MODEL              = "stepfun/step-3.5-flash:free"
EMBEDDING_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
MAX_REFLECTION_ROUNDS  = 3      # maximum critique-refine iterations
CONFIDENCE_THRESHOLD   = 0.85   # stop loop when confidence exceeds this
TOP_K_DOCUMENTS        = 5
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
from system import SelfReflectiveRAG

config = RAGConfig()
processor = DocumentProcessor(config)
processor.load_vector_store()

rag = SelfReflectiveRAG(config, processor.vector_store)
result = rag.query("Summarize the candidate's technical skills")
print(result["final_answer"])
print(f"Completed in {result['rounds']} reflection round(s)")
```

---

## Project Structure

```
Self_Reflective_RAG/
├── config.py          # Configuration
├── models.py          # ReflectionResult schema
├── prompts.py         # Prompt templates
├── processor.py       # PDF loading & vector store
├── system.py          # SelfReflectiveRAG core logic
├── main.py            # Entry point & CLI
├── requirements.txt   # Dependencies
└── README.md          # This file
```

---

## Key Concepts

- **Reflection** — a structured LLM self-critique that produces a `ReflectionResult` (is_sufficient, confidence, critique, refined_query)
- **Adaptive retrieval** — if reflection identifies missing info, a refined query fetches additional documents
- **Confidence threshold** — loop exits early when confidence score exceeds `CONFIDENCE_THRESHOLD`, saving LLM calls
