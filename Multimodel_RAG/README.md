# Multimodal RAG System

## What is Multimodal RAG?

Multimodal RAG extends traditional text-only RAG by incorporating **visual information** — images, charts, diagrams, and figures extracted from PDF pages — alongside text. A vision-capable LLM receives both text chunks and rendered page images, allowing it to answer questions that require reading figures, tables, or diagrams that text extraction alone would miss.

---

## Applications

- **Scientific paper Q&A** — answering questions about figures, plots, and experimental results
- **Technical manual analysis** — reading diagrams, circuit schematics, architecture drawings
- **Financial report analysis** — interpreting charts, tables, and graphs in PDFs
- **Medical imaging reports** — combining textual findings with visual scan references
- **Educational content** — textbooks with diagrams, equations, and annotated images

---

## Core Components

| File | Role |
|------|------|
| `config.py` | API keys, vision model selection, image DPI, vector DB settings |
| `processor.py` | PDF loading, page rendering to base64 JPEG (via PyMuPDF), vector store |
| `system.py` | `MultimodalRAG` class — text retrieval + image retrieval + vision LLM call |
| `main.py` | Entry point and interactive CLI |
| `requirements.txt` | Python dependencies |

### Pipeline

```
PDFs
 │
 ├──► [PyPDFLoader]  ──► Text Chunks ──► Vector Store
 │
 └──► [PyMuPDF]     ──► Page Images (base64 JPEG)
                                │
User Query ──► similarity_search ──► Top-K Text Chunks
                                │
                     matching page images
                                │
                                ▼
                    [Vision LLM] ──► Answer
```

---

## Configuration (`config.py`)

```python
LLM_MODEL  = "openai/gpt-4o"   # Must be a vision-capable model
IMAGE_DPI  = 150                # Resolution for page rendering
CHUNK_SIZE = 1000
TOP_K_DOCUMENTS = 5
```

> **Important:** This project requires a **vision-capable model** such as `openai/gpt-4o` or `anthropic/claude-3-5-sonnet`. Standard text-only models will ignore the image inputs.

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

### 3. First run — build vector store and render images

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

> **Note:** Page images are re-rendered on every fresh run (not persisted). For large PDFs consider saving `processor.page_images` to disk with `pickle`.

### 5. Use as a library

```python
from config import RAGConfig
from processor import MultimodalProcessor
from system import MultimodalRAG

config = RAGConfig()
processor = MultimodalProcessor(config)
docs = processor.load_pdfs_with_images(["./doc.pdf"])
processor.build_vector_store(docs)

rag = MultimodalRAG(config, processor)
result = rag.query("Describe the architecture diagram on page 3")
print(result["answer"])
```

---

## Project Structure

```
Multimodal_RAG/
├── config.py          # Configuration (model, DPI, DB settings)
├── processor.py       # PDF + image loading, vector store
├── system.py          # MultimodalRAG core logic
├── main.py            # Entry point & CLI
├── requirements.txt   # Dependencies
└── README.md          # This file
```

---

## Dependencies Note

- `pymupdf` (PyMuPDF) is required for page image rendering. If unavailable, the system gracefully falls back to text-only mode.
- A vision-capable LLM API key is required. OpenRouter supports `openai/gpt-4o` and `anthropic/claude-3-5-sonnet`.
