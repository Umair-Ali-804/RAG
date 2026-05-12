import warnings
warnings.filterwarnings("ignore")
from config import RAGConfig
from processor import DocumentProcessor
from system import SelfReflectiveRAG

def main():
    pdf_files = ["./your_document.pdf"]
    rebuild_db = False  # Set True on first run

    config = RAGConfig()
    processor = DocumentProcessor(config)
    if rebuild_db:
        processor.process_and_store(pdf_files)
    else:
        processor.load_vector_store()

    rag = SelfReflectiveRAG(config, processor.vector_store)
    print("\n=== Self-Reflective RAG === (type 'exit' to quit)")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() == "exit":
            break
        if q:
            rag.query(q)

if __name__ == "__main__":
    main()
