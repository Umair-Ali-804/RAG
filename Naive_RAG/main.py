import warnings
warnings.filterwarnings("ignore")
from config import RAGConfig
from processor import DocumentProcessor
from system import NaiveRAG

def main():
    pdf_files = ["./your_document.pdf"]
    rebuild_db = False  # Set True on first run

    config = RAGConfig()
    processor = DocumentProcessor(config)
    if rebuild_db:
        processor.process_and_store(pdf_files)
    else:
        processor.load_vector_store()

    rag = NaiveRAG(config, processor.vector_store)
    print("\n=== Naive RAG Interactive Mode === (type 'exit' to quit)")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() == "exit":
            break
        if q:
            rag.query(q)

if __name__ == "__main__":
    main()
