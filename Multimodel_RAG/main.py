import warnings
warnings.filterwarnings("ignore")
from config import RAGConfig
from processor import MultimodalProcessor
from system import MultimodalRAG

def main():
    pdf_files = ["./your_document.pdf"]
    rebuild_db = False  # Set True on first run

    config = RAGConfig()
    processor = MultimodalProcessor(config)

    if rebuild_db:
        docs = processor.load_pdfs_with_images(pdf_files)
        processor.build_vector_store(docs)
    else:
        processor.load_vector_store()

    rag = MultimodalRAG(config, processor)
    print("\n=== Multimodal RAG Interactive Mode === (type 'exit' to quit)")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() == "exit":
            break
        if q:
            rag.query(q)

if __name__ == "__main__":
    main()
