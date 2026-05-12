import warnings
warnings.filterwarnings("ignore")
from config import RAGConfig
from graph_builder import GraphBuilder
from system import GraphRAG

def main():
    pdf_files = ["./your_document.pdf"]
    config = RAGConfig()
    gb = GraphBuilder(config)

    print("Building knowledge graph from PDFs...")
    chunks = gb.load_and_chunk(pdf_files)
    gb.extract_graph(chunks)

    rag = GraphRAG(config, gb)
    print("\n=== Graph RAG Interactive Mode === (type 'exit' to quit)")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() == "exit":
            break
        if q:
            rag.query(q)

if __name__ == "__main__":
    main()
