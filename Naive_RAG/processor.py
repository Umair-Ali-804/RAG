from typing import List, Any
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import RAGConfig


class DocumentProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.vector_store = None

    def load_pdfs(self, pdf_paths: List[str]) -> List[Any]:
        docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            loaded = loader.load()
            for d in loaded:
                d.metadata["source_file"] = Path(path).name
            docs.extend(loaded)
        print(f"Loaded {len(docs)} pages from {len(pdf_paths)} PDF(s)")
        return docs

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def build_vector_store(self, chunks: List[Any]):
        if self.config.VECTOR_DB_TYPE == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=chunks, embedding=self.embeddings,
                collection_name=self.config.COLLECTION_NAME,
                persist_directory=self.config.PERSIST_DIRECTORY,
            )
            self.vector_store.persist()
        else:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.save_local(self.config.PERSIST_DIRECTORY)
        print(f"Vector store built with {len(chunks)} chunks")

    def load_vector_store(self):
        if self.config.VECTOR_DB_TYPE == "chroma":
            self.vector_store = Chroma(
                collection_name=self.config.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=self.config.PERSIST_DIRECTORY,
            )
        else:
            self.vector_store = FAISS.load_local(self.config.PERSIST_DIRECTORY, self.embeddings)
        print("Vector store loaded")

    def process_and_store(self, pdf_paths: List[str]):
        docs = self.load_pdfs(pdf_paths)
        chunks = self.chunk_documents(docs)
        self.build_vector_store(chunks)
