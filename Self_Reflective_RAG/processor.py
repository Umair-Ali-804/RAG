from typing import List, Any
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import RAGConfig


class DocumentProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.vector_store = None

    def process_and_store(self, pdf_paths: List[str]):
        docs = []
        for p in pdf_paths:
            loaded = PyPDFLoader(p).load()
            for d in loaded:
                d.metadata["source_file"] = Path(p).name
            docs.extend(loaded)
        chunks = self.splitter.split_documents(docs)
        self.vector_store = Chroma.from_documents(
            documents=chunks, embedding=self.embeddings,
            collection_name=self.config.COLLECTION_NAME,
            persist_directory=self.config.PERSIST_DIRECTORY,
        )
        self.vector_store.persist()
        print(f"Stored {len(chunks)} chunks")

    def load_vector_store(self):
        self.vector_store = Chroma(
            collection_name=self.config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.config.PERSIST_DIRECTORY,
        )
        print("Vector store loaded")
