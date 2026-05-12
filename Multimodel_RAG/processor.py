import base64
from io import BytesIO
from typing import List, Any
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import RAGConfig

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class MultimodalProcessor:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        self.vector_store = None
        self.page_images = {}   # page_number -> base64 image string

    def load_pdfs_with_images(self, pdf_paths: List[str]) -> List[Any]:
        all_docs = []
        for path in pdf_paths:
            docs = PyPDFLoader(path).load()
            for d in docs:
                d.metadata["source_file"] = Path(path).name
            all_docs.extend(docs)

            if PYMUPDF_AVAILABLE:
                pdf = fitz.open(path)
                for page_num, page in enumerate(pdf):
                    mat = fitz.Matrix(self.config.IMAGE_DPI / 72, self.config.IMAGE_DPI / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("jpeg")
                    self.page_images[f"{Path(path).name}_p{page_num}"] = base64.b64encode(img_bytes).decode()
                pdf.close()
                print(f"  Rendered {len(pdf)} pages from {Path(path).name}")

        print(f"Loaded {len(all_docs)} pages | {len(self.page_images)} page images")
        return all_docs

    def build_vector_store(self, docs: List[Any]):
        chunks = self.splitter.split_documents(docs)
        self.vector_store = Chroma.from_documents(
            documents=chunks, embedding=self.embeddings,
            collection_name=self.config.COLLECTION_NAME,
            persist_directory=self.config.PERSIST_DIRECTORY,
        )
        self.vector_store.persist()
        print(f"Vector store built with {len(chunks)} text chunks")

    def load_vector_store(self):
        self.vector_store = Chroma(
            collection_name=self.config.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.config.PERSIST_DIRECTORY,
        )
        print("Vector store loaded")
