import os

class RAGConfig:
    API_KEY = "your-openrouter-api-key-here"
    BASE_URL = "https://openrouter.ai/api/v1"
    LLM_MODEL = "openai/gpt-4o"            # vision-capable model required
    LLM_TEMPERATURE = 0.2
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_TYPE = "chroma"
    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "multimodal_rag_docs"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_DOCUMENTS = 5
    IMAGE_DPI = 150                         # PDF page render resolution

os.environ["OPENAI_API_KEY"] = RAGConfig.API_KEY
