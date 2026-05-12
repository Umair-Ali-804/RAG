import os

class RAGConfig:
    API_KEY = "your-openrouter-api-key-here"
    BASE_URL = "https://openrouter.ai/api/v1"
    LLM_MODEL = "stepfun/step-3.5-flash:free"
    LLM_TEMPERATURE = 0.2
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DB_TYPE = "chroma"
    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "self_reflective_rag_docs"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_DOCUMENTS = 5
    MAX_REFLECTION_ROUNDS = 3
    CONFIDENCE_THRESHOLD = 0.85

os.environ["OPENAI_API_KEY"] = RAGConfig.API_KEY
