import os

class RAGConfig:
    API_KEY = "your-openrouter-api-key-here"
    BASE_URL = "https://openrouter.ai/api/v1"
    LLM_MODEL = "stepfun/step-3.5-flash:free"
    LLM_TEMPERATURE = 0.2
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CROSSENCODER = "BAAI/bge-base-en-v1.5"
    VECTOR_DB_TYPE = "chroma"
    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "multistage_rag_docs"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    # Stage 1: broad retrieval
    STAGE1_TOP_K = 20
    # Stage 2: re-ranking
    STAGE2_TOP_K = 8
    # Stage 3: final selection
    STAGE3_TOP_K = 4

os.environ["OPENAI_API_KEY"] = RAGConfig.API_KEY
