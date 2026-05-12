import os

class RAGConfig:
    API_KEY = "your-openrouter-api-key-here"
    BASE_URL = "https://openrouter.ai/api/v1"
    LLM_MODEL = "stepfun/step-3.5-flash:free"
    LLM_TEMPERATURE = 0.2
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    TOP_K_DOCUMENTS = 5
    GRAPH_DEPTH = 2  # how many hops to traverse in the knowledge graph

os.environ["OPENAI_API_KEY"] = RAGConfig.API_KEY
