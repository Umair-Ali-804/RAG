from typing import List, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from config import RAGConfig

ANSWER_PROMPT = """You are a helpful assistant. Answer the question using ONLY the documents below.
If the answer is not present, say "I don't have enough information."

Question: {query}

Documents:
{docs_text}

Answer:"""


class NaiveRAG:
    def __init__(self, config: RAGConfig, vector_store):
        self.config = config
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL, openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL, temperature=config.LLM_TEMPERATURE,
        )

    def retrieve(self, query: str) -> List[Any]:
        return self.vector_store.similarity_search(query, k=self.config.TOP_K_DOCUMENTS)

    def generate(self, query: str, docs: List[Any]) -> str:
        docs_text = "\n\n".join(f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(docs))
        prompt = PromptTemplate(input_variables=["query", "docs_text"], template=ANSWER_PROMPT)
        return self.llm.invoke(prompt.format(query=query, docs_text=docs_text)).content

    def query(self, question: str, verbose: bool = True) -> Dict:
        if verbose:
            print(f"\n Retrieving {self.config.TOP_K_DOCUMENTS} documents...")
        docs = self.retrieve(question)
        answer = self.generate(question, docs)
        if verbose:
            print(f"\nAnswer:\n{answer}\n")
        return {"query": question, "docs": docs, "answer": answer}
