import json
from typing import List, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from models import ReflectionResult
from prompts import GENERATE_PROMPT, REFLECT_PROMPT, REFINE_PROMPT
from config import RAGConfig


class SelfReflectiveRAG:
    def __init__(self, config: RAGConfig, vector_store):
        self.config = config
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL, openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL, temperature=config.LLM_TEMPERATURE,
        )

    def _docs_text(self, docs: List[Any]) -> str:
        return "

".join(f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(docs))

    def retrieve(self, query: str, k: int = None) -> List[Any]:
        return self.vector_store.similarity_search(query, k=k or self.config.TOP_K_DOCUMENTS)

    def generate(self, query: str, docs: List[Any]) -> str:
        return self.llm.invoke(GENERATE_PROMPT.format(query=query, docs_text=self._docs_text(docs))).content

    def reflect(self, query: str, answer: str, docs: List[Any]) -> ReflectionResult:
        raw = self.llm.invoke(REFLECT_PROMPT.format(
            query=query, answer=answer, docs_text=self._docs_text(docs)
        )).content.strip().lstrip("```json").rstrip("```").strip()
        try:
            return ReflectionResult(**json.loads(raw))
        except Exception:
            return ReflectionResult(is_sufficient=True, confidence=0.8, critique=None, needs_more_retrieval=False, refined_query=None)

    def refine(self, query: str, answer: str, critique: str, additional_docs: List[Any]) -> str:
        return self.llm.invoke(REFINE_PROMPT.format(
            query=query, answer=answer, critique=critique,
            additional_docs=self._docs_text(additional_docs)
        )).content

    def query(self, question: str, verbose: bool = True) -> Dict:
        docs = self.retrieve(question)
        answer = self.generate(question, docs)
        history = [{"round": 0, "answer": answer}]

        for round_num in range(1, self.config.MAX_REFLECTION_ROUNDS + 1):
            reflection = self.reflect(question, answer, docs)
            if verbose:
                print(f"  Reflection round {round_num}: confidence={reflection.confidence:.2f}, sufficient={reflection.is_sufficient}")

            if reflection.is_sufficient and reflection.confidence >= self.config.CONFIDENCE_THRESHOLD:
                if verbose:
                    print("  Answer accepted.")
                break

            if reflection.needs_more_retrieval and reflection.refined_query:
                extra_docs = self.retrieve(reflection.refined_query, k=3)
                docs = docs + extra_docs

            answer = self.refine(question, answer, reflection.critique or "", docs)
            history.append({"round": round_num, "answer": answer})

        if verbose:
            print(f"
Final Answer:
{answer}
")
        return {"query": question, "final_answer": answer, "rounds": len(history)}
