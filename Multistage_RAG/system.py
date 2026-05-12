from typing import List, Any, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import ChatOpenAI
from config import RAGConfig


ANSWER_PROMPT = """Answer the question using only the provided documents.

Question: {query}

Documents:
{docs_text}

Answer:"""

QUERY_EXPANSION_PROMPT = """Generate 3 alternative phrasings of this query to improve retrieval.
Return ONLY a Python list of strings, e.g. ["query1", "query2", "query3"]

Original Query: {query}"""


class MultistageRAG:
    def __init__(self, config: RAGConfig, vector_store):
        self.config = config
        self.vector_store = vector_store
        self.cross_encoder = CrossEncoder(config.CROSSENCODER)
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL, openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL, temperature=config.LLM_TEMPERATURE,
        )

    def stage1_retrieve(self, query: str) -> List[Any]:
        """Broad vector search + query expansion"""
        docs = self.vector_store.similarity_search(query, k=self.config.STAGE1_TOP_K)
        # Query expansion: generate alternative queries
        try:
            expanded = self.llm.invoke(QUERY_EXPANSION_PROMPT.format(query=query)).content
            alt_queries = eval(expanded) if expanded.strip().startswith("[") else []
            for alt_q in alt_queries[:2]:
                extra = self.vector_store.similarity_search(alt_q, k=5)
                docs.extend(extra)
        except Exception:
            pass
        # Deduplicate
        seen, unique = set(), []
        for d in docs:
            key = d.page_content[:100]
            if key not in seen:
                seen.add(key)
                unique.append(d)
        return unique

    def stage2_rerank(self, query: str, docs: List[Any]) -> List[Any]:
        """Cross-encoder re-ranking"""
        pairs = [(query, d.page_content) for d in docs]
        scores = self.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:self.config.STAGE2_TOP_K]]

    def stage3_select(self, query: str, docs: List[Any]) -> List[Any]:
        """LLM-based final relevance filtering"""
        doc_summaries = "
".join(f"[{i+1}] {d.page_content[:200]}..." for i, d in enumerate(docs))
        prompt = f"""Rate each document 1-10 for relevance to the query.
Return ONLY a comma-separated list of numbers, e.g. "8,6,9,4,7,3,2,8"

Query: {query}
Documents:
{doc_summaries}

Scores:"""
        try:
            resp = self.llm.invoke(prompt).content.strip()
            scores = [float(s.strip()) for s in resp.split(",")]
            ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            return [d for _, d in ranked[:self.config.STAGE3_TOP_K]]
        except Exception:
            return docs[:self.config.STAGE3_TOP_K]

    def generate(self, query: str, docs: List[Any]) -> str:
        docs_text = "

".join(f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(docs))
        return self.llm.invoke(ANSWER_PROMPT.format(query=query, docs_text=docs_text)).content

    def query(self, question: str, verbose: bool = True) -> Dict:
        if verbose:
            print("  Stage 1: Broad retrieval + query expansion...")
        stage1_docs = self.stage1_retrieve(question)
        if verbose:
            print(f"    Retrieved {len(stage1_docs)} candidates")
            print("  Stage 2: Cross-encoder re-ranking...")
        stage2_docs = self.stage2_rerank(question, stage1_docs)
        if verbose:
            print(f"    Kept top {len(stage2_docs)}")
            print("  Stage 3: LLM relevance filtering...")
        final_docs = self.stage3_select(question, stage2_docs)
        if verbose:
            print(f"    Final {len(final_docs)} documents selected")
            print("  Generating answer...")
        answer = self.generate(question, final_docs)
        if verbose:
            print(f"
Answer:
{answer}
")
        return {"query": question, "answer": answer, "docs_used": len(final_docs)}
