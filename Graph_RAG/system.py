from typing import Dict
import networkx as nx
from langchain_openai import ChatOpenAI
from graph_builder import GraphBuilder
from config import RAGConfig


ANSWER_PROMPT = """Answer the question using the knowledge graph context below.

Question: {query}

Knowledge Graph Context (entities and relationships):
{graph_context}

Original Text Chunks:
{text_context}

Answer:"""


class GraphRAG:
    def __init__(self, config: RAGConfig, graph_builder: GraphBuilder):
        self.config = config
        self.gb = graph_builder
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL, openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL, temperature=config.LLM_TEMPERATURE,
        )

    def _extract_query_entities(self, query: str):
        nodes = list(self.gb.graph.nodes())
        return [n for n in nodes if n.lower() in query.lower()]

    def _graph_context_str(self, subgraph: nx.DiGraph) -> str:
        lines = []
        for u, v, data in subgraph.edges(data=True):
            lines.append(f"{u} --[{data.get('relation_type','related')}]--> {v}")
        return "
".join(lines) if lines else "No graph context found."

    def _text_context(self, query: str) -> str:
        results = []
        for chunk in self.gb.chunks[:50]:
            if any(word.lower() in chunk.page_content.lower() for word in query.split()):
                results.append(chunk.page_content[:500])
            if len(results) >= self.config.TOP_K_DOCUMENTS:
                break
        return "

".join(results) if results else "No text context found."

    def query(self, question: str, verbose: bool = True) -> Dict:
        entities = self._extract_query_entities(question)
        if verbose:
            print(f"  Matched entities: {entities or ['none — using full graph sample']}")

        if entities:
            subgraph = self.gb.get_subgraph(entities[0], depth=self.config.GRAPH_DEPTH)
        else:
            nodes = list(self.gb.graph.nodes())[:20]
            subgraph = self.gb.graph.subgraph(nodes)

        graph_ctx = self._graph_context_str(subgraph)
        text_ctx = self._text_context(question)

        prompt = ANSWER_PROMPT.format(query=question, graph_context=graph_ctx, text_context=text_ctx)
        answer = self.llm.invoke(prompt).content

        if verbose:
            print(f"
Answer:
{answer}
")
        return {"query": question, "entities": entities, "answer": answer}
