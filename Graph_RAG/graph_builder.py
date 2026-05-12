import json
import networkx as nx
from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from models import Entity, Relation, KnowledgeGraph
from config import RAGConfig


EXTRACTION_PROMPT = """Extract entities and relations from the text below.
Return ONLY valid JSON with this structure:
{
  "entities": [{"name": "...", "entity_type": "...", "description": "..."}],
  "relations": [{"source": "...", "target": "...", "relation_type": "...", "description": "..."}]
}

Text:
{text}
"""


class GraphBuilder:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL, openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL, temperature=0,
        )
        self.graph = nx.DiGraph()
        self.chunks = []

    def load_and_chunk(self, pdf_paths: List[str]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE, chunk_overlap=self.config.CHUNK_OVERLAP
        )
        docs = []
        for p in pdf_paths:
            docs.extend(PyPDFLoader(p).load())
        self.chunks = splitter.split_documents(docs)
        print(f"Created {len(self.chunks)} chunks")
        return self.chunks

    def extract_graph(self, chunks: List[Any]) -> KnowledgeGraph:
        all_entities, all_relations = [], []
        for i, chunk in enumerate(chunks[:20]):  # limit for demo
            print(f"  Extracting from chunk {i+1}/{min(len(chunks),20)}...")
            try:
                resp = self.llm.invoke(EXTRACTION_PROMPT.format(text=chunk.page_content[:1500]))
                content = resp.content.strip().lstrip("```json").rstrip("```").strip()
                data = json.loads(content)
                all_entities.extend([Entity(**e) for e in data.get("entities", [])])
                all_relations.extend([Relation(**r) for r in data.get("relations", [])])
            except Exception as e:
                print(f"  Skipping chunk {i+1}: {e}")
        kg = KnowledgeGraph(entities=all_entities, relations=all_relations)
        self._build_nx_graph(kg)
        return kg

    def _build_nx_graph(self, kg: KnowledgeGraph):
        for e in kg.entities:
            self.graph.add_node(e.name, entity_type=e.entity_type, description=e.description)
        for r in kg.relations:
            self.graph.add_edge(r.source, r.target, relation_type=r.relation_type, description=r.description)
        print(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def get_subgraph(self, entity: str, depth: int = 2) -> nx.DiGraph:
        nodes = {entity}
        frontier = {entity}
        for _ in range(depth):
            new = set()
            for n in frontier:
                new.update(self.graph.successors(n))
                new.update(self.graph.predecessors(n))
            nodes.update(new)
            frontier = new
        return self.graph.subgraph(nodes)
