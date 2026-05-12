from typing import List, Any, Dict
from langchain_openai import ChatOpenAI
from config import RAGConfig


class MultimodalRAG:
    def __init__(self, config: RAGConfig, processor):
        self.config = config
        self.processor = processor
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL, openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL, temperature=config.LLM_TEMPERATURE,
        )

    def retrieve_text(self, query: str) -> List[Any]:
        return self.processor.vector_store.similarity_search(query, k=self.config.TOP_K_DOCUMENTS)

    def get_relevant_images(self, docs: List[Any]) -> List[str]:
        images = []
        for doc in docs:
            source = doc.metadata.get("source_file", "")
            page = doc.metadata.get("page", 0)
            key = f"{source}_p{page}"
            if key in self.processor.page_images:
                images.append(self.processor.page_images[key])
        return images[:3]  # max 3 images per query

    def query(self, question: str, verbose: bool = True) -> Dict:
        docs = self.retrieve_text(question)
        text_ctx = "

".join(f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(docs))
        images = self.get_relevant_images(docs)

        content = [{"type": "text", "text": f"Question: {question}

Text context:
{text_ctx}

Answer based on text and any images provided:"}]
        for img_b64 in images:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})

        if verbose:
            print(f"  Using {len(docs)} text chunks + {len(images)} page image(s)")

        response = self.llm.invoke([{"role": "user", "content": content}])
        answer = response.content

        if verbose:
            print(f"
Answer:
{answer}
")
        return {"query": question, "answer": answer, "images_used": len(images)}
