GENERATE_PROMPT = """Answer this question using only the provided documents.

Question: {query}

Documents:
{docs_text}

Answer:"""

REFLECT_PROMPT = """Critically evaluate this answer for the given question.

Question: {query}
Answer: {answer}
Source Documents: {docs_text}

Evaluate:
1. Is the answer accurate and complete?
2. Is it grounded in the documents?
3. Does it need more information?

Return ONLY valid JSON:
{{
  "is_sufficient": true,
  "confidence": 0.9,
  "critique": null,
  "needs_more_retrieval": false,
  "refined_query": null
}}"""

REFINE_PROMPT = """Improve this answer based on the critique and additional documents.

Original Question: {query}
Previous Answer: {answer}
Critique: {critique}
Additional Documents: {additional_docs}

Improved Answer:"""
