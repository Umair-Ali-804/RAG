from pydantic import BaseModel, Field
from typing import List, Optional

class ReflectionResult(BaseModel):
    is_sufficient: bool = Field(description="Is the answer sufficient?")
    confidence: float = Field(description="Confidence score 0-1")
    critique: Optional[str] = Field(description="What is wrong or missing?")
    needs_more_retrieval: bool = Field(description="Should retrieve more documents?")
    refined_query: Optional[str] = Field(description="Better query for next retrieval round")
