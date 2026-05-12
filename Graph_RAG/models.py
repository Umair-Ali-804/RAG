from pydantic import BaseModel, Field
from typing import List, Optional

class Entity(BaseModel):
    name: str
    entity_type: str
    description: Optional[str] = None

class Relation(BaseModel):
    source: str
    target: str
    relation_type: str
    description: Optional[str] = None

class KnowledgeGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
