from pydantic import BaseModel
from typing import Dict, List


class EntitySchema(BaseModel):
    entities: Dict[str, List[List[int, int]]]


class QuerySchema(BaseModel):
    query: str
    relevent_docs: List[str]
