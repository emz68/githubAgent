from pydantic import BaseModel
from typing import Optional, Dict, List

class QueryResult(BaseModel):
    code: str
    file: str
    type: str
    line: int
    name: str
    similarity: float
    explanation: Optional[str] = None