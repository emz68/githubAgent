from pydantic import BaseModel
from typing import Optional

class QueryResult(BaseModel):
    code: str
    file: str
    type: str
    line: int
    name: str
    similarity: float
    explanation: Optional[str] = None