from pydantic import BaseModel
from typing import Optional, Dict, List

class CodeChunk(BaseModel):
    content: str
    metadata: Dict[str, str]

class QueryResult(BaseModel):
    code: str
    file: str
    type: str
    line: int
    name: str
    similarity: float
    explanation: Optional[str] = None

class GitHubRepoConfig(BaseModel):
    url: str
    token: Optional[str] = None

class AnalysisRequest(BaseModel):
    source_type: str  # 'local', 'github', 'gist'
    source_path: str
    github_config: Optional[GitHubRepoConfig] = None