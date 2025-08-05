from typing import List, Dict, Optional
from pathlib import Path
from ..models.schemas import CodeChunk, QueryResult
from ..integrations.github import GitHubIntegration
from ..integrations.langchain import LangChainIntegration
from ..integrations.openai import OpenAIInterface
from .embeddings import EmbeddingManager
from .vector_stores import VectorStoreManager
from .chunk import CodeChunker
from ..utils.file_utils import FileUtils

class CodeAnalyzer:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStoreManager()
        #self.vector_store = VectorStoreManager(embedding_dimension=1024)
        self.vector_store.reset_collection() 

        self.chunker = CodeChunker()
        self.github = GitHubIntegration()
        self.openai = OpenAIInterface()
        self.langchain = LangChainIntegration(self.vector_store)
        self.file_utils = FileUtils()
    
    def process_local_folder(self, folder_path: str) -> None:
        """Process a local folder of Python files"""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Process with ChromaDB
        documents, metadatas, ids = [], [], []
        chunk_count = 0
        
        for py_file in folder.rglob("*.py"):
            for chunk, metadata in self.chunker.chunk_file(py_file):
                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(f"chunk_{chunk_count}")
                chunk_count += 1
        
        if documents:
            self.vector_store.add_documents(documents, metadatas, ids)
            print(f"Processed {chunk_count} chunks from {folder_path}")
        
        # Process for LangChain
        self.vector_store.process_folder_for_langchain(folder_path)
    
    def process_github_repo(self, repo_url: str, token: Optional[str] = None) -> None:
        """Process a GitHub repository"""
        temp_dir = self.file_utils.create_temp_dir()
        try:
            self.github.clone_repo(repo_url, temp_dir, token)
            self.process_local_folder(temp_dir)
        finally:
            self.file_utils.cleanup_temp_dir(temp_dir)
    
    def query_codebase(self, question: str, n_results: int = 3, explain: bool = False) -> List[QueryResult]:
        """Query the codebase with optional explanations"""
        # Basic vector similarity search
        results = self.vector_store.query(question, n_results)
        
        if explain:
            for result in results:
                result.explanation = self.openai.generate_explanation(
                    question=question,
                    code=result.code,
                    context={
                        "file": result.file,
                        "type": result.type,
                        "name": result.name
                    }
                )
        return results
    
    def advanced_query(self, question: str, conversational: bool = False) -> str:
        """Use LangChain for more sophisticated queries"""
        return self.langchain.query(question, conversational)