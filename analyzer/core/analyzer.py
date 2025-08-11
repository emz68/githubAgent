from typing import List, Optional
from pathlib import Path
from ..models.schemas import QueryResult
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
        self.vector_store.reset_collection() 
        self.chunker = CodeChunker()
        self.github = GitHubIntegration()
        self.openai = OpenAIInterface()
        self.langchain = LangChainIntegration(self.vector_store, self.openai)
        self.file_utils = FileUtils()
    
    def process_local_folder(self, folder_path: str) -> None:
        """Process files in multiple languages"""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
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
    
    def print_usage_stats(self):
        """Convenience method to show current usage"""
        stats = self.openai.get_usage_summary()
        print(f"\nAPI Usage Summary:")
        print(f"Total Calls: {stats['total_calls']}")
        print(f"Total Tokens: {stats['total_tokens']}")

    def smart_query(self, question: str, *, max_code_results: int = 3,force_analytical: bool = False):
        # Step 1: Pure ChromaDB search
        code_results = self.vector_store.query(question, max_code_results)
        
        # Step 2: Heuristic to determine if analysis is needed
        needs_analysis = force_analytical or self._requires_analysis(question, code_results)
        
        if not needs_analysis:
            return code_results
        
        # Step 3: Prepare context for LangChain
        context = "\n\n".join(
            f"File: {r.file}\nCode:\n{r.code}" 
            for r in code_results
        )
        
        # Use LangChain for the analytical response
        return self.langchain.query(
            question=question,
            conversational=True,
            context=context  # Pass the context directly
        )
        
    
    def _requires_analysis(self, question: str, code_results: List[QueryResult]) -> bool:
        """Heuristic to decide between raw code vs LLM analysis"""
        question_lower = question.lower()
        
        # Case 1: Clearly analytical questions
        analytical_phrases = {
            'how', 'why', 'explain', 'analyze', 
            'compare', 'what is', 'walkthrough'
        }
        if any(phrase in question_lower for phrase in analytical_phrases):
            return True
        
        # Case 2: Poor semantic search results
        if not code_results or len(code_results[0].code) < 20:
            return True
            
        # Case 3: Very broad questions
        if len(question.split()) > 12:
            return True
            
        return False