from typing import List, Dict, Optional, Literal, Union
from pathlib import Path
from ..models.schemas import CodeChunk, QueryResult
from ..integrations.github import GitHubIntegration
from ..integrations.langchain import LangChainIntegration
from ..integrations.openai import OpenAIInterface
from .embeddings import EmbeddingManager
from .vector_stores import VectorStoreManager
from .chunk import CodeChunker
from ..utils.file_utils import FileUtils
from dataclasses import dataclass

@dataclass
class Response:
    content: Union[str, List[str]]  # Could be answer or code snippets
    context: dict = None
    needs_clarification: bool = False

class CodeAnalyzer:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.vector_store = VectorStoreManager()
        #self.vector_store = VectorStoreManager(embedding_dimension=1024)
        self.vector_store.reset_collection() 

        self.chunker = CodeChunker()
        self.github = GitHubIntegration()
        self.openai = OpenAIInterface()
        self.langchain = LangChainIntegration(self.vector_store, self.openai)
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
        results = self.vector_store.query(question, n_results)
        
        if explain:  # Only call OpenAI if explicitly requested
            for result in results:
                result.explanation = self.openai.generate_explanation(
                    question=question,
                    code=result.code,
                    context={"file": result.file, "type": result.type, "name": result.name}
                )
                
        return results
    
    def advanced_query(self, question: str, conversational: bool = False) -> str:
        """Use LangChain for more sophisticated queries"""
        return self.langchain.query(question, conversational)
    
    def print_usage_stats(self):
        """Convenience method to show current usage"""
        stats = self.openai.get_usage_summary()
        print(f"\nAPI Usage Summary:")
        print(f"Total Calls: {stats['total_calls']}")
        print(f"Total Tokens: {stats['total_tokens']}")

    def _classify_intent(self, question: str) -> Literal["SEMANTIC_SEARCH", "ANALYTICAL"]:
        """Rule-based intent classifier (Option A)."""
        question = question.lower().strip()
        
        analytical_keywords = [
            "explain", "how", "why", "describe", "walk me through",
            "analyze", "what is", "compare", "summarize"
        ]
        
        # Questions with these keywords or longer than 15 words -> ANALYTICAL
        if (any(keyword in question for keyword in analytical_keywords) or
            len(question.split()) > 15):
            return "ANALYTICAL"
        return "SEMANTIC_SEARCH"
    
    """ def query_auto(self, question: str, n_results: int = 3, conversational: bool = False) -> str | List[QueryResult]:

        intent = self._classify_intent(question)
        
        if intent == "SEMANTIC_SEARCH":
            return self.query_codebase(question, n_results=n_results, explain=False)  # No API
        else:
            return self.advanced_query(question, conversational=conversational)  # Uses API
         """

    def smart_query(self, question: str, *, max_code_results: int = 3,force_analytical: bool = False) -> Response:
        # Step 1: Pure ChromaDB search (no logging)
        code_results = self.vector_store.query(question, max_code_results)
        
        # Step 2: Heuristic to determine if analysis is needed
        needs_analysis = force_analytical or self._requires_analysis(question, code_results)
        
        if not needs_analysis:
            return Response(
                content=[r.code for r in code_results],
                context={
                    'files': [r.file for r in code_results],
                    'types': [r.type for r in code_results]
                }
            )
        
        # Step 3: Only log when LLM is actually used
        prompt = self._build_hybrid_prompt(question, code_results)
        response = self.openai.client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Log this API call
        self.openai._log_usage(
            operation="smart_query_analytical",
            model="o4-mini",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )
        
        return Response(
            content=response.choices[0].message.content,
            context={
                'supporting_code': [r.code for r in code_results],
                'sources': [r.file for r in code_results]
            }
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

    def _build_hybrid_prompt(self, question: str, code_results: List[QueryResult]) -> str:
        """Create prompt that combines question + code context"""
        code_context = "\n\n".join(
            f"File: {r.file}\nCode:\n{r.code}" 
            for r in code_results
        )
        return f"""
        Analyze this question about a codebase, using the following code snippets as reference.
        If the question can be answered by referencing specific code, quote the relevant parts.
        
        Question: {question}
        
        Relevant Code:
        {code_context}
        
        Answer:
        """