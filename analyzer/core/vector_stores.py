import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import logging
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from ..models.schemas import QueryResult
from ..core.chunk import CodeChunker
from pathlib import Path

class VectorStoreManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._initialize_chroma()
        self.text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=1000,
            chunk_overlap=200
        )

        self.embeddings = {
            "openai": OpenAIEmbeddings(),
            "hf": HuggingFaceEmbeddings()
        }
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection with proper embedding function"""
        self.client = chromadb.PersistentClient(path=".chromadb")
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            trust_remote_code=True
        )
        
        self.collection = self.client.get_or_create_collection(
            name="code_analysis",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def reset_collection(self):
        """Completely reset the collection"""
        try:
            self.client.delete_collection("code_analysis")
            self.logger.info("Deleted old collection")
        except Exception as e:
            self.logger.warning(f"Delete failed: {str(e)}")
        
        # Recreate collection
        self._initialize_chroma()
        self.logger.info("Created new collection")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add documents with automatic error handling"""
        try:
            # Validate metadata values
            validated_metadatas = []
            for meta in metadatas:
                validated_meta = {}
                for key, value in meta.items():
                    if value is None:
                        # Convert None to empty string or another default value
                        validated_meta[key] = ""
                    elif isinstance(value, (bool, int, float, str)):
                        validated_meta[key] = value
                    else:
                        # Convert other types to string
                        validated_meta[key] = str(value)
                validated_metadatas.append(validated_meta)
            
            self.collection.add(
                documents=documents,
                metadatas=validated_metadatas,
                ids=ids
            )
        except Exception as e:
            self.logger.error(f"Add failed: {str(e)}")
            raise    

    def query(self, question: str, n_results: int = 5) -> List[QueryResult]:
        try:
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results
            )
            return [
                QueryResult(
                    code=doc,
                    file=meta["file"],
                    type=meta["type"],
                    line=meta["line"],
                    name=meta["name"],
                    similarity=1 - dist
                )
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ]
        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            raise

    def process_folder_for_langchain(self, folder_path: str):
        """Process folder using AST-based semantic chunking"""
        chunker = CodeChunker()
        documents = []
        metadatas = []
        
        # Process all Python files in the folder
        for filepath in Path(folder_path).rglob("*.py"):
            chunks = chunker.chunk_file(filepath)
            for chunk_text, metadata in chunks:
                documents.append(chunk_text)
                metadatas.append(metadata)
        
        # Create vector store with both embedding types
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings["hf"],
            persist_directory=".chromadb_langchain",
            metadatas=metadatas
        )