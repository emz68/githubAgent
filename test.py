import os
import ast
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import chromadb
from chromadb.utils import embedding_functions
import requests
from github import Github  # PyGithub library
import tempfile
import shutil

class CodeAnalyzer:
    def __init__(self):
        # Initialize embedding model
        self.model_name = "Salesforce/SFR-Embedding-Code-400M_R"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=".chromadb")
        self.collection = self.client.get_or_create_collection(
            name="code_analysis",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.model_name, trust_remote_code=True
            )
        )

    def chunk_python_file(self, filepath: Path) -> list[tuple[str, dict]]:
        """Extract functions/classes using Python's AST"""
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
            chunks = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Get source segment with proper line tracking
                    chunk = ast.get_source_segment(source, node)
                    if chunk:
                        chunks.append((chunk, {
                            "file": str(filepath),
                            "type": "function" if isinstance(node, ast.FunctionDef) else "class",
                            "line": node.lineno,
                            "name": node.name
                        }))
            return chunks
            
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Skipping {filepath}: {str(e)}")
            return []

    def process_local_folder(self, folder_path: str):
        """Process all Python files in a local folder"""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Process all Python files
        documents, metadatas, ids = [], [], []
        chunk_count = 0
        
        for py_file in folder.rglob("*.py"):
            for chunk, metadata in self.chunk_python_file(py_file):
                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(f"local_chunk_{chunk_count}")
                chunk_count += 1
        
        # Store in ChromaDB
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Processed {chunk_count} chunks from local folder {folder_path}")

    def process_github_repo(self, repo_url: str, github_token: str = None):
        """Process a GitHub repository"""
        try:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Clone the repository
            if github_token:
                # Authenticated access (for private repos or higher rate limits)
                g = Github(github_token)
                repo_name = repo_url.replace("https://github.com/", "")
                repo = g.get_repo(repo_name)
                
                # Download the repository contents
                self._download_repo_contents(repo, temp_dir)
            else:
                # Public repository (limited to 60 requests/hour)
                if not repo_url.endswith('.git'):
                    repo_url = repo_url + '.git'
                
                # Simple clone (requires git installed)
                os.system(f"git clone {repo_url} {temp_dir}")
            
            # Process the downloaded files
            self.process_local_folder(temp_dir)
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _download_repo_contents(self, repo, path: str):
        """Download repository contents recursively"""
        contents = repo.get_contents("")
        
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                # Recursively process directories
                os.makedirs(os.path.join(path, file_content.path), exist_ok=True)
                contents.extend(repo.get_contents(file_content.path))
            else:
                # Download files (only Python files)
                if file_content.path.endswith('.py'):
                    file_path = os.path.join(path, file_content.path)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(file_content.decoded_content)

    def process_github_gist(self, gist_url: str):
        """Process a GitHub Gist"""
        try:
            gist_id = gist_url.split('/')[-1].split('.')[0]
            api_url = f"https://api.github.com/gists/{gist_id}"
            
            response = requests.get(api_url)
            response.raise_for_status()
            gist_data = response.json()
            
            documents, metadatas, ids = [], [], []
            chunk_count = 0
            
            for filename, file_info in gist_data['files'].items():
                if filename.endswith('.py'):
                    # Create a temporary file to use our existing chunking logic
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', encoding='utf-8') as temp_file:
                        temp_file.write(file_info['content'])
                        temp_file.flush()
                        
                        for chunk, metadata in self.chunk_python_file(Path(temp_file.name)):
                            documents.append(chunk)
                            metadata['file'] = f"gist:{gist_id}/{filename}"
                            metadatas.append(metadata)
                            ids.append(f"gist_chunk_{chunk_count}")
                            chunk_count += 1
            
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Processed {chunk_count} chunks from GitHub Gist {gist_id}")
                
        except Exception as e:
            print(f"Error processing Gist: {str(e)}")

    def query_codebase(self, question: str, n_results: int = 3) -> list[dict]:
        """Search codebase with natural language"""
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        return [
            {
                "code": doc,
                "file": meta["file"],
                "type": meta["type"],
                "line": meta["line"],
                "name": meta["name"],
                "similarity": 1 - dist  # Convert distance to similarity score
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]

# Example Usage
if __name__ == "__main__":
    analyzer = CodeAnalyzer()
    
    # Option 1: Process a local folder
    # analyzer.process_local_folder("./repo_clone")
    
    # Option 2: Process a public GitHub repository
    analyzer.process_github_repo("https://github.com/oxylabs/Python-Web-Scraping-Tutorial")
    
    # Option 3: Process a GitHub Gist
    # analyzer.process_github_gist("https://gist.github.com/username/gist_id")
    
    # Option 4: Process a private GitHub repository (requires token)
    # analyzer.process_github_repo("https://github.com/username/private-repo", github_token="your_github_token")
    
    # Query the codebase
    results = analyzer.query_codebase(
        "Show me how database connections are handled",
        n_results=3
    )
    
    for result in results:
        print(f"\n{result['type'].title()} '{result['name']}' (Line {result['line']})")
        print(f"File: {result['file']}")
        print(f"Similarity: {result['similarity']:.2f}")
        print(f"\n{result['code']}\n{'-'*50}")