import os
import shutil
from pathlib import Path
from git import Repo
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())

class GitHubEmbedder:
    def __init__(self):
        self.model_name = "Salesforce/SFR-Embedding-Code-400M_R"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.eval()

        self.parsers = Parser(PY_LANGUAGE)

    def clone_repo(self, repo_url: str, local_dir: str = "./repo_clone") -> str:
        """Clone repository with fresh clone each time"""
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        Repo.clone_from(repo_url, local_dir)
        return local_dir

    def get_chunks(self, filepath: Path) -> list[str]:
        """Precise AST-aware chunking"""
        ext = filepath.suffix.lower()
        lang = None
        
        # Language detection
        if ext == '.py':
            lang = 'python'
        elif ext in ('.js', '.ts'):
            lang = 'javascript'
        elif ext == '.go':
            lang = 'go'
        else:
            return []

        try:
            with open(filepath, 'rb') as f:
                code = f.read()

            parser = Parser()
            parser.set_language(self.parsers[lang])
            
            # Language-specific queries
            query = self.parsers[lang].query("""
            (function_definition body: (block) @func)  # Python example
            """)
            
            chunks = []
            for node, _ in query.captures(parser.parse(code)):
                chunks.append(code[node.start_byte:node.end_byte].decode('utf8'))
            
            return chunks
        except Exception as e:
            print(f"Error parsing {filepath}: {str(e)}")
            return []

    def embed_repository(self, repo_url: str):
        """Complete embedding workflow with error handling"""
        try:
            # 1. Clone fresh copy
            repo_path = "./repo_clone"
            print(f"Successfully cloned to: {repo_path}")
            
            # 2. Get code chunks
            code_chunks = self.get_chunks(repo_path)
            if not code_chunks:
                raise ValueError("No valid code files found")
            
            # 3. Generate embeddings
            with torch.no_grad():
                inputs = self.tokenizer(
                    code_chunks,
                    max_length=8192,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                outputs = self.model(**inputs)
                embeddings = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
            
            return embeddings.tolist(), code_chunks
        
        except Exception as e:
            print(f"Error processing repository: {str(e)}")
            return [], []

if __name__ == "__main__":
    embedder = GitHubEmbedder()
    
    # Test with a small known-good repository
    test_repo = "https://github.com/rtyley/small-test-repo"
    print(f"\nProcessing repository: {test_repo}")
    
    embeddings, chunks = embedder.embed_repository(test_repo)
    
    if chunks:
        print("\nSuccess! Sample code chunks:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\n--- Chunk {i+1} ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
    else:
        print("\nFailed to process repository")