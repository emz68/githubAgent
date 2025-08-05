from github import Github
import requests
import tempfile
import shutil
import os
from typing import Optional

class GitHubIntegration:
    def clone_repo(self, repo_url: str, dest_path: str, token: Optional[str] = None):
        """Clone a GitHub repository to destination path"""
        if token:
            self._clone_with_token(repo_url, dest_path, token)
        else:
            self._clone_public(repo_url, dest_path)
    
    def _clone_with_token(self, repo_url: str, dest_path: str, token: str):
        """Clone using GitHub API (supports private repos)"""
        g = Github(token)
        repo_name = repo_url.replace("https://github.com/", "")
        repo = g.get_repo(repo_name)
        self._download_repo_contents(repo, dest_path)
    
    def _clone_public(self, repo_url: str, dest_path: str):
        """Clone public repo using git CLI"""
        if not repo_url.endswith('.git'):
            repo_url += '.git'
        os.system(f"git clone {repo_url} {dest_path}")
    
    def _download_repo_contents(self, repo, path: str):
        """Download repository contents recursively"""
        contents = repo.get_contents("")
        while contents:
            file_content = contents.pop(0)
            if file_content.type == "dir":
                os.makedirs(os.path.join(path, file_content.path), exist_ok=True)
                contents.extend(repo.get_contents(file_content.path))
            else:
                if file_content.path.endswith('.py'):
                    file_path = os.path.join(path, file_content.path)
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, 'wb') as f:
                        f.write(file_content.decoded_content)
    
    def process_gist(self, gist_url: str):
        """Process a GitHub Gist"""
        gist_id = gist_url.split('/')[-1].split('.')[0]
        api_url = f"https://api.github.com/gists/{gist_id}"
        
        response = requests.get(api_url)
        response.raise_for_status()
        return response.json()