import tempfile
import shutil
import os
from pathlib import Path
from typing import Optional

class FileUtils:
    @staticmethod
    def create_temp_dir() -> str:
        """Create and return path to temporary directory"""
        return tempfile.mkdtemp()
    
    @staticmethod
    def cleanup_temp_dir(path: str) -> None:
        """Clean up temporary directory"""
        shutil.rmtree(path, ignore_errors=True)
    
    @staticmethod
    def ensure_dir_exists(path: str) -> None:
        """Ensure directory exists, create if it doesn't"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def find_files(directory: str, extension: str) -> list[str]:
        """Find all files with given extension in directory"""
        return [
            str(p) for p in Path(directory).rglob(f"*.{extension}")
        ]