import tempfile
import shutil

class FileUtils:
    """Utility class grouping file operations as static methods.
    Used without instantiation."""

    @staticmethod
    def create_temp_dir() -> str:
        """Create and return path to temporary directory"""
        return tempfile.mkdtemp()
    
    @staticmethod
    def cleanup_temp_dir(path: str) -> None:
        """Clean up temporary directory"""
        shutil.rmtree(path, ignore_errors=True)