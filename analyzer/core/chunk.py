import ast
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings

class CodeChunker:
    def __init__(self):
        """Initialize AST-based code chunker"""
        pass  # No setup needed for Python AST

    def chunk_file(self, filepath: Path) -> List[Tuple[str, Dict]]:
        """
        Extract semantic code chunks using Python's AST
        Returns: List of (code_chunk, metadata) tuples
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
            chunks = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    chunk_info = self._extract_chunk(node, source, filepath)
                    if chunk_info:
                        chunks.append(chunk_info)
            return chunks
            
        except (SyntaxError, UnicodeDecodeError) as e:
            warnings.warn(f"Skipping {filepath}: {str(e)}")
            return []

    def _extract_chunk(self, node: ast.AST, source: str, filepath: Path) -> Optional[Tuple[str, Dict]]:
        """Extract individual code chunk with metadata"""
        chunk_text = ast.get_source_segment(source, node)
        if not chunk_text:
            return None
            
        # Get decorators if present
        decorators = [
            ast.get_source_segment(source, d) 
            for d in getattr(node, 'decorator_list', [])
        ]
        
        # Combine decorators with chunk
        full_chunk = '\n'.join(decorators + [chunk_text]) if decorators else chunk_text
        
        return (full_chunk, {
            'file': str(filepath),
            'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
            'name': node.name,
            'line': node.lineno,
            'end_line': node.end_lineno,
            'docstring': ast.get_docstring(node),
            'signature': self._get_signature(node) if isinstance(node, ast.FunctionDef) else None
        })

    def _get_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []
        if node.args.posonlyargs:
            args.extend(f"/{arg.arg}" for arg in node.args.posonlyargs)
        if node.args.args:
            args.extend(arg.arg for arg in node.args.args)
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")
        if node.args.kwonlyargs:
            args.extend(f"{arg.arg}" for arg in node.args.kwonlyargs)
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")
            
        return f"{node.name}({', '.join(args)})"