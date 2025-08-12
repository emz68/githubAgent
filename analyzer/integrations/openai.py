from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
from typing import Dict
from collections import defaultdict
from openai import OpenAI
import json
from datetime import datetime
from pathlib import Path

_ = load_dotenv(find_dotenv())

class OpenAIInterface:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
        self.usage_stats = defaultdict(int)
        self.log_file = Path("logs/openai_usage.log")
        self._setup_logging()
    
    def _setup_logging(self):
        """Ensure logs directory exists"""
        self.log_file.parent.mkdir(exist_ok=True)
        if not self.log_file.exists():
            self.log_file.write_text("timestamp,operation,model,tokens\n")

    
    def _log_usage(self, operation: str, model: str, usage: Dict):
        """Record API usage with timestamps"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
        
        # Update running totals
        self.usage_stats['total_tokens'] += entry['total_tokens']
        self.usage_stats['calls'] += 1
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        
    def get_usage_summary(self) -> Dict:
        """Return current usage statistics"""
        return {
            "total_calls": self.usage_stats['calls'],
            "total_tokens": self.usage_stats['total_tokens']
        }
    