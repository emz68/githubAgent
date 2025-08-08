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
    
    def generate_explanation(self, question: str, code: str, context: Dict) -> str:
        """Generate natural language explanation of code"""
        prompt = self._create_explanation_prompt(question, code, context)
        try:
            response = self.client.chat.completions.create(
                model="o4-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful code analyst. You are a senior developer analyzing code. Explain and answer questions about the code to the best of your ability. If you are not sure about file content or codebase structure pertaining to the user’s request, use your tools to read files and gather the relevant information: do NOT make up an answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=256
            )

            if hasattr(response, 'usage'):
                self._log_usage(
                    operation="generate_explanation",
                    model="o4-mini",
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                )

            self.usage_stats['total_tokens'] += response.usage.total_tokens
            self.usage_stats['calls'] += 1
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return "Explanation unavailable"
    
    def _create_explanation_prompt(self, question: str, code: str, context: Dict) -> str:
        """Create prompt for explanation generation"""
        return f"""You are a senior developer analyzing code. Explain and answer questions about the code to the best of your ability. If you are not sure about file content or codebase structure pertaining to the user’s request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.

            Question: {question}

            Context:
            - File: {context['file']}
            - Type: {context['type']}
            - Name: {context['name']}

            Code:
            {code}

            Explanation:
            """
    
    def get_usage(self):
        """Return current usage statistics"""
        return {
            'total_calls': self.usage_stats['calls'],
            'total_tokens': self.usage_stats['total_tokens']
        }
    
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
    