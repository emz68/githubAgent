from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
from typing import Dict

_ = load_dotenv(find_dotenv())

class OpenAIInterface:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    
    def generate_explanation(self, question: str, code: str, context: Dict) -> str:
        """Generate natural language explanation of code"""
        prompt = self._create_explanation_prompt(question, code, context)
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful Python code analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=256
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return "Explanation unavailable"
    
    def _create_explanation_prompt(self, question: str, code: str, context: Dict) -> str:
        """Create prompt for explanation generation"""
        return f"""You are a senior Python developer analyzing code. Explain how this code relates to the question.

Question: {question}

Context:
- File: {context['file']}
- Type: {context['type']}
- Name: {context['name']}

Code:
{code}

Explanation:
"""