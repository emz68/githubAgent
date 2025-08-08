import sys
from pathlib import Path

# Add the package directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from analyzer.core.analyzer import CodeAnalyzer

def main():
    analyzer = CodeAnalyzer()
    
    # Example usage
    analyzer.process_github_repo("https://github.com/oxylabs/Python-Web-Scraping-Tutorial")
    
    
    answer = analyzer.smart_query("What is this repository about?")

    if isinstance(answer.content, str):
    # Analytical answer
        print(f"Analysis: {answer.content}")
        print("Supported by:", answer.context['sources'])
    else:
        # Raw code results
        for code in answer.content:
            print(code)

    analyzer.print_usage_stats()
    
    analyzer.langchain.clear_memory()

if __name__ == "__main__":
    main()