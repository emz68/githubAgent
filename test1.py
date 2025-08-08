import sys
from pathlib import Path

# Add the package directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from analyzer.core.analyzer import CodeAnalyzer

def main():
    analyzer = CodeAnalyzer()
    
    # Example usage
    analyzer.process_github_repo("https://github.com/oxylabs/Python-Web-Scraping-Tutorial")
    
    """ results = analyzer.query_codebase("Show web scraping examples", explain=True)
    for result in results:
        print(f"\nFound {result.type}: {result.name}")  # Use dot notation
        print(f"File: {result.file}")
        print(f"Explanation: {result.explanation}") """
    
    answer = analyzer.advanced_query(
        "What is this repository about?",
        conversational=True  # Keeps chat history
    )
    print(answer)

    analyzer.print_usage_stats()
    # Optional: Print raw log
    """ print("\nFull log:")
    with open("logs/openai_usage.log") as f:
        print(f.read()) """
    
    analyzer.langchain.clear_memory()

if __name__ == "__main__":
    main()