import sys
from pathlib import Path
from getpass import getpass
from analyzer.core.analyzer import CodeAnalyzer

def main():
    analyzer = CodeAnalyzer()
    print("GitHub Agent Console - Ask it something")
    print("Commands:\n")
    print("Process repository: process <repository_url>\n")
    print("Update private repository token: token\n")
    print("See API usage: stats\n")
    print("Clear history: clear\n")
    print("Quit: exit\n")
    
    current_token = None  # Stores the active GitHub token
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break
                
            # Special commands
            if user_input.startswith('process '):
                repo_url = user_input[8:].strip()
                try:
                    if "private" in repo_url.lower() and not current_token:
                        current_token = getpass("Enter GitHub token (hidden input): ")
                    analyzer.process_github_repo(repo_url, token=current_token)
                    print(f"✓ Processed repository: {repo_url}")
                except Exception as e:
                    print(f"Error: {str(e)}")
                continue
                
            if user_input == 'token':
                current_token = getpass("Enter new GitHub token (hidden input): ")
                print("✓ Token updated")
                continue
                
            if user_input == 'stats':
                analyzer.print_usage_stats()
                continue
                
            if user_input == 'clear':
                analyzer.conversation.clear()
                print("✓ Conversation history cleared")
                continue
                
            # Query handling
            if analyzer.vector_store.collection.count() > 0:
                response = analyzer.smart_query(user_input)
                print(response)
            else:
                print("! No codebase loaded. Use 'process <repo_url>' first")
                
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except Exception as e:
            print(f"Error: {str(e)}")

    print("\nSession ended. Final stats:")
    analyzer.print_usage_stats()

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()