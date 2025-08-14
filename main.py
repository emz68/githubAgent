import sys
from pathlib import Path
from getpass import getpass
from analyzer.core.analyzer import CodeAnalyzer

def main():
    analyzer = CodeAnalyzer()
    print("-------------------------------------------------------------\n")
    print("GitHub Agent Console")
    print("To begin, type 'process <repository_url>'")
    print("To quit, type 'exit'\n")
    
    current_token = None  # Stores the active GitHub token
    
    while True:
        print("-------------------------------------------------------------\n")
        print("Ask a question or type 'commands' to see the list of commands")

        try:
            user_input = input("> ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                break

            if user_input.lower() in ['commands']:
                print("Commands:\n")
                print("Process repository: process <repository_url>\n")
                #print("Update private repository token: token\n")
                print("See API usage: stats\n")
                print("Clear history: clear\n")
                print("Quit: exit\n")
                continue
                
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
                analyzer.langchain.clear_memory()
                print("✓ Memory cleared")
                continue
                
            # Query handling
            if analyzer.vector_store.collection.count() > 0:
                print("What's the maximum number of files that should be analyzed for this prompt?")
                user_file_num = int(input().strip())
                response = analyzer.smart_query(user_input, max_code_results = user_file_num)
                print(response)
            else:
                print("! No codebase loaded. Please use 'process <repo_url>' first")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

    print("\nSession ended. Final stats:")
    analyzer.print_usage_stats()

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()