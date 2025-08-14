
![GitHub Agent Console](https://img.shields.io/badge/agent-GitHub%20Console-blue)
![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A command-line tool for cloning Python repositories (public or private), extracting semantic code chunks via Python’s AST, storing them in a ChromaDB vector store, and interactively querying the codebase by invoking OpenAI (via LangChain) for detailed explanations.

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [CLI Usage](#cli-usage)
6. [Code Walkthrough](#code-walkthrough)
7. [Configuration & Environment](#configuration--environment)
8. [License](#license)

---

## Features

- **GitHub Integration**
  • Download repos via GitHub API
- **AST-based Code Chunking**
  • Extracts functions & classes, including decorators & signatures
  • Captures metadata: file path, line numbers, docstrings
- **Vector Store**
  • Uses ChromaDB + HuggingFace Salesforce embeddings for semantic search
  • Splits large code into overlapping chunks for better retrieval
- **LangChain-powered LLM Analysis**
  • System prompt tailored for “senior developer analyzing code”
  • Keeps conversational memory for follow-ups
- **Interactive CLI**
  • Process repos, ask questions, inspect usage stats, clear chat memory

---

## Architecture Overview

```
main.py
 └── CodeAnalyzer
      ├── GitHubIntegration   # clone/download repo contents
      ├── FileUtils           # temp dir management
      ├── CodeChunker         # AST parsing & chunk extraction
      ├── VectorStoreManager  # ChromaDB + text splitting + embeddings
      ├── OpenAIInterface     # wraps OpenAI client + usage logging
      └── LangChainIntegration# conversational LLM querying
```

1. **Chunking**
   • Walk AST, extract `FunctionDef` & `ClassDef` segments
   • Preserve decorators, signature, docstring, line span
2. **Indexing**
   • Add chunks → ChromaDB persistent collection
   • Also build a LangChain “HF” index in `.chromadb_langchain`
3. **Querying**
   - Wraps code in context + system prompt, sends to LLM

---

## Installation

```bash
# 1. Clone and cd into this repository

# 2. Create & activate a virtual env
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
.\venv\Scripts\activate         # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

1. **Set your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="..."
   ```
2. **Run the CLI**
   ```bash
   python main.py
   ```
3. **Process a repository**
   ```
   > process https://github.com/user/repo
   ```
4. **Ask questions**
   ```
   > What does the `process_folder` method do?
   > What is this repository about?
   ```
5. **View usage stats**
   ```
   > stats
   ```
6. **Clear chat memory**
   ```
   > clear
   ```
7. **Exit**
   ```
   > exit
   ```

---

## CLI Usage

When you launch `main.py`, you’ll see:

```
GitHub Agent Console
To begin, type 'process <repository_url>'
To quit,    type 'exit'
```

- **process `<repo_url>`**
  Clone & index the repo. 
- **stats**
  Print total LLM calls & tokens consumed (logged under `logs/openai_usage.log`).
- **clear**
  Clear the interactive LLM memory (LangChain’s ConversationBufferMemory).
- **exit / quit**
  Exit the console.

---

## Code Walkthrough

### Core Modules

#### `analyzer/core/chunk.py` – CodeChunker
- `chunk_file(Path) → List[(code_chunk, metadata)]`
- Uses `ast.parse` & `ast.walk`
- Extracts `FunctionDef` & `ClassDef` nodes
- Builds metadata: `{file, type, name, line, end_line, docstring, signature}`

#### `analyzer/core/vector_stores.py` – VectorStoreManager
- Initializes ChromaDB (`.chromadb`) + SentenceTransformer
- Splits code via `RecursiveCharacterTextSplitter`
- `add_documents`, `query`, `reset_collection`
- `process_folder_for_langchain` builds a separate HF-based Chroma store for LangChain

#### `analyzer/core/analyzer.py` – CodeAnalyzer
- High-level orchestration
- `process_local_folder`, `process_github_repo`
- `smart_query` → semantic search + LangChain integration
- `_requires_analysis` heuristic for LLM vs. raw code

### Integrations

- **GitHubIntegration**
  • `clone_repo`: public vs. private
  • `_download_repo_contents`: only `.py` files
- **OpenAIInterface**
  • Wraps LangChain’s `OpenAI` client
  • Logs usage to `logs/openai_usage.log`
- **LangChainIntegration**
  • `ConversationBufferMemory` + `PromptTemplate`
  • System prompt: *“You are a senior developer analyzing code.”*
  • Builds a chat conversation & calls `client.chat.completions.create(...)`

---

## Configuration & Environment

- **Environment Variables**
  - `OPENAI_API_KEY` – your OpenAI API key
- **Persistence**
  - Chroma store: `./.chromadb`
  - LangChain store: `./.chromadb_langchain`
  - Usage log: `./logs/openai_usage.log`

---

## License

This project is licensed under the [MIT License](LICENSE).
