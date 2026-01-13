# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Challenge Task 1 project - an OpenAI-compatible chat client with MCP (Model Context Protocol) integration, RAG (Retrieval Augmented Generation), and PDF document processing capabilities. The project is written in Python and uses a modular architecture with utilities, services, and clients.

**Key Technologies:**
- MCP (Model Context Protocol) for tool integration
- FastMCP for MCP server implementation
- OpenAI API for LLM interactions
- Qdrant for vector database (RAG)
- Ollama for embeddings generation
- Docker for containerization

## Development Commands

### Environment Setup
```bash
# Install dependencies (Windows with virtualenv)
pip install -r requirements.txt

# Activate virtual environment (if needed)
.venv\Scripts\activate

# Start services
docker run -p 6333:6333 qdrant/qdrant
ollama serve  # For RAG and PDF pipeline embeddings
```

### Running the Application
```bash
# Main chat client
python main.py

# Index project documentation (required before using /help commands)
python index_docs.py

# Index code for RAG
python index_code.py

# PDF processing pipelines
python pdf_pipeline.py --pdf_path document.pdf
python pdf_pipeline_async.py --pdf_path document.pdf --max_concurrent 10
python run_pipeline_gpu.py --pdf_path document.pdf --use_async --max_concurrent 15

# MCP server (runs as STDIO server)
python mcp_server.py

# PR reviewer
python pr_reviewer.py
```

### Testing
```bash
# RAG tests
python test_rag_changes.py
```

## Architecture

### Modular Structure (Post-Refactoring)

The codebase follows a layered architecture:

```
src/
├── config.py                 # Centralized configuration and constants
├── utils/                    # Shared utilities
│   ├── logging_config.py     # Logging setup
│   ├── pdf_chunker.py        # PDF chunking utilities
│   ├── docs_chunker.py       # Documentation chunking
│   ├── ollama_client.py      # Ollama embedding client
│   └── qdrant_client.py      # Qdrant vector DB client
├── services/                 # Business logic
│   ├── rag_service.py        # RAG search and retrieval
│   └── help_service.py       # Help/documentation service
└── clients/                  # MCP client implementations
    ├── mcp_client.py         # STDIO transport MCP client
    └── docker_mcp_client.py  # HTTP transport MCP client
```

### Key Components

**main.py**: Chat client with MCP tool integration
- Manages conversation history
- Handles user commands (quit, save, load, temp, rag, help)
- Integrates MCP tools from both STDIO and Docker servers
- Uses OpenAI API with configurable base URL

**mcp_server.py**: MCP server (FastMCP-based)
- Provides weather tools via OpenWeatherMap API
- Git tools for repository operations
- STDIO transport for local communication
- All logging goes to stderr (MCP requirement)

**RAG Service** (`src/services/rag_service.py`):
- Vector similarity search using Qdrant
- Embedding generation via Ollama
- Configurable relevance threshold
- Supports reranking

**Help Service** (`src/services/help_service.py`):
- Project documentation search via RAG
- Git context integration
- Code style and API reference

### MCP Tools Available

**Weather Tools** (from mcp_server.py):
- `get_weather(city_name)` - Current weather
- `get_weather_forecast(city_name, days)` - Multi-day forecast
- `convert_temperature(value, from_unit, to_unit)` - Unit conversion
- `save_weather_data(city_name, data)` - Save to file

**Git Tools** (from mcp_server.py):
- `git_get_current_branch()` - Current branch name
- `git_get_branches()` - List all branches
- `git_get_status()` - Working tree status
- `git_get_diff()` - Show changes (diff)
- `git_get_recent_commits(count)` - Recent commits
- `git_get_file_content(file_path)` - Read file from git
- `git_list_files()` - List repository files

### Environment Variables (.env)

Required variables:
```
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.example.com/v1
OPEN_WEATHER=your_openweather_api_key
DOCKER_MCP_HOST=localhost
DOCKER_MCP_PORT=9011
```

Optional variables:
```
OPENAI_VERIFY_SSL=false  # Disable SSL verification if needed
OLLAMA_NUM_PARALLEL=10   # For GPU optimization
OLLAMA_MAX_LOADED_MODELS=2
OLLAMA_GPU_MEMORY=10240  # GPU memory limit in MB
```

### Configuration

All constants are centralized in `src/config.py`:
- Default model: `glm-4.5-air`
- Default temperature: `0.7`
- RAG settings: threshold (0.1), top_k (5)
- Qdrant: localhost:6333
- Ollama: http://localhost:11434
- Default embedding model: `qwen3-embedding:latest`

## Chat Commands

### Basic Commands
- `quit` / `exit` - Exit the chat
- `save [name]` - Save conversation to JSON
- `load [name]` - Load conversation by name or number
- `temp <0-2>` - Set LLM temperature
- `clear` - Clear conversation history
- `print` - Display conversation history

### RAG Commands
- `/rag` - Toggle RAG mode on/off
- `/rag <question>` - Ask a question with RAG context
- `/rag rerank` - Toggle result reranking
- `/rag threshold <0-1>` - Set relevance threshold

### Help Commands
- `/help` - General project help
- `/help <question>` - Search documentation
- `/help style` - Code style guidelines
- `/help api [component]` - API reference
- `/help structure` - Project structure
- `/help git` - Current git context

## Important Patterns

### MCP Client Usage

The chat client connects to two MCP servers:

1. **STDIO Server** (mcp_server.py): Local server via subprocess
   - Started automatically when chat client runs
   - Weather and Git tools
   - Communication via stdin/stdout

2. **Docker Server** (optional): Remote MCP server via HTTP
   - Configured via DOCKER_MCP_HOST/PORT
   - HTTP transport

```python
from src.clients import MCPClient, DockerMCPClient

# STDIO client
client = MCPClient()
await client.connect_to_server("mcp_server.py")
tools = await client.list_tools()

# Docker client
docker_client = DockerMCPClient()
await docker_client.connect("http://localhost:9011/mcp")
```

### Logging Pattern

All MCP servers MUST log to stderr only (stdout reserved for MCP protocol):

```python
from src.utils import setup_logging

log = setup_logging("server-name", output_stream="stderr")
```

### Adding New MCP Tools

To add a new tool in `mcp_server.py`:

```python
@mcp.tool()
async def my_tool(param: str) -> str:
    """Tool description for LLM."""
    # Implementation
    return result
```

### RAG Usage

```python
from src.services import RAGService

rag = RAGService(collection_name="pdf_chunks")
results = await rag.search("query text")
context = rag.format_context(results)
```

## PDF Pipeline

The project has three versions for different use cases:

1. **pdf_pipeline.py** - Synchronous version
2. **pdf_pipeline_async.py** - Asynchronous with semaphore control
3. **run_pipeline_gpu.py** - GPU-optimized wrapper

For RTX 4070 GPU:
```bash
python run_pipeline_gpu.py --pdf_path doc.pdf --use_async --max_concurrent 15 --chunk_size 2048
```

## Code Style Notes

- Type hints are used throughout (Python 3.10+)
- Docstrings in English for code, Russian for user messages
- UTF-8 encoding required (`# -*- coding: utf-8 -*-`)
- Async/await for I/O operations
- Structured logging via `src/utils/logging_config.py`

## Common Pitfalls

1. **MCP Server Logging**: Never write to stdout in MCP servers - use stderr
2. **Ollama Connection**: Ensure Ollama is running before using RAG features
3. **Qdrant Collections**: Run `index_docs.py` before using `/help` commands
4. **Async Context**: Most I/O operations are async - use `await`
5. **Path Handling**: Use `pathlib.Path` for cross-platform compatibility
