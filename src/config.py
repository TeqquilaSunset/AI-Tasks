# -*- coding: utf-8 -*-
"""Configuration constants for AI Challenge Task 1."""

import os
from pathlib import Path

# -------------------- PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE_DIR / "saves"
SERVER_SCRIPT = str(BASE_DIR / "mcp_server.py")

# -------------------- OPENAI / LLM CONFIG --------------------
DEFAULT_MODEL = "glm-4.5-air"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 2048

# -------------------- SYSTEM PROMPT --------------------
SYSTEM_PROMPT = (
    "Ты помощник, который помогает с любыми вопросами. "
    "Ты можешь использовать доступные инструменты для получения информации. "
    "Используй только те инструменты которые у тебя есть. "
    "Если существующих инструментов не хватает не напиши пользователю об этом. "
    "Не придумывай новые инструменты. Вызывай инструменты в tool_calls"
)

# -------------------- MCP CONFIG --------------------
DOCKER_MCP_HOST = os.getenv("DOCKER_MCP_HOST", "localhost")
DOCKER_MCP_PORT = int(os.getenv("DOCKER_MCP_PORT", "9011"))
DOCKER_MCP_URL = f"http://{DOCKER_MCP_HOST}:{DOCKER_MCP_PORT}/mcp"

# -------------------- RAG CONFIG --------------------
DEFAULT_COLLECTION_NAME = "pdf_chunks"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:latest"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_RELEVANCE_THRESHOLD = 0.1
DEFAULT_TOP_K = 5

# -------------------- PDF PROCESSING CONFIG --------------------
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_OVERLAP = 50
DEFAULT_MAX_CONCURRENT = 10
DEFAULT_EMBEDDING_TIMEOUT = 30
DEFAULT_BATCH_SIZE = 100

# -------------------- OLLAMA CONFIG --------------------
OLLAMA_NUM_PARALLEL = os.getenv("OLLAMA_NUM_PARALLEL", "10")
OLLAMA_MAX_LOADED_MODELS = os.getenv("OLLAMA_MAX_LOADED_MODELS", "2")
OLLAMA_GPU_MEMORY = os.getenv("OLLAMA_GPU_MEMORY", "")

# -------------------- WEATHER API --------------------
OPEN_WEATHER_API_KEY = os.getenv("OPEN_WEATHER")

# -------------------- CREATE DIRECTORIES --------------------
SAVE_DIR.mkdir(exist_ok=True)
