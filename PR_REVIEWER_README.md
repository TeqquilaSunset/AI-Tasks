# AI PR Reviewer - CI Pipeline

Автоматическая система ревью Pull Request с использованием RAG и MCP.

## Обзор

Система анализирует PR и генерирует ревью с замечаниями, используя:

- **RAG (Retrieval Augmented Generation)**: Поиск по документации и коду проекта
- **MCP (Model Context Protocol)**: Получение данных из GitHub API
- **LLM**: Генерация структурированного ревью

## Архитектура

```
┌─────────────────┐
│  GitHub PR Event│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  GitHub Actions Workflow        │
│  - Checkout code                │
│  - Setup Python & dependencies  │
│  - Start Qdrant                │
│  - Index documentation          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  pr_reviewer.py                 │
│  - Get PR info via GitHub API   │
│  - Get diff                     │
│  - Search docs via RAG          │
│  - Search code via RAG          │
│  - Generate review with LLM     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Post comment to PR             │
└─────────────────────────────────┘
```

## Компоненты

### 1. GitHub Actions Workflow (`.github/workflows/pr-review.yml`)

Запускается автоматически при создании или обновлении PR:

- Устанавливает зависимости
- Запускает Qdrant (векторная БД)
- Индексирует документацию проекта
- Запускает скрипт ревью
- Публикует результат в PR

### 2. PR Reviewer Script (`pr_reviewer.py`)

Основной скрипт для анализа PR:

**Функции**:
- Получение информации о PR через GitHub API
- Извлечение diff изменений
- Поиск релевантной документации через RAG
- Поиск релевантного кода через RAG
- Генерация структурированного ревью
- Публикация комментария в PR

**Структура ревью**:
1. **Summary**: Краткий обзор изменений
2. **Strengths**: Что сделано хорошо
3. **Concerns**: Потенциальные проблемы
4. **Suggestions**: Конкретные улучшения
5. **Documentation**: Проверка документации
6. **Testing**: Проверка тестов

### 3. GitHub MCP Server (`github_mcp_server.py`)

MCP сервер для работы с GitHub API:

**Инструменты**:
- `get_pull_request`: Информация о PR
- `get_pr_diff`: Diff изменений
- `get_pr_files`: Список измененных файлов
- `get_file_content`: Содержимое файла
- `get_pr_commits`: Коммиты PR
- `create_pr_comment`: Создание комментария
- `get_repo_info`: Информация о репозитории

### 4. Code Indexer (`index_code.py`)

Индексация кода проекта для RAG:

- Извлечение функций и классов из Python файлов
- Создание чанков с контекстом
- Генерация эмбеддингов через Ollama
- Сохранение в Qdrant

## Установка

### 1. Настройка GitHub Secrets

Добавьте в репозиторий следующие secrets:

```
OPENAI_API_KEY          # API ключ для LLM
OPENAI_BASE_URL         # (опционально) URL OpenAI-совместимого API
```

`GITHUB_TOKEN` создается автоматически в GitHub Actions.

### 2. Локальная настройка (для тестирования)

```bash
# Установка зависимостей
pip install -r requirements.txt

# Настройка .env
cat > .env << EOF
GITHUB_TOKEN=ghp_your_token_here
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.example.com/v1  # опционально
EOF

# Запуск Qdrant
docker run -d -p 6333:6333 qdrant/qdrant:latest

# Индексация документации
python index_docs.py

# Индексация кода
python index_code.py
```

### 3. Локальный запуск ревью

```bash
python pr_reviewer.py
```

Переменные окружения:
- `GITHUB_TOKEN`: GitHub API токен
- `OPENAI_API_KEY`: API ключ для LLM
- `OPENAI_BASE_URL`: (опционально) URL API
- `PR_NUMBER`: Номер PR для анализа
- `REPO_NAME`: Имя репозитория (owner/repo)

## Использование MCP с GitHub MCP Server

### Запуск MCP сервера

```bash
python github_mcp_server.py
```

### Использование в main.py

Добавьте в `main.py`:

```python
from src.clients import MCPClient

class ChatClient:
    def __init__(self):
        self.github_mcp_client = MCPClient()

    async def start(self):
        await self.github_mcp_client.connect_to_server("github_mcp_server.py")
```

Доступные инструменты:
- `/github_pr_info` - информация о PR
- `/github_pr_diff` - получить diff
- `/github_pr_files` - список файлов
- `/github_get_file` - содержимое файла

## RAG для документации

### Индексация документации

```bash
python index_docs.py
```

Индексирует:
- README.md
- Документацию в папке `docs/`
- Jupyter notebooks (.ipynb)
- Markdown файлы

### Индексация кода

```bash
python index_code.py
```

Индексирует:
- Все .py файлы в проекте
- Разбивает на функции и классы
- Сохраняет с контекстом (docstring, файл, строка)

## Результат работы

### Пример ревью

```markdown
# PR Review: #42 - Add new feature

**Generated:** 2025-01-13 12:34:56 UTC

## Summary

This PR adds a new feature for user authentication...

## Strengths

- Clean code structure
- Good documentation
- Comprehensive tests

## Concerns

- Line 45: Potential null pointer exception
- Missing input validation

## Suggestions

1. Consider using async/await for database operations
2. Add type hints for better IDE support
3. Extract magic numbers to constants

## Documentation

✅ README updated
✅ Docstrings present
⚠️  API documentation needs update

## Testing

✅ Unit tests added
⚠️  Integration tests missing
```

## Конфигурация

### Параметры RAG

В `src/config.py`:

```python
DEFAULT_RELEVANCE_THRESHOLD = 0.5  # Порог релевантности для поиска
DEFAULT_TOP_K = 5                  # Количество результатов
```

### Параметры LLM

В `pr_reviewer.py`:

```python
system_prompt = """You are an expert code reviewer..."""
temperature = 0.3
max_tokens = 4096
```

## Troubleshooting

### Ошибка "Missing environment variables"

Проверьте, что все необходимые secrets добавлены в GitHub.

### Ошибка "Qdrant connection failed"

Убедитесь, что Qdrant запущен:
```bash
curl http://localhost:6333/health
```

### Пустое ревью

Проверьте, что документация и код проиндексированы:
```bash
# Просмотр коллекций
curl http://localhost:6333/collections
```

## Продвинутое использование

### Кастомный промпт для ревью

Отредактируйте `system_prompt` в `pr_reviewer.py` для изменения стиля ревью.

### Дополнительные правила проверки

Добавьте кастомную логику в метод `_generate_review`:
```python
# Пример: проверка на секреты
if re.search(r'password|secret|token', diff, re.IGNORECASE):
    review += "\n⚠️  **Security Warning**: Possible hardcoded secrets detected"
```

### Интеграция с другими MCP серверами

Можно использовать несколько MCP серверов одновременно:
```python
await self.mcp_client.connect_to_server("github_mcp_server.py")
await self.mcp_client.connect_to_server("other_mcp_server.py")
```

## Лицензия

MIT
