#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI-совместимый чат-клиент с MCP-инструментами
(современный подход с FastMCP и улучшенной архитектурой)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import typing as tp
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

import httpx
import openai
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# --------------------  КОНФИГУРАЦИЯ  --------------------
SYSTEM_PROMPT = "Ты помощник, который помогает с любыми вопросами. Ты можешь использовать доступные инструменты для получения информации."
SAVE_DIR = Path("saves")
SAVE_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = str(BASE_DIR / "mcp_server.py")

# --------------------  ЛОГИРОВАНИЕ  --------------------
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
log = logging.getLogger("mcp-client")

# --------------------  УТИИЛИТЫ  --------------------
def build_openai_client() -> openai.AsyncOpenAI:
    """Создает клиент OpenAI с настройками из окружения."""
    key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_BASE_URL")
    verify = os.getenv("OPENAI_VERIFY_SSL", "true").lower() != "false"
    http = httpx.AsyncClient(verify=verify)
    return openai.AsyncOpenAI(api_key=key, base_url=base, http_client=http)

def save_conversation(history: tp.List[dict], name: str | None = None) -> str:
    """Сохраняет историю разговора в JSON файл."""
    name = f"conversation_{datetime.now():%Y%m%d_%H%M%S}.json" if name is None else name
    if not name.endswith(".json"):
        name += ".json"
    
    path = SAVE_DIR / name
    try:
        path.write_text(json.dumps(history, ensure_ascii=False, indent=2))
        log.info(f"Разговор сохранен в {path}")
        return str(path)
    except Exception as exc:
        error_msg = f"Ошибка при сохранении: {exc}"
        log.error(error_msg)
        return error_msg

def load_conversation(name: str) -> tp.Tuple[tp.List[dict] | None, str]:
    """Загружает историю разговора из файла."""
    try:
        if name.isdigit():
            # Загрузка по номеру
            files = sorted(SAVE_DIR.glob("conversation_*.json"), reverse=True)
            idx = int(name) - 1
            if 0 <= idx < len(files):
                path = files[idx]
                return json.loads(path.read_text()), str(path)
            return None, "Неверный номер сохранения."
        
        # Загрузка по имени
        path = SAVE_DIR / (name if name.endswith(".json") else f"{name}.json")
        if path.exists():
            return json.loads(path.read_text()), str(path)
        return None, f"Файл {path} не найден."
        
    except Exception as exc:
        error_msg = f"Ошибка при загрузке: {exc}"
        log.error(error_msg)
        return None, error_msg

def list_saved_conversations() -> str:
    """Возвращает список сохраненных разговоров."""
    files = sorted(SAVE_DIR.glob("conversation_*.json"), reverse=True)
    if not files:
        return "Нет сохранённых разговоров."
    
    lines = ["Сохранённые разговоры:", "=" * 40]
    for idx, fp in enumerate(files, 1):
        # Извлекаем дату из имени файла
        ts_match = fp.stem.replace("conversation_", "")
        try:
            dt = datetime.strptime(ts_match, "%Y%m%d_%H%M%S")
            nice_date = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            nice_date = fp.stem
        
        lines.append(f"{idx}. {nice_date} – {fp.name}")
    
    return "\n".join(lines)

async def create_summary(cli: openai.AsyncOpenAI, model: str, history: tp.List[dict]) -> str:
    """Создает краткое резюме разговора."""
    msgs = [m for m in history if m["role"] in ("user", "assistant")]
    if not msgs:
        return "Нет истории для резюме."
    
    text = "Пожалуйста, создай краткое резюме следующего диалога. Выдели основные темы и детали:\n\n"
    for msg in msgs:
        role = "Пользователь" if msg["role"] == "user" else "AI"
        text += f"{role}: {msg['content']}\n\n"
    
    try:
        resp = await cli.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            temperature=0.3,
            max_tokens=512,
        )
        return resp.choices[0].message.content or "Не удалось создать резюме."
    except Exception as exc:
        error_msg = f"Ошибка при создании резюме: {exc}"
        log.error(error_msg)
        return error_msg

# --------------------  MCP КЛИЕНТ  --------------------
class MCPClient:
    """Современный MCP клиент с улучшенным управлением ресурсами."""
    
    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.tools: list[dict] = []
        self._running = False

    async def connect_to_server(self, server_script_path: str) -> None:
        """Подключается к MCP серверу."""
        log.info(f"Подключение к серверу: {server_script_path}")
        
        if not Path(server_script_path).exists():
            raise FileNotFoundError(f"Сервер не найден: {server_script_path}")
        
        # Параметры для запуска сервера
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script_path],
            env={**os.environ}
        )
        
        # Создаем транспорт и сессию через контекстный менеджер
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                stdio_transport[0], 
                stdio_transport[1],
                client_info={"name": "mcp-client", "version": "1.0.0"}
            )
        )
        
        # Инициализируем сессию
        await self.session.initialize()
        
        # Получаем список инструментов
        tools_result = await self.session.list_tools()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                }
            }
            for tool in (tools_result.tools if tools_result else [])
        ]
        
        self._running = True
        log.info(f"Подключено к серверу. Доступно инструментов: {len(self.tools)}")
        
        if self.tools:
            for tool in self.tools:
                log.info(f"  - {tool['function']['name']}: {tool['function']['description']}")

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Вызывает инструмент MCP сервера."""
        if not self._running or not self.session:
            return "[Ошибка] Сервер не подключен"
        
        try:
            result = await self.session.call_tool(name, arguments)
            
            # Объединяем все текстовые блоки в один ответ
            text_parts = []
            for block in result.content or []:
                if hasattr(block, 'text'):
                    text_parts.append(block.text)
            
            return "\n".join(text_parts) if text_parts else "Инструмент выполнен без результата"
            
        except Exception as exc:
            error_msg = f"Ошибка вызова инструмента {name}: {exc}"
            log.error(error_msg)
            return f"[Ошибка] {error_msg}"

    async def cleanup(self) -> None:
        """Освобождает ресурсы."""
        if self._running:
            self._running = False
            await self.exit_stack.aclose()
            log.info("Ресурсы MCP клиента освобождены")

    @property
    def available_tools(self) -> list[dict]:
        """Возвращает список доступных инструментов."""
        return self.tools

# --------------------  ЧАТ КЛИЕНТ  --------------------
class ChatClient:
    """Основной чат-клиент с интеграцией MCP."""
    
    def __init__(self, model_name: str = "glm-4.5-air") -> None:
        self.model_name = model_name
        self.openai_client = build_openai_client()
        self.mcp_client = MCPClient()
        self.conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.temperature = 0.7

    async def process_query(self, query: str) -> str:
        """Обрабатывает запрос пользователя с использованием доступных инструментов."""
        log.info(f"Обработка запроса: {query[:50]}...")
        
        try:
            # Делаем запрос к OpenAI с доступными инструментами
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation + [{"role": "user", "content": query}],
                tools=self.mcp_client.available_tools or None,
                tool_choice="auto" if self.mcp_client.available_tools else None,
                temperature=self.temperature,
                max_tokens=2048,
            )
            
            assistant_message = response.choices[0].message
            content = assistant_message.content or ""
            
            # Проверяем, есть ли tool calls
            if assistant_message.tool_calls:
                log.info(f"Обнаружены вызовы инструментов: {[tc.function.name for tc in assistant_message.tool_calls]}")
                
                # Добавляем сообщение ассистента
                self.conversation.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Выполняем все tool calls
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    log.info(f"Вызов инструмента: {tool_name} с аргументами: {tool_args}")
                    
                    # Вызываем инструмент MCP
                    tool_result = await self.mcp_client.call_tool(tool_name, tool_args)
                    
                    log.info(f"Результат инструмента {tool_name}: {tool_result[:100]}...")
                    
                    # Добавляем результат в контекст
                    self.conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                
                # Получаем финальный ответ
                final_response = await self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=self.conversation,
                    temperature=self.temperature,
                    max_tokens=2048,
                )
                
                content = final_response.choices[0].message.content or ""
            
            return content
            
        except Exception as exc:
            error_msg = f"Ошибка при обработке запроса: {exc}"
            log.error(error_msg)
            return f"Извините, произошла ошибка: {exc}"

    async def start(self, server_script: str) -> None:
        """Запускает клиента и подключается к MCP серверу."""
        log.info("Запуск чат-клиента...")
        
        # Подключаемся к MCP серверу
        await self.mcp_client.connect_to_server(server_script)
        
        log.info("Чат-клиент готов к работе!")

    async def cleanup(self) -> None:
        """Освобождает ресурсы."""
        await self.mcp_client.cleanup()
        await self.openai_client.close()
        log.info("Клиент остановлен")

    def add_message(self, role: str, content: str) -> None:
        """Добавляет сообщение в историю."""
        self.conversation.append({"role": role, "content": content})

# --------------------  ИНТЕРАКТИВНЫЙ ИНТЕРФЕЙС  --------------------
async def interactive_chat(client: ChatClient) -> None:
    """Интерактивный чат-цикл."""
    print("=" * 60)
    print("🤖 Чат-клиент с MCP инструментами")
    print("Команды: quit/exit, save <имя>, load <имя>, temp <0-2>, clear, print")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 До свидания!")
            break

        if not user_input:
            continue

        # Обработка команд
        if user_input.lower() in ("quit", "exit"):
            print("👋 До свидания!")
            break

        if user_input.lower() == "clear":
            client.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("🗑️ История очищена")
            continue

        if user_input.lower() == "print":
            print("=" * 50, "📋 История разговора:", "=" * 50, sep="\n")
            for i, msg in enumerate(client.conversation[1:], 1):
                print(f"{i}. {msg['role'].upper()}: {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")
            print("=" * 50)
            continue

        if user_input.lower().startswith("save "):
            name = user_input[5:].strip()
            path = save_conversation(client.conversation, name)
            print(f"💾 Сохранено: {path}")
            continue

        if user_input.lower().startswith("load "):
            name = user_input[5:].strip()
            loaded, msg = load_conversation(name)
            if loaded:
                client.conversation = loaded
                print(f"📂 Загружено: {msg}")
            else:
                print(f"❌ {msg}")
            continue

        if user_input.lower().startswith("temp "):
            try:
                temp = float(user_input[5:].strip())
                if 0.0 <= temp <= 2.0:
                    client.temperature = temp
                    print(f"🌡️ Температура установлена: {temp}")
                else:
                    print("⚠️ Температура должна быть от 0 до 2")
            except ValueError:
                print("⚠️ Пример: temp 0.7")
            continue

        # Обработка обычного запроса
        try:
            start_time = datetime.now()
            
            # Добавляем сообщение пользователя
            client.add_message("user", user_input)
            
            # Обрабатываем запрос
            response = await client.process_query(user_input)
            
            # Добавляем ответ ассистента
            client.add_message("assistant", response)
            
            # Выводим ответ
            print(f"\n🤖 Ассистент: {response}")
            
            # Статистика
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"⏱️ Время: {elapsed:.2f}с | Инструментов: {len(client.mcp_client.available_tools)}")
            
        except Exception as exc:
            error_msg = f"❌ Ошибка: {exc}"
            log.error(error_msg)
            print(error_msg)

# --------------------  ГЛАВНАЯ ФУНКЦИЯ  --------------------
async def main() -> None:
    """Главная функция запуска."""
    # Создаем клиента
    client = ChatClient(model_name="glm-4.5-air")
    
    try:
        # Подключаемся к MCP серверу
        await client.start(SERVER_SCRIPT)
        
        # Запускаем интерактивный чат
        await interactive_chat(client)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Прервано пользователем")
    except Exception as exc:
        error_msg = f"💥 Критическая ошибка: {exc}"
        log.exception(error_msg)
        print(error_msg)
    finally:
        # Освобождаем ресурсы
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())