# -*- coding: utf-8 -*-
"""
Support Chat System - AI-powered customer support

Interactive console application for support agents with:
- RAG-based FAQ and documentation search
- Ticket management (JSON storage)
- Automatic ticket context detection
- AI-powered response suggestions
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from dotenv import load_dotenv

from src.services.ticket_service import TicketService
from src.services.faq_service import FAQService
from src.utils import setup_logging
from src.config import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K
)

# Load environment variables
load_dotenv()

log = setup_logging("support-chat", output_stream="stderr")


class SupportChat:
    """
    Interactive support chat system.

    Features:
    - Ticket management (view, create, update)
    - RAG-based FAQ and documentation search
    - AI-powered response suggestions
    - Automatic ticket context detection
    - User information display
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        tickets_path: str = "data/tickets.json"
    ):
        """
        Initialize support chat system.

        Args:
            model: OpenAI model name
            temperature: LLM temperature
            tickets_path: Path to tickets JSON file
        """
        self.model = model
        self.temperature = temperature

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # Initialize services
        self.ticket_service = TicketService(tickets_path)
        self.faq_service = FAQService(
            collection_name="project_docs",
            threshold=DEFAULT_RELEVANCE_THRESHOLD,
            top_k=DEFAULT_TOP_K
        )

        # Current state
        self.current_ticket_id: Optional[str] = None
        self.current_user_id: Optional[str] = None
        self.conversation_history: List[Dict[str, str]] = []

        log.info("Support chat system initialized")

    def _print_banner(self):
        """Print welcome banner."""
        print("\n" + "=" * 60)
        print("  СИСТЕМА ПОДДЕРЖКИ (Support Chat System)")
        print("=" * 60)
        print("\nДоступные команды:")
        print("  help / ?          - Показать справку")
        print("  tickets           - Список всех тикетов")
        print("  my <user_id>      - Установить текущего пользователя")
        print("  ticket <id>       - Открыть тикет")
        print("  new               - Создать новый тикет")
        print("  stats             - Статистика поддержки")
        print("  search <query>    - Поиск тикетов")
        print("  faq <query>       - Поиск по FAQ")
        print("  clear             - Очистить историю чата")
        print("  quit / exit       - Выход")
        print("\nПримеры сообщений:")
        print('  "Почему не работает авторизация?"')
        print('  "Как проиндексировать PDF файл?"')
        print("=" * 60 + "\n")

    def _print_ticket(self, ticket_id: str):
        """Print ticket details."""
        ticket = self.ticket_service.get_ticket(ticket_id)
        if not ticket:
            print(f"❌ Тикет не найден: {ticket_id}")
            return

        user = self.ticket_service.get_user(ticket.user_id)

        print(f"\n{'=' * 60}")
        print(f"Тикет: #{ticket.id}")
        print(f"Тема: {ticket.subject}")
        print(f"Статус: {ticket.status} | Приоритет: {ticket.priority}")
        print(f"Категория: {ticket.category}")
        print(f"Создан: {ticket.created_at}")
        print(f"Обновлён: {ticket.updated_at}")
        print(f"{'=' * 60}")

        if user:
            print(f"\n👤 Пользователь: {user.name} ({user.email})")
            print(f"   Компания: {user.company}")
            print(f"   Уровень: {user.tier}")

        print(f"\n💬 Сообщений: {len(ticket.messages)}")
        print("-" * 60)

        for msg in ticket.messages:
            role_icon = "👤" if msg["role"] == "user" else "🎧"
            print(f"\n{role_icon} {msg['role'].upper()} [{msg['timestamp']}]")
            print(f"   {msg['content']}")

        print(f"\n{'=' * 60}\n")

    def _print_tickets_list(self, tickets: List, title: str = "Тикеты"):
        """Print list of tickets."""
        if not tickets:
            print(f"\n❌ Нет тикетов для отображения")
            return

        print(f"\n{title} ({len(tickets)}):")
        print("-" * 80)

        for ticket in tickets:
            user = self.ticket_service.get_user(ticket.user_id)
            user_name = user.name if user else "Unknown"

            status_icon = {
                "open": "🔴",
                "in_progress": "🟡",
                "closed": "🟢"
            }.get(ticket.status, "⚪")

            print(f"{status_icon} #{ticket.id} | {ticket.subject}")
            print(f"   👤 {user_name} | 📅 {ticket.updated_at}")
            print(f"   Статус: {ticket.status} | Приоритет: {ticket.priority}")
            print()

    async def _generate_response(
        self,
        user_message: str,
        ticket_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate AI response with RAG context.

        Args:
            user_message: User's message/question
            ticket_context: Optional ticket context

        Returns:
            AI-generated response
        """
        try:
            # Get FAQ context
            faq_data = await self.faq_service.get_answer_suggestion(
                user_message,
                ticket_context
            )

            # Build system prompt
            system_prompt = """Ты - профессиональный агент поддержки технической продукции.
Твоя задача - помогать пользователям решать проблемы с системой.

Правила:
1. Отвечай на русском языке
2. Будь вежливым и эмпатичным
3. Давай четкие и конкретные инструкции
4. Если не уверен в ответе, предложи создать тикет для технической команды
5. Используй предоставленный контекст из FAQ и документации
6. Структурируй ответы с использованием списков и подзаголовков
7. Включай примеры команд или конфигураций где это уместно

При ответе учитывай контекст тикета (если предоставлен)."""

            # Build user prompt with context
            user_prompt = f"Вопрос пользователя: {user_message}\n\n"

            if faq_data.get("context"):
                user_prompt += f"Релевантная информация из документации:\n{faq_data['context']}\n\n"

            if ticket_context:
                user_prompt += f"\nКонтекст тикета: #{ticket_context.get('ticket_id')}\n"

            # Add conversation history
            if self.conversation_history:
                history_text = "\n".join([
                    f"{msg['role']}: {msg['content']}"
                    for msg in self.conversation_history[-5:]  # Last 5 messages
                ])
                user_prompt += f"\nПредыдущая переписка:\n{history_text}\n"

            # Call OpenAI API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )

            ai_response = response.choices[0].message.content.strip()

            # Add to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })

            return ai_response

        except Exception as e:
            log.error(f"Error generating response: {e}")
            return f"Произошла ошибка при генерации ответа: {str(e)}"

    def _detect_ticket_in_message(self, message: str) -> Optional[str]:
        """
        Auto-detect ticket from message.

        Args:
            message: User message

        Returns:
            Ticket ID or None
        """
        # Check if message contains ticket ID
        for ticket in self.ticket_service.get_all_tickets():
            if ticket.id in message or str(ticket.id.split("_")[1]) in message:
                return ticket.id

        # Use auto-detection from service
        if self.current_user_id:
            ticket = self.ticket_service.detect_ticket_from_message(
                message,
                self.current_user_id
            )
            if ticket:
                return ticket.id

        return None

    async def _handle_user_message(self, message: str):
        """
        Handle user message.

        Args:
            message: User's message
        """
        message = message.strip()

        if not message:
            return

        # Check for ticket ID in message
        detected_ticket_id = self._detect_ticket_in_message(message)

        if detected_ticket_id:
            self.current_ticket_id = detected_ticket_id
            print(f"\n🔗 Обнаружен тикет: #{detected_ticket_id}")
            self._print_ticket(detected_ticket_id)

        # Get ticket context if available
        ticket_context = None
        if self.current_ticket_id:
            ticket_context = self.ticket_service.get_ticket_context(self.current_ticket_id)

        # Generate AI response
        print("\n🤖 Генерация ответа...")
        response = await self._generate_response(message, ticket_context)

        print(f"\n🎧 Ответ агента поддержки:")
        print("-" * 60)
        print(response)
        print("-" * 60)

        # Offer to add message to ticket
        if self.current_ticket_id:
            choice = input("\nДобавить ответ в тикет? (y/n): ").strip().lower()
            if choice == 'y' or choice == 'д':
                try:
                    self.ticket_service.add_message(
                        self.current_ticket_id,
                        response,
                        role="agent"
                    )
                    print(f"✅ Ответ добавлен в тикет #{self.current_ticket_id}")
                except Exception as e:
                    print(f"❌ Ошибка добавления сообщения: {e}")

    async def run(self):
        """Run support chat interactive loop."""
        self._print_banner()

        # Show open tickets on start
        open_tickets = self.ticket_service.get_all_tickets(status="open")
        if open_tickets:
            print(f"📋 Открытых тикетов: {len(open_tickets)}")
            self._print_tickets_list(open_tickets[:5], "Последние открытые тикеты")

        while True:
            try:
                # Get user input
                prompt = "\nВы: "
                if self.current_ticket_id:
                    prompt = f"\nВы [#{self.current_ticket_id}]: "

                user_input = input(prompt).strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ["quit", "exit", "выход"]:
                    print("\n👋 До свидания!")
                    break

                elif user_input.lower() in ["help", "?"]:
                    self._print_banner()

                elif user_input.lower() == "tickets":
                    all_tickets = self.ticket_service.get_all_tickets()
                    self._print_tickets_list(all_tickets, "Все тикеты")

                elif user_input.lower() == "new":
                    print("\n📝 Создание нового тикета")
                    user_id = input("ID пользователя: ").strip()

                    subject = input("Тема: ").strip()
                    content = input("Описание проблемы: ").strip()

                    priority = input("Приоритет (low/medium/high) [medium]: ").strip().lower()
                    if not priority:
                        priority = "medium"

                    try:
                        ticket = self.ticket_service.create_ticket(
                            user_id=user_id,
                            subject=subject,
                            content=content,
                            priority=priority
                        )
                        print(f"\n✅ Тикет создан: #{ticket.id}")
                        self.current_ticket_id = ticket.id
                        self._print_ticket(ticket.id)
                    except Exception as e:
                        print(f"\n❌ Ошибка создания тикета: {e}")

                elif user_input.lower().startswith("ticket "):
                    ticket_id = user_input[7:].strip()
                    self.current_ticket_id = ticket_id
                    self._print_ticket(ticket_id)

                elif user_input.lower().startswith("my "):
                    user_id = user_input[3:].strip()
                    self.current_user_id = user_id
                    user = self.ticket_service.get_user(user_id)
                    if user:
                        print(f"\n✅ Текущий пользователь: {user.name} ({user.company})")
                        tickets = self.ticket_service.get_user_tickets(user_id)
                        print(f"   Тикетов: {len(tickets)}")
                    else:
                        print(f"\n❌ Пользователь не найден: {user_id}")

                elif user_input.lower() == "stats":
                    stats = self.ticket_service.get_statistics()
                    print("\n📊 Статистика поддержки:")
                    print("-" * 40)
                    print(f"Пользователей: {stats['total_users']}")
                    print(f"Всего тикетов: {stats['total_tickets']}")
                    print(f"Открыто: {stats['open_tickets']}")
                    print(f"В работе: {stats['in_progress_tickets']}")
                    print(f"Закрыто: {stats['closed_tickets']}")
                    print(f"\nКатегории:")
                    for cat, count in stats['categories'].items():
                        print(f"  - {cat}: {count}")

                elif user_input.lower().startswith("search "):
                    query = user_input[7:].strip()
                    results = self.ticket_service.search_tickets(query)
                    self._print_tickets_list(results, f"Результаты поиска: '{query}'")

                elif user_input.lower().startswith("faq "):
                    query = user_input[4:].strip()
                    print(f"\n🔍 Поиск по FAQ: '{query}'")
                    faq_results = await self.faq_service.search_faq(query)

                    if faq_results:
                        for i, result in enumerate(faq_results, 1):
                            score = result.get("score", 0)
                            text = result.get("text", "")
                            source = result.get("metadata", {}).get("source", "unknown")
                            print(f"\n{i}. [{source}] (релевантность: {score:.2f})")
                            print(f"   {text[:300]}...")
                    else:
                        print("❌ Ничего не найдено")

                elif user_input.lower() == "clear":
                    self.conversation_history.clear()
                    self.current_ticket_id = None
                    print("\n✅ История очищена")

                else:
                    # Handle as user message
                    await self._handle_user_message(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Прервано. До свидания!")
                break
            except Exception as e:
                log.error(f"Error in main loop: {e}")
                print(f"\n❌ Ошибка: {e}")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Support Chat System")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model")
    parser.add_argument("--temp", type=float, default=DEFAULT_TEMPERATURE, help="Temperature")
    parser.add_argument("--tickets", default="data/tickets.json", help="Path to tickets file")

    args = parser.parse_args()

    try:
        chat = SupportChat(
            model=args.model,
            temperature=args.temp,
            tickets_path=args.tickets
        )
        await chat.run()
    except Exception as e:
        log.error(f"Fatal error: {e}")
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
