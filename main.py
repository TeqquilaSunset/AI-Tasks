from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime
import httpx
import json
import glob
from typing import List, Dict
import re

load_dotenv()

SYSTEM_PROMPT = "Ты помощник, который помогает с любыми вопросами"

def save_conversation(conversation_history: List[Dict], filename: str = None):
    """Save the current conversation history to a JSON file"""
    if not os.path.exists("saves"):
        os.makedirs("saves")

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saves/conversation_{timestamp}.json"
    else:
        if not filename.endswith('.json'):
            filename = f"saves/{filename}.json"
        else:
            filename = f"saves/{filename}"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, indent=2, ensure_ascii=False)
        return filename
    except Exception as e:
        return f"Ошибка при сохранении: {e}"

def list_saved_conversations():
    """List all saved conversations with their IDs"""
    save_files = glob.glob("saves/conversation_*.json")
    save_files.sort(reverse=True)  # Most recent first

    if not save_files:
        return "Нет сохраненных разговоров."

    result = "Список сохраненных разговоров:\n"
    result += "="*40 + "\n"

    for i, file in enumerate(save_files):
        # Extract timestamp from filename
        match = re.search(r'conversation_(\d{8}_\d{6})', file)
        if match:
            timestamp = match.group(1)
            # Format timestamp nicely
            formatted_time = f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
        else:
            formatted_time = os.path.basename(file)

        result += f"{i+1}. [{i+1}] {formatted_time} - {os.path.basename(file)}\n"

    return result

def load_conversation(save_id: str):
    """Load a conversation from a saved file by ID"""
    save_files = glob.glob("saves/conversation_*.json")
    save_files.sort(reverse=True)  # Most recent first

    try:
        # Convert save_id to integer index
        index = int(save_id) - 1

        if 0 <= index < len(save_files):
            filename = save_files[index]
            with open(filename, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
            return conversation, filename
        else:
            return None, "Неверный ID сохранения."
    except ValueError:
        return None, "ID сохранения должен быть числом."
    except Exception as e:
        return None, f"Ошибка при загрузке: {e}"

def create_summary_with_llm(client, model_name, conversation_history):
    """Create a summary of all previous user requests and AI responses using LLM"""
    # Exclude the system prompt (index 0) and only include user and assistant messages
    user_ai_messages = [msg for msg in conversation_history[1:] if msg["role"] in ["user", "assistant"]]

    if len(user_ai_messages) == 0:
        return "Нет истории диалога для создания резюме."

    # Format the conversation history for the LLM to summarize
    formatted_history = "Пожалуйста, создай краткое резюме следующей истории диалога. Выдели основные темы и детали диалога, которые могут понадобится при дальнейшем общении:\n\n"
    for i, message in enumerate(user_ai_messages):
        role = "Пользователь" if message["role"] == "user" else "AI"
        content = message["content"]
        formatted_history += f"{role}: {content}\n\n"

    try:
        # Send the formatted history to the LLM for summarization
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": formatted_history}],
            temperature=0.3,
            max_tokens=2048
        )

        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        return f"Ошибка при создании резюме: {e}"

def print_context(system_prompt, conversation_history):
    """Print the full context (system prompt + conversation history)"""
    print("="*50)
    print("ПОЛНЫЙ КОНТЕКСТ:")
    print("="*50)
    print(f"Системный промпт: {system_prompt}")
    print("-"*50)
    print("История разговора:")

    for i, message in enumerate(conversation_history[1:], 1):  # Skip system prompt
        role = message["role"].upper()
        content = message["content"]
        print(f"{i}. {role}: {content}")

    print("="*50)

def main():
    # Initialize OpenAI client
    # You can use either OpenAI API or an OpenAI-compatible service

    # Z.AI
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = "glm-4.5-air"

    # Disable SSL certificate verification
    http_client = httpx.Client(verify=False)

    if base_url:
        client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    else:
        client = OpenAI(api_key=api_key, http_client=http_client)

    print("=" * 50)
    print("Консольный клиент OpenAI-совместимой модели. Введите 'quit' для выхода.")
    print("=" * 50)

    conversation = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"}
    ]

    temp = 1.0
    print(f"Используемая модель: {model_name}")

    while True:
        # Получение запроса от пользователя
        try:
            user_input = input("\nВы: ")
        except (EOFError, KeyboardInterrupt):
            print("\n\nВыход.")
            break

        if user_input.lower() == 'quit':
            print("До свидания!")
            break

        # Check if the user wants to change the temperature
        if user_input.lower().startswith('temp '):
            try:
                # Extract the numeric value after 'temp '
                temp_value = float(user_input[5:].strip())  # Skip 'temp ' (5 characters) and get the number
                if 0.0 <= temp_value <= 2.0:  # Validate temperature range
                    temp = temp_value
                    print(f"Температура установлена на {temp}")
                    continue  # Skip to the next iteration without sending to AI
                else:
                    print("Температура должна быть в диапазоне от 0.0 до 2.0")
                    continue
            except ValueError:
                print("Пожалуйста, укажите числовое значение для температуры, например: temp 0.7")
                continue

        # Check if the user wants to print the full context
        if user_input.lower() == 'print':
            print_context(SYSTEM_PROMPT, conversation)
            continue  # Skip to the next iteration without sending to AI

        # Check if the user wants to get a summary of previous requests
        if user_input.lower() == 'summary':
            print("Создание резюме предыдущей истории...")
            summary = create_summary_with_llm(client, model_name, conversation)
            print(f"\nРезюме: {summary}")

            # Replace conversation history with system prompt and summary only
            system_message = conversation[0]  # Keep the system prompt
            summary_message = f"Суммаризация предыдущего разговора: {summary}"
            conversation = [system_message, {"role": "assistant", "content": summary_message}]
            continue  # Skip to the next iteration without sending to AI

        # Check if the user wants to save the conversation
        if user_input.lower() == 'save':
            filename = save_conversation(conversation)
            if filename.startswith("Ошибка"):
                print(f"\n{filename}")
            else:
                print(f"\nИстория разговора сохранена в {filename}")
            continue  # Skip to the next iteration without sending to AI

        # Check if the user wants to list saved conversations
        if user_input.lower() == 'load':
            saved_list = list_saved_conversations()
            print(f"\n{saved_list}")
            continue  # Skip to the next iteration without sending to AI

        # Check if the user wants to load a specific conversation (format: "load id")
        if user_input.lower().startswith('load '):
            parts = user_input.split()
            if len(parts) == 2:
                save_id = parts[1]
                loaded_conversation, result = load_conversation(save_id)
                if loaded_conversation is not None:
                    conversation = loaded_conversation  # Replace current conversation
                    print(f"\nИстория разговора загружена из {result}")
                else:
                    print(f"\n{result}")
            else:
                print("\nПожалуйста, укажите ID сохранения. Пример: load 1")
            continue  # Skip to the next iteration without sending to AI

        # Добавление запроса в историю диалога
        conversation.append({"role": "user", "content": user_input})

        try:
            # Record start time for request
            start_time = datetime.now()

            # Отправка запроса к модели
            response = client.chat.completions.create(
                model=model_name,  # Используемая модель
                messages=conversation,  # История разговора
                temperature=temp,
                max_tokens=2048,
                # Note: The thinking parameter from GLM is removed as it's not compatible with OpenAI interface
            )

            # Получение и вывод ответа
            ai_response = response.choices[0].message.content

            conversation.append({"role": "assistant", "content": ai_response})

            # Calculate request time
            end_time = datetime.now()
            request_duration = end_time - start_time

            # Get token usage information
            usage_info = response.usage
            tokens_prompt = usage_info.prompt_tokens if hasattr(usage_info, 'prompt_tokens') else "N/A"
            tokens_completion = usage_info.completion_tokens if hasattr(usage_info, 'completion_tokens') else "N/A"
            tokens_total = usage_info.total_tokens if hasattr(usage_info, 'total_tokens') else "N/A"

            # Print the raw AI response and additional information
            print(f"\nTemperature: {temp}")
            print(f"\nAI: {ai_response}")

            # Print request time and token usage
            print(f"\n--- Справочная информация ---")
            print(f"Время запроса: {request_duration.total_seconds():.2f} секунд")
            print(f"Расход токенов:")
            print(f"  - Вопрос: {tokens_prompt}")
            print(f"  - Ответ: {tokens_completion}")
            print(f"  - Всего: {tokens_total}")
            print(f"-----------------------------")

        except Exception as e:
            print(f"\n Произошла ошибка: {e}")

if __name__ == "__main__":
    main()
