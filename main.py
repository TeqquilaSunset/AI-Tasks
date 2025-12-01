from zai import ZaiClient
import os
from dotenv import load_dotenv 

load_dotenv()

def main():
    client = ZaiClient(api_key=os.getenv("ZAI_API_KEY"))

    print("=" * 50)
    print("Консольный клиент GLM-4.5-flash. Введите 'quit' для выхода.")
    print("=" * 50)

    conversation = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

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

        # Добавление запроса в историю диалога
        conversation.append({"role": "user", "content": user_input})

        try:
            # Отправка запроса к модели
            response = client.chat.completions.create(
                model="glm-4.5-flash",  # Используемая модель
                messages=conversation,  # История разговора
                temperature=1, 
                max_tokens=1024,
                thinking={ "type": "enabled" }
            )
            
            # Получение и вывод ответа
            ai_response = response.choices[0].message.content
            print(f"\nAI: {ai_response}") 

            # Сохранение ответа в историю для поддержания контекста
            conversation.append({"role": "assistant", "content": ai_response})

        except Exception as e:
            print(f"\n Произошла ошибка: {e}")

if __name__ == "__main__":
    main()