#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP-сервер с погодой на основе FastMCP (современный подход)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any

import httpx
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# --------------------  ЛОГИРОВАНИЕ  --------------------
# Важно: пишем только в stderr для STDIO-based серверов
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)]
)
log = logging.getLogger("weather-server")

load_dotenv()

API_KEY = os.getenv("OPEN_WEATHER")
if not API_KEY:
    log.error("OPEN_WEATHER не найден в окружении!")
    sys.exit(1)
else:
    log.info(f"OPEN_WEATHER загружен: {API_KEY[:5]}...")

# --------------------  FASTMCP СЕРВЕР  --------------------
# Инициализируем FastMCP сервер
mcp = FastMCP("weather")

# --------------------  ПОГОДНЫЕ ИНСТРУМЕНТЫ  --------------------
@mcp.tool()
async def get_weather(city_name: str) -> str:
    """Получить текущую погоду для города.
    
    Args:
        city_name: Название города (например, "Moscow", "London")
    """
    log.info(f"Запрашиваю погоду для города: {city_name}")
    
    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?q={city_name}&appid={API_KEY}&units=metric&lang=ru"
    )
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json()
        
        log.info(f"Погода получена: {data.get('name', city_name)} "
                 f"{data['main']['temp']}°C, {data['weather'][0]['description']}")
        
        return (
            f"Погода в {data['name']}, {data['sys']['country']}:\n"
            f"- Состояние: {data['weather'][0]['description'].title()}\n"
            f"- Температура: {data['main']['temp']:.1f}°C (ощущается как {data['main']['feels_like']:.1f}°C)\n"
            f"- Min/Max: {data['main']['temp_min']:.1f}°C / {data['main']['temp_max']:.1f}°C\n"
            f"- Влажность: {data['main']['humidity']}%\n"
            f"- Ветер: {data['wind']['speed']} м/с\n"
            f"- Давление: {data['main']['pressure']} гПа"
        )
        
    except httpx.HTTPStatusError as e:
        error_msg = f"Ошибка OpenWeather: HTTP {e.response.status_code}"
        if e.response.status_code == 404:
            error_msg += f" - Город '{city_name}' не найден"
        else:
            error_msg += f" - {e.response.text}"
        log.error(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"Ошибка при получении погоды: {e}"
        log.exception(error_msg)
        return error_msg

@mcp.tool()
async def get_weather_forecast(city_name: str, days: int = 3) -> str:
    """Получить прогноз погоды на несколько дней.
    
    Args:
        city_name: Название города
        days: Количество дней для прогноза (по умолчанию 3, максимум 5)
    """
    days = min(max(days, 1), 5)  # Ограничиваем от 1 до 5 дней
    
    log.info(f"Запрашиваю прогноз для {city_name} на {days} дней")
    
    # Сначала получаем координаты города
    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?q={city_name}&appid={API_KEY}&units=metric"
    )
    
    try:
        async with httpx.AsyncClient() as client:
            # Получаем текущую погоду для координат
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            current_data = response.json()
            
            lat, lon = current_data['coord']['lat'], current_data['coord']['lon']
            
            # Теперь получаем прогноз по координатам
            forecast_url = (
                "https://api.openweathermap.org/data/2.5/forecast"
                f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&lang=ru"
            )
            
            forecast_response = await client.get(forecast_url, timeout=30.0)
            forecast_response.raise_for_status()
            forecast_data = forecast_response.json()
        
        log.info(f"Прогноз получен для {current_data['name']}")
        
        # Группируем прогноз по дням
        daily_forecasts = {}
        for item in forecast_data['list']:
            date = item['dt_txt'].split(' ')[0]
            if date not in daily_forecasts:
                daily_forecasts[date] = []
            daily_forecasts[date].append(item)
        
        # Формируем ответ
        result = f"Прогноз погоды в {current_data['name']} на {days} дней:\n\n"
        
        count = 0
        for date, forecasts in daily_forecasts.items():
            if count >= days:
                break
            
            # Берем прогноз на полдень для каждого дня
            midday_forecast = None
            for forecast in forecasts:
                if '12:00:00' in forecast['dt_txt']:
                    midday_forecast = forecast
                    break
            
            if not midday_forecast and forecasts:
                midday_forecast = forecasts[len(forecasts)//2]
            
            if midday_forecast:
                result += (
                    f"{midday_forecast['dt_txt'][:10]}:\n"
                    f"- {midday_forecast['weather'][0]['description'].title()}\n"
                    f"- Температура: {midday_forecast['main']['temp']:.1f}°C\n"
                    f"- Ощущается как: {midday_forecast['main']['feels_like']:.1f}°C\n"
                    f"- Влажность: {midday_forecast['main']['humidity']}%\n"
                    f"- Ветер: {midday_forecast['wind']['speed']} м/с\n\n"
                )
                count += 1
        
        return result.strip()
        
    except httpx.HTTPStatusError as e:
        error_msg = f"Ошибка OpenWeather: HTTP {e.response.status_code}"
        if e.response.status_code == 404:
            error_msg += f" - Город '{city_name}' не найден"
        log.error(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"Ошибка при получении прогноза: {e}"
        log.exception(error_msg)
        return error_msg

@mcp.tool()
async def convert_temperature(value: float, from_unit: str = "celsius", to_unit: str = "fahrenheit") -> str:
    """Конвертировать температуру между различными единицами измерения.
    
    Args:
        value: Значение температуры
        from_unit: Исходная единица измерения (celsius, fahrenheit, kelvin)
        to_unit: Целевая единица измерения (celsius, fahrenheit, kelvin)
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    # Конвертируем в Цельсий сначала
    if from_unit == "celsius":
        celsius = value
    elif from_unit == "fahrenheit":
        celsius = (value - 32) * 5/9
    elif from_unit == "kelvin":
        celsius = value - 273.15
    else:
        return f"Ошибка: неизвестная исходная единица '{from_unit}'. Используйте: celsius, fahrenheit, kelvin"
    
    # Конвертируем из Цельсия в целевую единицу
    if to_unit == "celsius":
        result = celsius
    elif to_unit == "fahrenheit":
        result = celsius * 9/5 + 32
    elif to_unit == "kelvin":
        result = celsius + 273.15
    else:
        return f"Ошибка: неизвестная целевая единица '{to_unit}'. Используйте: celsius, fahrenheit, kelvin"
    
    # Определяем символы единиц
    symbols = {
        "celsius": "°C",
        "fahrenheit": "°F", 
        "kelvin": "K"
    }
    
    return f"{value} {symbols[from_unit]} = {result:.2f} {symbols[to_unit]}"

@mcp.tool()
async def stub_tool(input: str) -> str:
    """Простая заглушка для тестов.
    
    Args:
        input: Любой текст для обработки
    """
    log.info(f"Вызвана заглушка с input: {input}")
    return f"stub-tool выполнен. Аргументы: {input}"

# --------------------  ЗАПУСК СЕРВЕРА  --------------------
def main():
    """Запуск MCP сервера."""
    log.info("Запуск Weather MCP сервера...")
    try:
        # FastMCP автоматически использует stdio транспорт
        log.info("Сервер запущен и готов к работе.")
        mcp.run()
    except KeyboardInterrupt:
        log.info("Сервер остановлен пользователем")
    except Exception as e:
        log.exception(f"Ошибка при запуске сервера: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()