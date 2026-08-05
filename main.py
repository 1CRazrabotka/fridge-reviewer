"""
Ревизор холодильника - Local-First AI приложение
Использует локальную модель Qwen2.5-VL через Ollama (без API-ключей!)
"""

import streamlit as st
import ollama
import json
import base64
from typing import Dict, Any
from PIL import Image

# === Константы ===
MODEL_NAME = "minicpm-v" # Локальная мультимодальная модель

# === Функции для работы с AI ===

def analyze_fridge_image(image_bytes: bytes) -> str:
    """
    Отправляет изображение в локальную Ollama для анализа.
    Использует умный промпт, который не даёт модели схалтурить.
    """
    # ВАЖНО: не даём модели готовый пример JSON, чтобы она не копировала его
    prompt = """Посмотри на это фото холодильника очень внимательно.

Твоя задача:
1. Найди ВСЕ продукты, которые реально видны на фото. Называй их конкретно (например, "молоко Простоквашино", "яблоки", "колбаса"), а НЕ пиши "название продукта".
2. Для каждого продукта оцени свежесть по внешнему виду: "свежее", "среднее" или "испорчено".
3. Придумай 3 РЕАЛЬНЫХ рецепта, которые можно приготовить из этих конкретных продуктов.
4. Дай 2 практических совета по хранению.

Ответ верни СТРОГО в формате JSON. Никакого текста до или после JSON. Никаких markdown-обёрток типа ```json.

Структура JSON:
- products: массив объектов с полями name (строка) и freshness (строка)
- recipes: массив объектов с полями name, ingredients (массив строк), steps (массив строк)
- storage_tips: массив строк

НАЧНИ ответ сразу с открывающей фигурной скобки {"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_bytes]
                }
            ],
            options={
                "temperature": 0.2,  # Ещё строже, чтобы не фантазировала
                "num_predict": 1500
            }
        )

        response_text = response['message']['content']

        # Очистка от markdown-оберток, если модель всё-таки их добавила
        if "```" in response_text:
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
            if json_match:
                response_text = json_match.group(1).strip()

        # Если модель добавила текст ДО JSON, найдём первую { и последнюю }
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1:
            response_text = response_text[start:end + 1]

        return response_text

    except Exception as e:
        st.error(f"❌ Ошибка при вызове модели '{MODEL_NAME}'.")
        st.error(f"Детали: {e}")
        return ""

# === UI функции ===

def show_main_interface() -> None:
    """Основной интерфейс приложения"""

    # === Боковое меню ===
    with st.sidebar:
        st.title("⚙️ О проекте")
        st.markdown("""
        **Ревизор холодильника** 🧊
        
        AI-приложение для анализа продуктов.
        
        - 🤖 Модель: **minicpm-v"* (Local)
        - 🆓 **0 рублей**, без API-ключей
        - 🔒 **100% приватно**: данные не покидают твой компьютер
        - ⚡ Работает даже без интернета
        """)

        st.divider()
        st.markdown("### 📝 Как использовать")
        st.markdown("""
        1. Сфотографируй открытый холодильник
        2. Загрузи фото в приложение
        3. Нажми "Анализировать"
        4. Получи рекомендации
        """)

    # === Основной контент ===
    st.title("🧊 Ревизор холодильника")
    st.markdown("### Загрузи фото своего холодильника, и локальный AI проанализирует продукты")

    uploaded_file = st.file_uploader(
        "Выберите фото",
        type=["jpg", "jpeg", "png"],
        help="Поддерживаются форматы: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📸 Твое фото")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("#### 🤖 Результаты анализа")

            if st.button("🔍 Анализировать", use_container_width=True, type="primary"):
                with st.spinner("🤖 Анализирую ваш холодильник... (первый запуск может занять 10-20 сек)"):
                    # Читаем байты изображения
                    image_bytes = uploaded_file.getvalue()

                    # Вызываем локальный AI
                    raw_response = analyze_fridge_image(image_bytes)

                    if raw_response:
                        st.session_state.last_response = raw_response
                        st.rerun()

            # Отображаем результаты
            if "last_response" in st.session_state and st.session_state.last_response:
                display_results(st.session_state.last_response)

def display_results(raw_text: str) -> None:
    """Отображает результаты анализа, пытаясь распарсить JSON"""
    try:
        result = json.loads(raw_text)

        # === Распознанные продукты ===
        if "products" in result and result["products"]:
            st.markdown("#### 🥦 Распознанные продукты")
            for product in result["products"]:
                name = product.get("name", "Неизвестно")
                freshness = product.get("freshness", "неизвестно")

                if "свежее" in freshness.lower():
                    st.success(f"✅ **{name}** — свежее")
                elif "среднее" in freshness.lower() or "норм" in freshness.lower():
                    st.warning(f"⚠️ **{name}** — среднее")
                elif "испорчено" in freshness.lower() or "плох" in freshness.lower():
                    st.error(f"❌ **{name}** — испорчено")
                else:
                    st.info(f"ℹ️ **{name}** — {freshness}")

        # === Рецепты ===
        if "recipes" in result and result["recipes"]:
            st.markdown("#### 🍳 Рекомендуемые рецепты")
            for i, recipe in enumerate(result["recipes"], 1):
                with st.expander(f"**{i}. {recipe.get('name', 'Рецепт')}**", expanded=False):
                    if "ingredients" in recipe:
                        st.markdown("**Ингредиенты:**")
                        for ing in recipe["ingredients"]:
                            st.markdown(f"- {ing}")
                    if "steps" in recipe:
                        st.markdown("**Приготовление:**")
                        for j, step in enumerate(recipe["steps"], 1):
                            st.markdown(f"{j}. {step}")

        # === Советы по хранению ===
        if "storage_tips" in result and result["storage_tips"]:
            st.markdown("#### 💡 Советы по хранению")
            for tip in result["storage_tips"]:
                st.info(f"💡 {tip}")

    except json.JSONDecodeError:
        # Если модель вернула не строгий JSON, показываем текст как есть (fallback)
        st.markdown("#### 🤖 Ответ нейросети:")
        st.markdown(raw_text)
        st.info("💡 Подсказка: Нейросеть ответила свободным текстом, а не строгим JSON. Это нормальная ситуация для локальных моделей, приложение корректно это обработало!")

# === Главная функция ===

def main() -> None:
    st.set_page_config(
        page_title="🧊 Ревизор холодильника",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    show_main_interface()

if __name__ == "__main__":
    main()