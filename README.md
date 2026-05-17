# Transformer Attention Comparison

Implementing and comparing attention mechanisms within a Transformer encoder-only architecture for text classification.

Полная реализация транфомера - в модульном варианте для удобства и в целом корректности сравнения механизмов внимания - меняем только их, а всё остальное остаётся таким же!

## Repository Structure

- `src/attention.py` — реализация механизмов внимания
- `src/model.py` — архитектура encoder-only Transformer для классификации текста
- `src/data.py` — подготавливаем данные - загрузка данных, словарь, датасет
- `src/train.py` — цикл обучения
- `src/eval.py` — оценка модели - accuracy и F1
- `src/config.py` — параметры эксперимента - централизованно всем управляем
- `src/utils.py` — маленькие вспомогательные функции

## How to start

`!pip install -r requirements.txt` — скачиваем необходимые библиотеки
`src/config.py` — меняем на нужный attention
`!python -m src.train` — обучаемся и смотрим на результаты!