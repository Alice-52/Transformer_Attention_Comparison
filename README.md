# Transformer Attention Comparison

Implementing and comparing attention mechanisms within a Transformer encoder-only architecture for text classification.

Полная реализация транфомера - в модульном варианте для удобства и в целом корректности сравнения механизмов внимания - меняем только их, а всё остальное остаётся таким же!

## Repository Structure

- `src/attention.py` — реализация механизмов внимания
- `src/model.py` — архитектура encoder-only Transformer для классификации текста
- `src/data.py` — подготавливаем данные - загрузка данных, словарь, датасет
- `src/train.py` — цикл обучения, промежуточные результаты и графики
- `src/eval.py` — оценка модели - accuracy и F1
- `src/config.py` — параметры эксперимента - централизованно всем управляем
- `src/utils.py` — маленькие вспомогательные функции
- `src/experiment.py` — запуск серии экспериментов по attention и seed

## How to start

Если работать в Google Collab, то сначала монтируем Drive, копируем папку в корневую /content (для правильной работы с Path) - и далее по инструкции.

`!pip install -r requirements.txt` — скачиваем необходимые библиотеки
`!python -m src.experiment --attention_types single_head multihead local additive --seeds 42 123 777` — эксперименты с выбранными механизмами и сидами

## Results

После запуска получаем результаты в папке `artifacts/`:
- `history.csv` — история обучения по эпохам - в метриках;
- `loss_curve.png` — график loss по эпохам - как исправляется по эпохам;
- `quality_curve.png` — график качества - как обучается по эпохам;
- `summary.json` — метрики одного запуска;
- `best_model.pt` — веса лучшей модели;
- `all_runs.csv` и `summary_by_attention.csv` — итоговые таблицы по всем экспериментам