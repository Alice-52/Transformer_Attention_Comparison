# Для удобства - запускаем каждый эксперимент не вручную
# Прогоняет все механизмы внимания по всем выбранным сидам
# Также собираем результаты и считает метрики
from __future__ import annotations

# Запускаем обучение из командной строки
import argparse
# Сохраняем результаты в удобные таблички
import csv
import json
import statistics
# Меняем поля в config - всё эксперименты здесь
from dataclasses import replace
# Для работы с путями
from pathlib import Path
# Для записи словарей и т.д.
from typing import Dict, List

from .config import Config
from .train import run_training
from .utils import ensure_dir


# Обрабатывает наш запрос на эксперименты обучения - задаём всё
# Сиды и виды внимания - по умолчанию какие выбрали
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run full attention experiments.')
    parser.add_argument(
        '--attention_types',
        nargs='+',
        default=['single_head', 'multihead', 'local', 'additive'],
        choices=['single_head', 'multihead', 'local', 'additive'],
    )
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 777])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser


# Записываем результаты в табличку
def _write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

# Берём базовый config И меняем по полученным новым параметрам
def main(argv: List[str] | None = None) -> None:
    # Полученные параметры
    parser = build_parser()
    args = parser.parse_args(argv)

    # Обновляем параметры, если нужно
    base_cfg = Config()
    if args.epochs is not None:
        base_cfg = replace(base_cfg, epochs=args.epochs)
    if args.batch_size is not None:
        base_cfg = replace(base_cfg, batch_size=args.batch_size)
    if args.output_dir is not None:
        base_cfg = replace(base_cfg, output_dir=args.output_dir)

    # Задаём корневую папку
    root = ensure_dir(Path(base_cfg.output_dir))
    all_rows: List[Dict[str, object]] = []

    # Двойной цикл по механизму и сиду
    for attention_type in args.attention_types:
        for seed in args.seeds:
            # Создаём cfg - запускаем обучение и результат добавляется в список all_rows
            cfg = replace(base_cfg, attention_type=attention_type, seed=seed)

            print('\n' + '=' * 80)
            print(f'Running attention_type={attention_type} seed={seed}')
            result = run_training(cfg=cfg, save_artifacts=True)
            all_rows.append(
                {
                    'attention_type': result['attention_type'],
                    'seed': result['seed'],
                    'test_loss': result['test_loss'],
                    'test_accuracy': result['test_accuracy'],
                    'test_f1_macro': result['test_f1_macro'],
                    'best_val_f1': result['best_val_f1'],
                    'trainable_parameters': result['trainable_parameters'],
                    'wall_time_sec': result['wall_time_sec'],
                }
            )

    # После всех запусков создаём общую папку с конечными результатами - средние и отклонения
    _write_csv(all_rows, root / 'all_runs.csv')

    summary_rows: List[Dict[str, object]] = []
    for attention_type in args.attention_types:
        subset = [row for row in all_rows if row['attention_type'] == attention_type]
        summary_rows.append(
            {
                'attention_type': attention_type,
                'test_accuracy_mean': statistics.mean(row['test_accuracy'] for row in subset),
                'test_accuracy_std': statistics.pstdev(row['test_accuracy'] for row in subset) if len(subset) > 1 else 0.0,
                'test_f1_mean': statistics.mean(row['test_f1_macro'] for row in subset),
                'test_f1_std': statistics.pstdev(row['test_f1_macro'] for row in subset) if len(subset) > 1 else 0.0,
                'wall_time_mean': statistics.mean(row['wall_time_sec'] for row in subset),
                'wall_time_std': statistics.pstdev(row['wall_time_sec'] for row in subset) if len(subset) > 1 else 0.0,
            }
        )

    _write_csv(summary_rows, root / 'summary_by_attention.csv')
    with (root / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump({'runs': all_rows, 'summary_by_attention': summary_rows}, f, ensure_ascii=False, indent=2)

    print('\nSaved:')
    print(root / 'all_runs.csv')
    print(root / 'summary_by_attention.csv')
    print(root / 'summary.json')


if __name__ == '__main__':
    main()
