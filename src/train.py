# Как раз берём данные - обучаем - validation - лучшая версия и test - записываем результаты и графики
from __future__ import annotations

# Для создания поля аргументов - запуска всего исследования из одной строки
import argparse
# Запись в таблички
import csv
import json
# Подсчёт времени обучения
import time
# Чуть удобнее менять поля аргументов в config на новые
from dataclasses import replace
# Для сохранения в папки всех результатов
from pathlib import Path

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import (
    TextClassificationDataset,
    build_vocab,
    extract_texts_labels,
    load_ag_news_splits,
    pad_collate_fn,
)
from .eval import evaluate
from .model import TransformerClassifier
from .utils import count_parameters, ensure_dir, set_seed

# Одна эпоха обучения и контроль качества
def train_one_epoch(model, dataloader, optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    # Стадарт функции потерь для классификации
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    # Идём циклом по батчам, а tqdm показываем нам прогресс обучения
    for batch in tqdm(dataloader, desc='Training', leave=False):
        # Переносим на гпюшку
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Один шаг обучения
        # Обнуляем градиенты - получаем предсказаение - считаем loss - делаем backprop - обновляем веса
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Считаем средний loss и accuracy
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {'loss': total_loss / total, 'accuracy': correct / total}

# Сохраняем в табличку результаты
# После каждой эпохи по всем метрикам будут результаты
def _save_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

# Сохраняем итоговые результаты в полноценную табличку
def _save_json(data: Dict[str, float], path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Наши графики
def _plot_history(history: List[Dict[str, float]], output_dir: Path) -> None:
    if not history:
        return

    epochs = [row['epoch'] for row in history]

    # ПЕРВЫЙ ГРАФИК - train loss и val loss - loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row['train_loss'] for row in history], label='train_loss')
    plt.plot(epochs, [row['val_loss'] for row in history], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss during training')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=150)
    plt.close()

    # ВТОРОЙ ГРАФИК - train accuracy, val accuracy и val f1 - как модель учится по эпохам
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row['train_accuracy'] for row in history], label='train_accuracy')
    plt.plot(epochs, [row['val_accuracy'] for row in history], label='val_accuracy')
    plt.plot(epochs, [row['val_f1'] for row in history], label='val_f1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Quality during training')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'quality_curve.png', dpi=150)
    plt.close()

# Аргументы подготавливаем и можно запускать без редактирования вручную
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train Transformer classifier for text attention coursework.')
    parser.add_argument('--attention_type', choices=['single_head', 'multihead', 'local', 'additive'], default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    return parser

# Подтачиваем конфиг - без полного переписывания - только обновлённые аккуратненько
def resolve_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    updates = {}
    for key in ['attention_type', 'seed', 'epochs', 'batch_size', 'lr', 'weight_decay', 'output_dir']:
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    if updates:
        cfg = replace(cfg, **updates)
    return cfg


# ГЛАВНАЯ ФУНКЦИЯ
# Запускаем обучение
# (теперь можно и отдельной функцией - для удобства)
def run_training(cfg: Optional[Config] = None, save_artifacts: bool = True) -> Dict[str, object]:
    # Берём параметры из обновлённой конфигурации
    cfg = cfg or Config()
    # Фиксируем seed для воспроизводимости
    set_seed(cfg.seed)

    # У нас это GPU
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    print(f'Using device: {device}')

    # Загрузка данных и деление
    train_split, val_split, test_split = load_ag_news_splits(seed=cfg.seed)

    # Извлекаем тексты и метки
    train_texts, train_labels = extract_texts_labels(train_split)
    val_texts, val_labels = extract_texts_labels(val_split)
    test_texts, test_labels = extract_texts_labels(test_split)

    # Создаём словарь по train - говорим нет утечке данных
    vocab = build_vocab(train_texts, max_vocab_size=cfg.max_vocab_size, min_freq=cfg.min_freq)

    # Dataset
    train_ds = TextClassificationDataset(train_texts, train_labels, vocab=vocab, max_len=cfg.max_len)
    val_ds = TextClassificationDataset(val_texts, val_labels, vocab=vocab, max_len=cfg.max_len)
    test_ds = TextClassificationDataset(test_texts, test_labels, vocab=vocab, max_len=cfg.max_len)

    # padding
    pad_idx = vocab.stoi['<pad>']
    collate = lambda batch: pad_collate_fn(batch, pad_idx=pad_idx, max_len=cfg.max_len)

    # DataLoader - создаём готовые батчи для модели
    # Перемешиваем лишь train
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    # Создаём модель базово
    model = TransformerClassifier(
        vocab_size=len(vocab),
        num_classes=cfg.num_classes,
        max_len=cfg.max_len,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        attention_type=cfg.attention_type,
        local_window_size=cfg.local_window_size,
        pad_idx=pad_idx,
    ).to(device)

    # Выводим сколько параметров будет обучаться
    # Теперь с помощью доп функции
    trainable_parameters = count_parameters(model)
    print(f'Trainable parameters: {trainable_parameters:,}')

    # Оптимизатор - стандартный для трансфомеров - AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Выбираем лучшую модель по F1
    best_val_f1 = -1.0
    best_state = None
    history: List[Dict[str, float]] = []

    # Начинаем подсчёт времени обучения
    total_start = time.perf_counter()
    # Цикл по эпохам
    for epoch in range(1, cfg.epochs + 1):
        # Подсчёт времени по эпохам
        epoch_start = time.perf_counter()
        # Обучаем
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        # Оцениваем по validation
        val_metrics = evaluate(model, val_loader, device)
        # Таймер эпохи стоп
        epoch_time = time.perf_counter() - epoch_start

        # Записываем всё, чтобы следить и исследовать
        record = {
            'epoch': float(epoch),
            'train_loss': float(train_metrics['loss']),
            'train_accuracy': float(train_metrics['accuracy']),
            'val_loss': float(val_metrics['loss']),
            'val_accuracy': float(val_metrics['accuracy']),
            'val_f1': float(val_metrics['f1_macro']),
            'epoch_time_sec': float(epoch_time),
        }
        history.append(record)

        # Выводим всё, чтобы следить -_-
        print(
            f'Epoch {epoch:02d} | '
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # Сохраняем лучшее по F1
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Таймер обучения стоп
    total_time = time.perf_counter() - total_start

    # Считаем тест один раз по лучшей версии (тест не участвует никак в подборе модели)
    if best_state is not None:
        model.load_state_dict(best_state)

    # Теперь подсчёт метрик
    test_metrics = evaluate(model, test_loader, device)
    print('\nTest metrics:')
    for key, value in test_metrics.items():
        print(f'{key}: {value:.4f}')

    # Записываем всё в результаты - своя папка с history, summary, loss_curve, quality_curve, best_model
    output_dir = ensure_dir(Path(cfg.output_dir) / cfg.attention_type / f'seed_{cfg.seed}')
    if save_artifacts:
        # История обучения
        _save_history_csv(history, output_dir / 'history.csv')
        _save_json(
            {
                'attention_type': cfg.attention_type,
                'seed': cfg.seed,
                'best_val_f1': best_val_f1,
                'test_loss': float(test_metrics['loss']),
                'test_accuracy': float(test_metrics['accuracy']),
                'test_f1_macro': float(test_metrics['f1_macro']),
                'trainable_parameters': trainable_parameters,
                'wall_time_sec': total_time,
            },
            output_dir / 'summary.json',
        )
        _plot_history(history, output_dir)
        torch.save(model.state_dict(), output_dir / 'best_model.pt')

    # Возвращаем результат в словаре
    return {
        'attention_type': cfg.attention_type,
        'seed': cfg.seed,
        'best_val_f1': best_val_f1,
        'test_loss': float(test_metrics['loss']),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_f1_macro': float(test_metrics['f1_macro']),
        'trainable_parameters': trainable_parameters,
        'wall_time_sec': total_time,
        'output_dir': str(output_dir),
        'history': history,
    }

# Запуск обновления параметров и соответственно обучения
def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = resolve_config(args)
    run_training(cfg=cfg, save_artifacts=True)

# Для запуска как отдельного скрипта в Collab-е
if __name__ == '__main__':
    main()
