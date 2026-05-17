# Как раз берём данные - обучаем - validation - лучшая версия и test
from typing import Dict

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
from .utils import count_parameters, set_seed

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
        # Обнуляем градиенты - получаем предсказаение - считаем loss - делаем backdrop - обновляем веса
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

# Запуск всего
def main(config=None):
    # Берём параметры из конфигурации, которую меняем в Collab
    if config is None:
        cfg = Config()
    else:
        cfg = config
    # Фиксируем seed для воспроизводимости
    set_seed(cfg.seed)
    print(f"Seed: {cfg.seed}")
    # Вид внимания
    print(f"Attention type: {cfg.attention_type}")

    # У нас это GPU
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu')
    print(f'Using device: {device}')

    # Загрузка данных
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

    # Создаём модель - выбираем механизм внимания из конфигурации
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
    print(f'Trainable parameters: {count_parameters(model):,}')

    # Оптимизатор - стандартный для трансфомеров - AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Выбираем лучшую модель по F1
    best_val_f1 = -1.0
    best_state = None

    # Цикл по эпохам
    for epoch in range(1, cfg.epochs + 1):
        # Обучаем
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        # Оцениваем по validation
        val_metrics = evaluate(model, val_loader, device)

        # Выводим всё, чтобы следить -_-
        print(
            f'Epoch {epoch:02d} | '
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
        )

        # Сохраняем лучшее по F1
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Считаем тест один раз по лучшей версии (тест не участвует никак в подборе модели)
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    print('\nTest metrics:')
    for key, value in test_metrics.items():
        print(f'{key}: {value:.4f}')


# Для запуска как отдельного скрипта в Collab-е
if __name__ == '__main__':
    main()
