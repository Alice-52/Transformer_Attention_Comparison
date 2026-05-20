from typing import Dict

import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Не строим граф вычилений, не сохраняем градиенты во время оценки- не сохраняем для backprop
# Быстрота и эконмность!
@torch.no_grad()
def evaluate(model, dataloader, device: torch.device) -> Dict[str, float]:
    # Вырбуем всё и переводим модель в режим оценки
    model.eval()
    # Предсказания
    all_preds = []
    # Истинные метки
    all_labels = []

    total_loss = 0.0
    # Та же функция потерь, что и при обучении - стабильность
    criterion = torch.nn.CrossEntropyLoss()

    # Цикл по батчам
    for batch in tqdm(dataloader, desc='Evaluating', leave=False):
        # Переносим на gpu-шку
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Проходим через модель
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        # Считаем loss
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        # Сохраняем предсказания и метки - максимальный логит
        preds = logits.argmax(dim=-1)
        # Списки собираем на cpu, чтобы можно было посчитать метрики sklearn
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Средний loss по всему
    avg_loss = total_loss / len(dataloader.dataset)
    # Accuracy - доля правильных ответов
    acc = accuracy_score(all_labels, all_preds)
    # F1 - macro - считаем сначала отдельно по каждому классу, а потом усредняем 
    # нужно качество не только по самому частому классу
    f1 = f1_score(all_labels, all_preds, average='macro',zero_division=0)

    # Выводим ключевые метрики
    return {'loss': avg_loss, 'accuracy': acc, 'f1_macro': f1}
