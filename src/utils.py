# Доп функции
import random
from typing import Iterable, List

import numpy as np
import torch

# Определяем токены
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# Фиксируем нашу случайность для каждой библиотеки
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# Для токенизации
def tokenize(text: str) -> List[str]:
    return text.lower().replace('\n', ' ').split()

# Считаем количество параметров
# numel - число элементов в тензоре
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def chunks(iterable: Iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
