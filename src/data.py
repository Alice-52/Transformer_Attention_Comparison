# Подготавливаем данные для последующей работы

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from .utils import PAD_TOKEN, UNK_TOKEN, tokenize

# Наш словарь: слово - число
@dataclass
class Vocab:
    # string to index
    stoi: Dict[str, int]

    # Обратный словарь - декодирование
    @property
    def itos(self) -> Dict[int, str]:
        return {idx: token for token, idx in self.stoi.items()}

    def __len__(self) -> int:
        return len(self.stoi)

    # Превращаем каждый токен в id и ненайденные в unknown
    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk_idx = self.stoi[UNK_TOKEN]
        return [self.stoi.get(tok, unk_idx) for tok in tokens]

# Создаём наш словарь по специальным обучающим текстам
# Проходимся по всем текстам, токенизируем, вычисляем частые (ограничение на словарь - зачем нам огромный словарь)
def build_vocab(texts: Sequence[str], max_vocab_size: int = 30000, min_freq: int = 2) -> Vocab:
    # Для частоты - добавляем только частые
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    # Те самые специальные токены - padding для заполнения, unknown для неизвестных
    specials = [PAD_TOKEN, UNK_TOKEN]
    stoi = {token: idx for idx, token in enumerate(specials)}

    # Как раз добавляем
    for token, freq in counter.most_common():
        # Если хотя бы два раза не встретили, то скипаем
        if freq < min_freq:
            continue
        # Если уже у нас есть, то зачем он нам
        if token in stoi:
            continue
        stoi[token] = len(stoi)
        # Добавили до максимума - оставливаемся вовремя
        if len(stoi) >= max_vocab_size:
            break

    return Vocab(stoi=stoi)

# Готовый PyTorch dataset
class TextClassificationDataset(Dataset):
    # Передаём текст, метки, словарь, макс. длину пос-ти
    def __init__(self, texts: Sequence[str], labels: Sequence[int], vocab: Vocab, max_len: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    # Токенизируем текст, переводим в id, обрезаем до макс. длины и возвращаем
    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = int(self.labels[idx])
        token_ids = self.vocab.encode(tokenize(text))[: self.max_len]
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Тексты имеют разную длины, а нам удобнее обучать на батчах одинаковой формы
def pad_collate_fn(batch, pad_idx: int, max_len: int):
    sequences, labels = zip(*batch)
    batch_size = len(sequences)

    # Создаём матрицу, заполняем pad
    input_ids = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    # Маска внимания - сначала нули, а потом где настоящий токен - true
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    # Цикл - последовательность обрезается, записывается в inout_ids, и помечаем настоящие для внимания
    for i, seq in enumerate(sequences):
        truncated = seq[:max_len]
        length = len(truncated)
        if length > 0:
            input_ids[i, :length] = truncated
            attention_mask[i, :length] = True

    labels = torch.stack(labels)
    # Возвращаем готовый формат для модели
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

# Загружаем датачет AG News
def load_ag_news_splits(seed: int = 42):
    # Загружаем из Hugging Face
    dataset = load_dataset('ag_news')
    # Перемешиваем train и test
    train_split = dataset['train'].shuffle(seed=seed)
    test_split = dataset['test'].shuffle(seed=seed)

    split = train_split.train_test_split(test_size=0.1, seed=seed)
    train_data = split['train']
    # Выделяем validation
    val_data = split['test']

    return train_data, val_data, test_split

# Чисто для помощи палочка-выручалочка - достаём тексты и метки
def extract_texts_labels(dataset_split, text_column: str = 'text', label_column: str = 'label') -> Tuple[List[str], List[int]]:
    texts = list(dataset_split[text_column])
    labels = list(dataset_split[label_column])
    return texts, labels
