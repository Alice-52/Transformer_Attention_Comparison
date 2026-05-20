# Конфигурация главных параметров для проекта
from dataclasses import dataclass

# Через dataclass - удобно хранить настройки как объект - можно к нему без проблем обращаться
@dataclass
class Config:
    # Параметры данных
    dataset_name: str = 'ag_news'
    text_column: str = 'text'
    label_column: str = 'label'
    # Максимальный размер словаря
    max_vocab_size: int = 30000
    # Минимальная частота, чтобы попасть в словарь
    min_freq: int = 2
    # Минимальная длина текста после токенизации (длинее - придётся обрезать, короче - придётся pad-ить)
    max_len: int = 128

    # Параметры модели
    # Главная скрытая размерность модели - все векторы такого размера
    d_model: int = 128
    # Количество head в multi во внимании
    num_heads: int = 4
    # Количество слоёв - encoder-блоков
    num_layers: int = 2
    # Размер скрытого слоя в FeedForward (обычно больше d_model)
    d_ff: int = 256
    # Наша регуляризация против переобучения
    dropout: float = 0.1
    # Выбираем тип механизма внимания!!
    # Single_head / multihead / local / additive
    attention_type: str = 'multihead'
    local_window_size: int = 8
    # В AG_News - World/Sports/Business/Sci-Tech
    num_classes: int = 4

    # Параметры обучения
    # Размер батча
    batch_size: int = 64
    # Сколько раз пройдём по train - 3 эпохи супер для быстрых тестов
    epochs: int = 3
    # Learning rate
    lr: float = 3e-4
    # Для оптимизации AdamW - L2 регуляризация
    weight_decay: float = 1e-2

    # Параметры воспроизводимости
    # Фиксируем случайность - перемешивание данных, инициализация весов
    seed: int = 42
    # Где обучаем
    device: str = 'cuda'
    # Папка с результатами
    output_dir: str = 'artifacts'
