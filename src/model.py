# Собираем всё в единую модель

# Необходимые импорты
from typing import Optional

import torch
import torch.nn as nn

from .attention import AdditiveAttention, LocalMultiHeadAttention, MultiHeadAttention, SingleHeadAttention

# Позиционное кодирование - где находится токен
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        # Таблица позиционных векторов
        pe = torch.zeros(max_len, d_model)
        # Номера позиций
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Коэффициенты для разных частот
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # В зависимости от чётности - sin/cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        # Так как pe не должен обучаться, но должен быть вместе с моделью - в буффер его
        self.register_buffer('pe', pe.unsqueeze(0))

    # Просто добавляем информацию о позиции
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# Просто обычный двухслойный MLP - преобразуем токены
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Encoder - подключаем attention, который нам нужен
class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, attention_type: str, local_window_size: int):
        super().__init__()
        if attention_type == 'single_head':
            self.attn = SingleHeadAttention(d_model=d_model, dropout=dropout)
        elif attention_type == 'multihead':
            self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        elif attention_type == 'local':
            self.attn = LocalMultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                window_size=local_window_size,
                dropout=dropout,
            )
        elif attention_type == 'additive':
            self.attn = AdditiveAttention(d_model=d_model, dropout=dropout)
        else:
            raise ValueError(f'Unknown attention_type: {attention_type}')

        # Нормализуем
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # ff
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        # Регулизируем
        self.dropout = nn.Dropout(dropout)

    # По классической схеме трансформера - attention -> residual connection (+dropout) -> norm -> ff -> res. conn -> norm
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attn(x, mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# Вот и вся модель целиком
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_len: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        attention_type: str = 'multihead',
        local_window_size: int = 8,
        pad_idx: int = 0,
    ):
        super().__init__()
        # Padding
        self.pad_idx = pad_idx
        # Embedding - токены в вектор
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # О позициях
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        # Как зададим наши блоки
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attention_type=attention_type,
                    local_window_size=local_window_size,
                )
                for _ in range(num_layers)
            ]
        )
        # Классификатор - выдаём классы
        self.classifier = nn.Linear(d_model, num_classes)

    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input -> embeddings -> +position -> +dropout
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Проходим через encoder layers - как задали
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        # Классификация - нужно получить один вектор на всю последовательность
        # Если ничего не дали - берём среднее по всем токенам
        if attention_mask is None:
            pooled = x.mean(dim=1)
        else:
            # С маской берём среднее по настоящим токенам
            mask = attention_mask.unsqueeze(-1).float()
            summed = (x * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / denom

        # Наконец получаем logits по классам
        return self.classifier(pooled)
