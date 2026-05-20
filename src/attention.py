# Нужно для определения маски (может быть, а может и поленимся и не добавим)
from typing import Optional

# Основные библиотеки
import torch
import torch.nn as nn

# Делаем каркас для работы и добавляем модульность для простого сравнения механизмов внимания

# САМАЯ ВАЖНАЯ ФУНКЦИЯ ДЛЯ РАБОТЫ
# Тут считается Attention для трансформера
# Матрицы
# Q - query - что обрабатываем (какое слово)
# K - key - с чем сравниваем (с каким словом)
# V - value - как сравниваем/какая информация уже есть (насколько они связаны)
# Маска может понадобится для игнорирования padding или скрытия токенов ненужных
def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Нужно для формулы рассчёта внимания - стабилизирует значения
    d_k = q.size(-1)

    # Подсчитываем матрицу схожести/сопряжённости
    # (transpose - меняем две размерности местами - для умножения Q*K^T)
    # ("нормируем", поделив на d_k - стабилизируем, без него значения слишком большие - градиенты ломаются)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

    # Маску можно добавить для того, чтобы убрать padding - зачем нам пустые значения
    if mask is not None:
        # Если маска [B,L] -> [B, 1, 1, L] - ожидаемый ввод
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)
        # Если маска [B,L,L] -> [B,1,L,L]
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
        # Все ненужные значения меняем на супер маленькие числа - игнорируем во внимании
        scores = scores.masked_fill(~mask, float('-inf'))

    # Тот самый softmax - превращаем scores в вероятности - важность других слов для нашего
    attn = torch.softmax(scores, dim=-1)
    # Если вдруг появились nan - меняем на нули, чтобы без ошибок всё работало
    attn = torch.nan_to_num(attn, nan=0.0)

    # Умножаем веса на Value - новое представление токена, которое учитывает контекст
    # То есть смешали прошлую информацию с новой
    context = torch.matmul(attn, v)
    return context


# РАЗНЫЕ МЕХАНИЗМЫ

# SINGLE HEAD - один раз смотрим на контекс - одним глазком
# Модульность для удобства работы - меняем/вытаскиваем без проблем
class SingleHeadAttention(nn.Module):

    # Создаём слои
    # d_model - размер скрытого представления токена (128/256/...)
    # По классике Q, K, V -> в out в размерность d_model -> для последующей работы с ним в трансформере
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        # Говорим нет переобучению - зануляем рандомные признаки
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Добавляем в форму размерность головы - 1
        q = self.w_q(x).unsqueeze(1)
        k = self.w_k(x).unsqueeze(1)
        v = self.w_v(x).unsqueeze(1)

        # Подготавливаем необходимую маску с размерность 4
        attn_mask = mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        
        # Основные рассчёты
        # (для маски ещё добавляем размерность головы 1 для соответствия для работы)
        context = _scaled_dot_product_attention(q, k, v, mask=attn_mask)
        # Ну и убираем эту единичку в соответствие с размерностью
        if context.dim() == 4 and context.size(1) == 1:
            context = context.squeeze(1)
        elif context.dim() == 3:
            pass
        else:
            # Если размерность всё ещё 4 (всё очень плохо) - берём среднее по измерению
            if context.dim() == 4:
                context = context.mean(dim=1)

        # Итоговый выход для последубщей работы трансформера
        return self.out(self.dropout(context))



# MULTI HEAD - рассматриваем с разных сторон - более гибкий подход
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # Проверка - размер скрытого слоя d_model должен ровно делиться на количество голов
        # После разбиения каждая голова должна получить одинаковую разммерность head_dim
        # Потому что потом не собрать обратно в один тензор правильно без этого
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    # Разбиваем головы - превращаем [B,L,d_model] -> [B,L,num_heads, num_dim] -> [B,num_heads,L,head_dim]
    # (B-размер батча, L-длина последовательности)
    # Считаем параллельно внимание для всех голов
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        x = x.view(b, l, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    # Обратно пересчитываем после пересчёта с контекстом
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, l, hd = x.shape
        # contiguous - после permute нам нужна непрерывность, чтобы view правильно сработал
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(b, l, h * hd)


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Разбиваем на Q,K,V и разбиваем на головы
        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        # Подсчёты и рассчёты
        context = _scaled_dot_product_attention(q, k, v, mask=mask)
        # Сборка обратно
        context = self._combine_heads(context)
        # Выход и всё готово для последующей работы трансформера
        return self.out(self.dropout(context))


# ADDITIVE - Bahdnau
class AdditiveAttention(nn.Module):
    # Все параметры как обычно
    # hidden_sim - размерность внутренней сети оценки (если не передали, то равен d_model)
    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or d_model
        # Те же Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # САМОЕ ГЛАВНОЕ В ADDITIVE
        # Считается связанность/совместимость между токенами
        # Вектор признаков проходит через маленькую нейросеть -> score
        # (в других - dot-product - просто скалярное произведение)
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

        # Возвращаем обратно к размерности d_model
        self.out = nn.Linear(d_model, d_model)
        # Снова говорим НЕТ переобучению
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Превращаем в векторы
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Размерности подготавливаем для этой пары для обучения
        q_exp = q.unsqueeze(2)
        k_exp = k.unsqueeze(1)

        # Энергия - совместимость  - query и key складываются, проходимся tanh - нелинейное представление пары токенов
        # Вместо произведения получаем обучаемую совместимость
        energy = torch.tanh(q_exp + k_exp)
        # Превращаем пару токенос в число
        scores = self.score(energy).squeeze(-1)

        # Учитываем пустые токены - padding
        query_mask = None
        if mask is not None:
            mask = mask.to(dtype=torch.bool)
            if mask.dim() == 2:
                key_mask = mask.unsqueeze(1)
                query_mask = mask.unsqueeze(-1)
                scores = scores.masked_fill(~key_mask, -1e9)
            elif mask.dim() == 3:
                key_mask = mask.squeeze(1) if mask.size(1) == 1 else mask
                scores = scores.masked_fill(~key_mask, -1e9)
            else:
                # Уже натыкалась на ошибки с размерностями - нам это больше не надо
                raise ValueError('Wrong mask shape!! (add. att.)')

        # Классически подсчитываем финальное внимание
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Умножаем веса на value - новое представления токенов
        context = torch.matmul(attn, v)

        # Обнуляем pad-токены, чтобы не мешались
        if query_mask is not None:
            context = context * query_mask.float()

        # Регуляризация и вывод
        return self.out(self.dropout(context))




# LOCAL - локально вокруг слова - не вся последовательность
# Наследуемся от Multi
class LocalMultiHeadAttention(MultiHeadAttention):

    # window_size - размер окошка, где рассматриваем контекст
    # (super - вызываем инициализацию Multi - расширяем размром окошка)
    def __init__(self, d_model: int, num_heads: int, window_size: int, dropout: float = 0.1):
        super().__init__(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.window_size = window_size

    # Ограничивание
    def _local_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # Типичная маска с padding
        b, l = attention_mask.shape
        device = attention_mask.device

        # Создаём список позиций
        positions = torch.arange(l, device=device)
        # Матрица расстояний между всеми позициями
        distance = (positions[None, :] - positions[:, None]).abs()
        # И ограничиваем окошком
        local = distance <= self.window_size
        # Теперь эта матрица растягиваем до размера [B, 1, L, L]
        local = local.unsqueeze(0).unsqueeze(1).expand(b, 1, l, l)

        # Учитываем padding, чтобы пустые слова не могли ни давать информацию, ни спрашивать контекст
        key_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(b, 1, l, l)
        query_mask = attention_mask.unsqueeze(1).unsqueeze(3).expand(b, 1, l, l)

        # Итоговая маска: находимся в окошке, key не pad, query не pad
        return local & key_mask & query_mask

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Без маски нельзя построить правильное локальное окно
        if mask is None:
            raise ValueError('LocalMultiHeadAttention requires attention mask')
        # Строим локальную маску и идём
        local_mask = self._local_mask(mask)
        return super().forward(x, mask=local_mask)
