"""Сингулярно-спектральный анализ (SSA): частота главной гармоники.

Метод SSA пошагово (https://ru.wikipedia.org/wiki/SSA_(метод)):
  1. Вложение (Embedding): строим матрицу траекторий X размером L x K,
     где L — длина окна, K = N - L + 1 — число столбцов.
  2. SVD: разлагаем X = U * S * V^T, получаем сингулярные значения и векторы.
  3. Группировка: отбираем компоненту с наибольшей энергией (σ_i^2),
     пропуская компоненту i=0 (она обычно захватывает тренд).
  4. Восстановление (Diagonal Averaging): из элементарной матрицы σ_i * u_i * v_i^T
     восстанавливаем временной ряд диагональным усреднением.
  5. БПФ восстановленной компоненты → доминирующая частота.
"""

from __future__ import annotations

import numpy as np

from config import HR_MAX_BPM, HR_MIN_BPM, SAMPLE_RATE_HZ, SSA_MAX_COMPONENTS, SSA_WINDOW_LENGTH


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _trajectory_matrix(x: np.ndarray, window_length: int) -> np.ndarray:
    """Шаг 1 SSA — матрица вложения (Hankel-подобная) размером L x K.

    Каждый столбец — подпоследовательность длины L, сдвинутая на 1 отсчёт.
    """
    k_cols = x.size - window_length + 1
    return np.column_stack([x[i : i + window_length] for i in range(k_cols)])


def _diagonal_averaging(elementary_matrix: np.ndarray, n: int) -> np.ndarray:
    """Шаг 4 SSA — восстановление ряда диагональным усреднением.

    Для каждого индекса t = col + row складываем все элементы по диагонали
    и делим на число слагаемых. Векторизованная реализация через np.add.at.
    """
    window_length, k_cols = elementary_matrix.shape
    series = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)

    # Строим индексный массив t = col + row для всех элементов матрицы
    rows_idx, cols_idx = np.indices((window_length, k_cols))
    t_idx = (cols_idx + rows_idx).ravel()

    np.add.at(series, t_idx, elementary_matrix.ravel())
    np.add.at(counts, t_idx, 1.0)
    return series / counts


def _reconstruct_component(
    u_col: np.ndarray,
    singular_value: float,
    vt_row: np.ndarray,
    window_length: int,
    n: int,
) -> np.ndarray:
    """Шаги 2-4 SSA — элементарная матрица и её диагональное усреднение."""
    elementary = singular_value * np.outer(u_col, vt_row)
    return _diagonal_averaging(elementary, n)


# ---------------------------------------------------------------------------
# Публичные функции
# ---------------------------------------------------------------------------

def dominant_frequency_ssa(
    signal,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    window_length: int = SSA_WINDOW_LENGTH,
    max_components: int = SSA_MAX_COMPONENTS,
    decimate: int = 1,
    max_samples: int | None = None,
    min_bpm: float = HR_MIN_BPM,
    max_bpm: float = HR_MAX_BPM,
) -> float:
    """Доминирующая частота сигнала методом SSA.

    Параметры ускорения (не влияют на алгоритм, только на точность):
      decimate    — прореживание: взять каждый N-й отсчёт.
      max_samples — обрезать сигнал до первых max_samples отсчётов.
    """
    x = np.asarray(signal, dtype=np.float64)

    # --- опциональное ускорение ---
    if decimate > 1:
        x = x[::decimate]
        sample_rate_hz = sample_rate_hz / decimate
    if max_samples and x.size > max_samples:
        x = x[:max_samples]

    x = x - x.mean()         # убираем постоянную составляющую
    n = x.size
    if n < 4:
        return 0.0

    # Длина окна не может превышать N/2
    window_length = min(max(window_length, 2), n // 2)

    # Шаг 1: матрица вложения
    trajectory = _trajectory_matrix(x, window_length)

    # Шаг 2: SVD
    u, singular_values, vt = np.linalg.svd(trajectory, full_matrices=False)

    # Шаг 3: выбираем компоненту с максимальной энергией σ_i^2
    # Пропускаем i=0 (медленный тренд) — берём начиная с i=1
    rank = min(max_components, singular_values.size)
    start = 1 if rank > 1 else 0
    best_i = int(np.argmax(singular_values[start:rank] ** 2)) + start

    # Шаг 4: восстанавливаем компоненту
    component = _reconstruct_component(
        u[:, best_i], singular_values[best_i], vt[best_i], window_length, n
    )
    component -= component.mean()

    # Шаг 5: БПФ восстановленной компоненты → доминирующая частота
    spectrum = np.abs(np.fft.rfft(component))[1:]
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)[1:]
    if spectrum.size == 0:
        return 0.0

    # Ищем пик в физиологическом диапазоне
    mask = (freqs >= min_bpm / 60.0) & (freqs <= max_bpm / 60.0)
    target_freqs = freqs[mask] if mask.any() else freqs
    target_spec = spectrum[mask] if mask.any() else spectrum
    return float(target_freqs[int(np.argmax(target_spec))])


def dominant_frequencies_ssa_batch(
    signals,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    window_length: int = SSA_WINDOW_LENGTH,
    max_components: int = SSA_MAX_COMPONENTS,
    decimate: int = 1,
    max_samples: int | None = None,
    min_bpm: float = HR_MIN_BPM,
    max_bpm: float = HR_MAX_BPM,
) -> np.ndarray:
    """Применяет dominant_frequency_ssa ко всей выборке сигналов."""
    return np.array(
        [
            dominant_frequency_ssa(
                s,
                sample_rate_hz=sample_rate_hz,
                window_length=window_length,
                max_components=max_components,
                decimate=decimate,
                max_samples=max_samples,
                min_bpm=min_bpm,
                max_bpm=max_bpm,
            )
            for s in signals
        ],
        dtype=np.float64,
    )


def compute_ssa_components(
    signal,
    window_length: int = SSA_WINDOW_LENGTH,
    max_components: int = SSA_MAX_COMPONENTS,
) -> dict:
    """Все восстановленные компоненты SSA — для построения графиков."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    window_length = min(max(window_length, 2), n // 2)

    trajectory = _trajectory_matrix(x, window_length)
    u, singular_values, vt = np.linalg.svd(trajectory, full_matrices=False)

    rank = min(max_components, singular_values.size)
    components = [
        _reconstruct_component(u[:, i], singular_values[i], vt[i], window_length, n)
        for i in range(rank)
    ]
    return {
        "components": components,
        "singular_values": singular_values[:rank],
        "window_length": window_length,
    }


def pick_dominant_component_index(
    singular_values: np.ndarray,
    skip_trend: bool = True,
) -> int:
    """Индекс компоненты с максимальной энергией σ_i^2 (для графиков)."""
    rank = singular_values.size
    start = 1 if skip_trend and rank > 1 else 0
    return int(np.argmax(singular_values[start:rank] ** 2)) + start
