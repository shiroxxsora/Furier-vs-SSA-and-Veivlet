"""Сингулярно-спектральный анализ (SSA): частота главной гармоники."""

from __future__ import annotations

import numpy as np

from config import SAMPLE_RATE_HZ, SSA_MAX_COMPONENTS, SSA_WINDOW_LENGTH


def _trajectory_matrix(x, window_length):
    """Шаг 1 SSA: матрица вложения L x K."""
    n = x.size
    k_cols = n - window_length + 1
    return np.column_stack([x[i : i + window_length] for i in range(k_cols)])


def _diagonal_averaging(elementary_matrix, n):
    """Шаг 4 SSA: диагональное усреднение."""
    window_length, k_cols = elementary_matrix.shape
    series = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    for col in range(k_cols):
        for row in range(window_length):
            idx = col + row
            series[idx] += elementary_matrix[row, col]
            counts[idx] += 1.0
    return series / counts


def _reconstruct_component(u, singular_value, vt_row, window_length, n):
    elementary = singular_value * np.outer(u, vt_row)
    return _diagonal_averaging(elementary, n)


def dominant_frequency_ssa(
    signal,
    sample_rate_hz=SAMPLE_RATE_HZ,
    window_length=SSA_WINDOW_LENGTH,
    max_components=SSA_MAX_COMPONENTS,
):
    """
    SSA: вложение -> SVD -> восстановление главной осцилляторной компоненты -> ДПФ.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size
    window_length = min(max(window_length, 2), n // 2)

    trajectory = _trajectory_matrix(x, window_length)
    u, singular_values, vt = np.linalg.svd(trajectory, full_matrices=False)

    rank = min(max_components, singular_values.size)
    start_index = 1 if rank > 1 else 0
    best_index = start_index
    best_energy = -1.0
    for i in range(start_index, rank):
        energy = singular_values[i] ** 2
        if energy > best_energy:
            best_energy = energy
            best_index = i

    component = _reconstruct_component(
        u[:, best_index],
        singular_values[best_index],
        vt[best_index, :],
        window_length,
        n,
    )
    component = component - np.mean(component)
    spectrum = np.abs(np.fft.rfft(component))
    if spectrum.size <= 1:
        return 0.0
    k_peak = int(np.argmax(spectrum[1:]) + 1)
    return k_peak * sample_rate_hz / n


def dominant_frequencies_ssa_batch(signals, sample_rate_hz=SAMPLE_RATE_HZ, window_length=SSA_WINDOW_LENGTH):
    return np.array(
        [dominant_frequency_ssa(s, sample_rate_hz, window_length) for s in signals],
        dtype=np.float64,
    )

def compute_ssa_components(
    signal,
    window_length=SSA_WINDOW_LENGTH,
    max_components=SSA_MAX_COMPONENTS,
):
    """Возвращает восстановленные элементарные компоненты SSA для графиков."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size
    window_length = min(max(window_length, 2), n // 2)
    trajectory = _trajectory_matrix(x, window_length)
    u, singular_values, vt = np.linalg.svd(trajectory, full_matrices=False)
    rank = min(max_components, singular_values.size)
    components = []
    for i in range(rank):
        comp = _reconstruct_component(u[:, i], singular_values[i], vt[i, :], window_length, n)
        components.append(comp)
    return {
        "components": components,
        "singular_values": singular_values[:rank],
        "window_length": window_length,
    }


def pick_dominant_component_index(singular_values, skip_trend=True):
    rank = singular_values.size
    start = 1 if skip_trend and rank > 1 else 0
    best_index = start
    best_energy = -1.0
    for i in range(start, rank):
        energy = singular_values[i] ** 2
        if energy > best_energy:
            best_energy = energy
            best_index = i
    return best_index
