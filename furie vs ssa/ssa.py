"""SSA: частота главной гармоники (шаг 5 — подгонка синусоиды)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.utils.extmath import randomized_svd

from lib.io import HR_MAX_BPM, HR_MIN_BPM, SAMPLE_RATE_HZ
from lib.signal import center, fit_sinusoid_freq

SSA_WINDOW = 300       # длина окна вложения
SSA_COMPONENTS = 10    # число компонент SVD (≥2 для пропуска тренда)
SSA_WORKERS = 16       # потоков в batch(); 0 = авто (os.cpu_count)


def embed_trajectory(x: np.ndarray, L: int) -> np.ndarray:
    """Шаг 1 SSA — матрица вложения (Hankel) размером L×K."""
    return sliding_window_view(x, L).T


def diagonal_average(M: np.ndarray, n: int) -> np.ndarray:
    """Шаг 4 SSA — восстановление ряда диагональным усреднением."""
    L, k = M.shape
    out = np.zeros(n)
    cnt = np.zeros(n)
    r, c = np.indices((L, k))
    t = (c + r).ravel()
    np.add.at(out, t, M.ravel())
    np.add.at(cnt, t, 1.0)
    return out / cnt


def reconstruct_component(u, s, v, n) -> np.ndarray:
    """Шаги 2-4 SSA — элементарная матрица + диагональное усреднение."""
    return diagonal_average(s * np.outer(u, v), n)


def dominant_freq(
    signal: np.ndarray,
    fs: float = SAMPLE_RATE_HZ,
    window: int = SSA_WINDOW,
) -> float:
    x = center(signal)
    n = x.size
    if n < 4:
        return 0.0

    # Шаг 1: вложение
    L = min(max(window, 2), n // 2)
    traj = embed_trajectory(x, L)

    # Шаг 2: SVD (усечённый для скорости)
    k = max(2, min(SSA_COMPONENTS, min(traj.shape)))
    u, s, vt = randomized_svd(traj, n_components=k, random_state=0)

    # Шаг 3: выбор компоненты — пропускаем i=0 (тренд), берём argmax(σᵢ²)
    start = 1 if k > 1 else 0
    best_i = int(np.argmax(s[start:] ** 2)) + start

    # Шаг 4: восстановление компоненты
    comp = center(reconstruct_component(u[:, best_i], s[best_i], vt[best_i], n))

    # Шаг 5: оценка частоты через least_squares
    return fit_sinusoid_freq(comp, fs, HR_MIN_BPM, HR_MAX_BPM)


def batch(
    signals: list[np.ndarray],
    fs: float = SAMPLE_RATE_HZ,
    workers: int = SSA_WORKERS,
) -> np.ndarray:
    n = len(signals)
    if n == 0:
        return np.array([], dtype=np.float64)

    if workers <= 0:
        workers = min(n, os.cpu_count() or 1)
    workers = max(1, workers)

    if workers == 1:
        try:
            from tqdm import tqdm
            it = tqdm(signals, desc="SSA", unit="sig")
        except ImportError:
            it = signals
        return np.array([dominant_freq(s, fs) for s in it], dtype=np.float64)

    out = np.empty(n, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(dominant_freq, sig, fs): i for i, sig in enumerate(signals)
        }
        try:
            from tqdm import tqdm
            it = tqdm(as_completed(futures), total=n, desc="SSA", unit="sig")
        except ImportError:
            it = as_completed(futures)
        for fut in it:
            out[futures[fut]] = fut.result()
    return out


def components(signal: np.ndarray, window: int = SSA_WINDOW, k: int = SSA_COMPONENTS):
    """Все k восстановленных компонент SSA — для графиков и анализа."""
    x = center(signal)
    n = x.size
    L = min(max(window, 2), n // 2)
    traj = embed_trajectory(x, L)
    k = max(2, min(k, min(traj.shape)))
    u, s, vt = randomized_svd(traj, n_components=k, random_state=0)
    return [reconstruct_component(u[:, i], s[i], vt[i], n) for i in range(s.size)], s, L
