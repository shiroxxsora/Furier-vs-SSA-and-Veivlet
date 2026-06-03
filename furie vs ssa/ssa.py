"""SSA: частота главной гармоники (шаг 5 — подгонка синусоиды)."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.utils.extmath import randomized_svd

from lib.io import HR_MAX_BPM, HR_MIN_BPM, SAMPLE_RATE_HZ
from lib.signal import center, fit_sinusoid_freq

SSA_WINDOW = 300
SSA_COMPONENTS = 5


def embed_trajectory(x: np.ndarray, L: int) -> np.ndarray:
    return sliding_window_view(x, L).T


def diagonal_average(M: np.ndarray, n: int) -> np.ndarray:
    L, k = M.shape
    out = np.zeros(n)
    cnt = np.zeros(n)
    r, c = np.indices((L, k))
    t = (c + r).ravel()
    np.add.at(out, t, M.ravel())
    np.add.at(cnt, t, 1.0)
    return out / cnt


def reconstruct_component(u, s, v, n) -> np.ndarray:
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

    L = min(max(window, 2), n // 2)
    traj = embed_trajectory(x, L)
    k = max(2, min(SSA_COMPONENTS, min(traj.shape)))
    u, s, vt = randomized_svd(traj, n_components=k, random_state=0)

    i = int(np.argmax(s ** 2))
    comp = center(reconstruct_component(u[:, i], s[i], vt[i], n))

    return fit_sinusoid_freq(comp, fs, HR_MIN_BPM, HR_MAX_BPM)


def batch(signals: list[np.ndarray], fs: float = SAMPLE_RATE_HZ) -> np.ndarray:
    try:
        from tqdm import tqdm
        it = tqdm(signals, desc="SSA", unit="sig")
    except ImportError:
        it = signals
    return np.array([dominant_freq(s, fs) for s in it], dtype=np.float64)


def components(signal: np.ndarray, window: int = SSA_WINDOW, k: int = SSA_COMPONENTS):
    x = center(signal)
    n, L = x.size, min(max(window, 2), x.size // 2)
    traj = embed_trajectory(x, L)
    k = max(2, min(k, min(traj.shape)))
    u, s, vt = randomized_svd(traj, n_components=k, random_state=0)
    return [reconstruct_component(u[:, i], s[i], vt[i], n) for i in range(s.size)], s, L
