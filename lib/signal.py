"""Обработка сигналов: центрирование, ЧСС, подгонка синусоиды, сохранение графиков."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


def center(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean()


def time_axis(n: int, fs: float) -> np.ndarray:
    return np.arange(n) / fs


def hz_to_bpm(hz: np.ndarray | float) -> np.ndarray | float:
    return hz * 60.0


def peak_in_band(amps: np.ndarray, freqs: np.ndarray, f_min: float, f_max: float) -> float:
    mask = (freqs >= f_min) & (freqs <= f_max)
    f, a = (freqs[mask], amps[mask]) if mask.any() else (freqs, amps)
    return float(f[int(np.argmax(a))])


def sinusoid(t, A, w, phi, A0):
    return A * np.sin(w * t + phi) + A0


def linear_fit_at_omega(t, y, w):
    S, C = np.sin(w * t), np.cos(w * t)
    X = np.column_stack([S, C, np.ones_like(t)])
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b, a0 = c
    err = y - X @ c
    return float(np.hypot(a, b)), w, float(np.arctan2(b, a)), float(a0), float(np.mean(err**2))


def sinusoid_residuals(p, t, y):
    A, w, phi, A0 = p
    return y - sinusoid(t, A, w, phi, A0)


def fit_sinusoid_freq(y: np.ndarray, fs: float, min_bpm: float, max_bpm: float) -> float:
    """f(t) = A·sin(ωt+φ)+A₀ → частота (сетка по амплитуде + least_squares, без БПФ).

    Частота дискретизации при прореживании выбирается так, чтобы она была
    не менее 8× максимальной искомой частоты (требование теоремы Котельникова).
    """
    n = y.size
    t_all = np.arange(n) / fs
    w_min, w_max = 2 * np.pi * min_bpm / 60, 2 * np.pi * max_bpm / 60

    # Минимум 8× oversampling максимальной возможной ЧСС
    max_freq_hz = max_bpm / 60.0
    step = max(1, int(fs / (max_freq_hz * 8)))
    t, yd = t_all[::step], y[::step]

    fits = [linear_fit_at_omega(t, yd, float(w)) for w in np.linspace(w_min, w_max, 200)]
    # Выбираем частоту с максимальной амплитудой — аналог argmax(|FFT|) в диапазоне bpm.
    A, w0, phi, a0, _ = max(fits, key=lambda f: f[0])
    fallback = abs(w0) / (2 * np.pi)
    w_lo, w_hi = max(w_min, w0 * 0.85), min(w_max, w0 * 1.15)

    try:
        res = least_squares(
            sinusoid_residuals,
            x0=[A, w0, phi, a0],
            args=(t, yd),
            bounds=([-np.inf, w_lo, -np.pi, -np.inf], [np.inf, w_hi, np.pi, np.inf]),
            max_nfev=3000,
            method="trf",
        )
        return abs(res.x[1]) / (2 * np.pi)
    except Exception:
        return fallback


def save_fig(fig, path: Path | str, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
