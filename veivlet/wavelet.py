"""Вейвлет-спектры (DWT)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt

from lib.io import SAMPLE_RATE_HZ
from lib.signal import center, save_fig, time_axis

WAVELETS = ["db4", "sym5", "coif3", "haar"]
PLOT_SECONDS = 30.0


def save_png(signal, path: Path | str, wavelet: str, idx: int, fs: float = SAMPLE_RATE_HZ):
    x = np.asarray(signal, dtype=np.float64)
    x = x[: int(PLOT_SECONDS * fs)]
    x = center(x)
    n = x.size

    lvl = pywt.dwt_max_level(n, wavelet)
    coeffs = pywt.wavedec(x, wavelet, level=lvl, mode="symmetric")
    labels = [f"A{lvl}"] + [f"D{j}" for j in range(lvl, 0, -1)]

    def upsample(c):
        m = np.abs(c)
        if m.size == n:
            return m
        return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, m.size), m)

    matrix = np.vstack([upsample(c) for c in coeffs])
    time = time_axis(n, fs)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 2.2]})
    axes[0].plot(time, x, lw=0.7); axes[0].set_title(f"Сигнал #{idx}"); axes[0].grid(True, alpha=0.3)
    im = axes[1].imshow(matrix, aspect="auto", cmap="magma",
                         extent=[time[0], time[-1], len(labels) - 0.5, -0.5], interpolation="bilinear")
    axes[1].set_yticks(range(len(labels))); axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Время, с"); axes[1].set_ylabel("Уровень DWT")
    axes[1].set_title(f"Базис {wavelet}")
    fig.colorbar(im, ax=axes[1], label="|коэфф.|")
    fig.tight_layout()
    save_fig(fig, path, dpi=80)
