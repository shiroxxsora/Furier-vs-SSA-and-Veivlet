"""Вейвлет-спектры на основе дискретного ВП (DWT), PyWavelets."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt

from config import SAMPLE_RATE_HZ


def _upsample_coeffs(coeff: np.ndarray, target_len: int) -> np.ndarray:
    """Растягивание ряда коэффициентов до длины исходного сигнала (для скалограммы)."""
    if coeff.size == target_len:
        return np.abs(coeff)
    src_x = np.linspace(0.0, 1.0, coeff.size)
    dst_x = np.linspace(0.0, 1.0, target_len)
    return np.abs(np.interp(dst_x, src_x, coeff))


def compute_dwt_spectrum_matrix(
    signal: np.ndarray,
    wavelet: str,
    level: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Матрица |коэффициентов| DWT: строки — уровни разложения (a=2^j).

    Прямое ДВП: wavedec — вейвлет-коэффициенты на дискретных масштабах.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size

    max_level = pywt.dwt_max_level(n, wavelet)
    if max_level < 1:
        raise ValueError(f"Слишком короткий сигнал для вейвлета {wavelet}")
    level = min(level or max_level, max_level)

    coeffs = pywt.wavedec(x, wavelet, level=level, mode="symmetric")
    labels = [f"A{level}"]
    labels.extend([f"D{j}" for j in range(level, 0, -1)])

    rows = [_upsample_coeffs(c, n) for c in coeffs]
    matrix = np.vstack(rows)
    return matrix, labels


def save_wavelet_spectrum_png(
    signal: np.ndarray,
    output_path: str | Path,
    wavelet: str,
    signal_index: int,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    max_plot_seconds: float | None = 30.0,
) -> None:
    """PNG: сигнал + вейвлет-спектр (|коэффициенты DWT| по уровням)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(signal, dtype=np.float64)
    if max_plot_seconds is not None:
        n_max = int(max_plot_seconds * sample_rate_hz)
        x = x[: min(x.size, n_max)]

    matrix, labels = compute_dwt_spectrum_matrix(x, wavelet)
    n = x.size
    time_axis = np.arange(n) / sample_rate_hz

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 2.2]})

    axes[0].plot(time_axis, x - np.mean(x), color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("Амплитуда")
    axes[0].set_title(f"Сигнал #{signal_index}")
    axes[0].grid(True, alpha=0.3)

    extent = [time_axis[0], time_axis[-1], len(labels) - 0.5, -0.5]
    im = axes[1].imshow(
        matrix,
        aspect="auto",
        cmap="magma",
        extent=extent,
        interpolation="bilinear",
    )
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Время, с")
    axes[1].set_ylabel("Уровень DWT (масштаб)")
    axes[1].set_title(f"Вейвлет-спектр (|коэфф.|), базис {wavelet}")
    fig.colorbar(im, ax=axes[1], label="|коэффициент|")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
