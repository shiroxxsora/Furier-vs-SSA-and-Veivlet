"""Вейвлет-спектры на основе дискретного вейвлет-преобразования (DWT).

Метод:
  pywt.wavedec(x, wavelet, level=J) — прямое ДВП.
  Возвращает список коэффициентов [cA_J, cD_J, cD_{J-1}, ..., cD_1]:
    cA_J — аппроксимирующие коэффициенты уровня J (низкие частоты),
    cD_j — детализирующие коэффициенты уровня j (полоса частот 2^j).

  Масштаб a = 2^j: чем выше уровень j, тем более низкочастотные колебания.
  Частотная полоса уровня j: [fs / 2^(j+1), fs / 2^j].
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt

from config import SAMPLE_RATE_HZ


def _upsample_to(coeff: np.ndarray, target_len: int) -> np.ndarray:
    """Растягивает массив коэффициентов до длины target_len (для скалограммы).

    DWT-коэффициенты на уровне j имеют длину ~N/2^j, поэтому для тепловой
    карты их нужно интерполировать до длины исходного сигнала N.
    """
    c = np.abs(coeff)
    if c.size == target_len:
        return c
    src_x = np.linspace(0.0, 1.0, c.size)
    dst_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst_x, src_x, c)


def compute_dwt_spectrum_matrix(
    signal: np.ndarray,
    wavelet: str,
    level: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Строит матрицу |коэффициентов| DWT размером (level+1) x N.

    Строки соответствуют уровням: [A_J, D_J, D_{J-1}, ..., D_1].
    Столбцы — временная ось (интерполированная до длины N).

    Возвращает:
      matrix — матрица абсолютных значений коэффициентов,
      labels — подписи уровней для оси Y.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()          # убираем постоянную составляющую
    n = x.size

    # Автоматически определяем максимально возможный уровень разложения
    max_level = pywt.dwt_max_level(n, wavelet)
    if max_level < 1:
        raise ValueError(f"Слишком короткий сигнал для вейвлета '{wavelet}'")
    level = min(level or max_level, max_level)

    # Прямое ДВП: возвращает [cA_J, cD_J, ..., cD_1]
    coeffs = pywt.wavedec(x, wavelet, level=level, mode="symmetric")

    # Метки уровней: A — аппроксимация, D — детализация
    labels = [f"A{level}"] + [f"D{j}" for j in range(level, 0, -1)]

    # Интерполируем каждый уровень до длины N и собираем в матрицу
    matrix = np.vstack([_upsample_to(c, n) for c in coeffs])
    return matrix, labels


def save_wavelet_spectrum_png(
    signal: np.ndarray,
    output_path: str | Path,
    wavelet: str,
    signal_index: int,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    max_plot_seconds: float | None = 30.0,
) -> None:
    """Сохраняет PNG: исходный сигнал + тепловая карта вейвлет-спектра.

    Верхний subplot — временной ряд сигнала.
    Нижний subplot — матрица |коэффициентов DWT| по уровням (скалограмма).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(signal, dtype=np.float64)

    # Ограничиваем длину для отображения (полный сигнал 100 с может быть большим)
    if max_plot_seconds is not None:
        x = x[: int(max_plot_seconds * sample_rate_hz)]

    matrix, labels = compute_dwt_spectrum_matrix(x, wavelet)
    n = x.size
    time_axis = np.arange(n) / sample_rate_hz

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 2.2]}
    )

    # --- Верхний график: исходный сигнал ---
    x_centered = x - x.mean()
    axes[0].plot(time_axis, x_centered, color="#1f77b4", linewidth=0.7)
    axes[0].set_ylabel("Амплитуда")
    axes[0].set_title(f"Сигнал #{signal_index}")
    axes[0].grid(True, alpha=0.3)

    # --- Нижний график: скалограмма (тепловая карта DWT) ---
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
    axes[1].set_ylabel("Уровень DWT (масштаб a = 2^j)")
    axes[1].set_title(f"Вейвлет-спектр (|коэфф.|), базис {wavelet}")
    fig.colorbar(im, ax=axes[1], label="|коэффициент|")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
