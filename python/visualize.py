"""Наглядные графики: сигнал, спектр Фурье, компоненты SSA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import SAMPLE_RATE_HZ, SSA_WINDOW_LENGTH
from fourier_analysis import compute_fourier_spectrum, dominant_frequency_fourier
from ssa_analysis import (
    compute_ssa_components,
    dominant_frequency_ssa,
    pick_dominant_component_index,
)

# Окна для временных рядов на графиках
PLOT_ZOOM_SECONDS = 1.0
PLOT_OVERVIEW_SECONDS = 30.0


def _time_axis(n_samples: int, sample_rate_hz: float) -> np.ndarray:
    return np.arange(n_samples) / sample_rate_hz


def _n_samples_for_seconds(seconds: float, sample_rate_hz: float, max_samples: int) -> int:
    return min(max_samples, max(1, int(seconds * sample_rate_hz)))


def save_signal_demo(
    signal: np.ndarray,
    signal_index: int,
    output_path: str | Path,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    window_length: int = SSA_WINDOW_LENGTH,
    zoom_seconds: float = PLOT_ZOOM_SECONDS,
    overview_seconds: float = PLOT_OVERVIEW_SECONDS,
) -> None:
    """
    Один сигнал: пульс и SSA на 1 с (крупно), обзор на 30 с, спектр и частоты.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(signal, dtype=np.float64)
    n_zoom = _n_samples_for_seconds(zoom_seconds, sample_rate_hz, x.size)
    n_over = _n_samples_for_seconds(overview_seconds, sample_rate_hz, x.size)
    t_zoom = _time_axis(n_zoom, sample_rate_hz)
    t_over = _time_axis(n_over, sample_rate_hz)

    fourier_hz = dominant_frequency_fourier(x, sample_rate_hz)
    ssa_hz = dominant_frequency_ssa(x, sample_rate_hz, window_length)
    freqs, amps = compute_fourier_spectrum(x, sample_rate_hz)

    ssa_data = compute_ssa_components(x, window_length)
    dom_idx = pick_dominant_component_index(ssa_data["singular_values"])
    main_comp = ssa_data["components"][dom_idx]

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.75], hspace=0.35, wspace=0.28)
    fig.suptitle(
        f"Сигнал #{signal_index}: Фурье vs SSA (фрагмент {zoom_seconds:g} с)",
        fontsize=13,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t_zoom, x[:n_zoom], color="#1f77b4", linewidth=1.2)
    ax0.set_title(f"Пульсовый сигнал ({zoom_seconds:g} с)")
    ax0.set_xlabel("Время, с")
    ax0.set_ylabel("Амплитуда")
    ax0.grid(True, alpha=0.3)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(freqs, amps, color="#2ca02c", linewidth=0.9)
    ax1.axvline(
        fourier_hz,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Пик: {fourier_hz:.2f} Гц",
    )
    ax1.set_xlim(0, min(50, freqs[-1]))
    ax1.set_title("Спектр Фурье (|X(f)|)")
    ax1.set_xlabel("Частота, Гц")
    ax1.set_ylabel("Амплитуда")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_zoom, x[:n_zoom], color="#cccccc", linewidth=0.8, label="исходный")
    ax2.plot(
        t_zoom,
        main_comp[:n_zoom],
        color="#ff7f0e",
        linewidth=1.3,
        label=f"SSA комп. {dom_idx}",
    )
    ax2.set_title(f"SSA, осцилляторная компонента ({zoom_seconds:g} с)")
    ax2.set_xlabel("Время, с")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    bars = ax3.bar(
        ["Фурье", "SSA"],
        [fourier_hz, ssa_hz],
        color=["#d62728", "#9467bd"],
        width=0.5,
    )
    ax3.set_ylabel("Частота, Гц")
    ax3.set_title("Сравнение главных частот")
    ax3.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, [fourier_hz, ssa_hz]):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    diff = abs(ssa_hz - fourier_hz)
    ax3.text(
        0.5,
        0.95,
        f"|разница| = {diff:.2f} Гц",
        transform=ax3.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(t_over, x[:n_over], color="#1f77b4", linewidth=0.6, alpha=0.9)
    ax4.axvspan(0, zoom_seconds, color="yellow", alpha=0.2, label=f"окно {zoom_seconds:g} с")
    ax4.set_title(f"Обзор сигнала ({overview_seconds:g} с), жёлтая зона — фрагмент выше")
    ax4.set_xlabel("Время, с")
    ax4.set_ylabel("Амплитуда")
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, alpha=0.3)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_ssa_components_plot(
    signal: np.ndarray,
    output_path: str | Path,
    n_components: int = 6,
    zoom_seconds: float = PLOT_ZOOM_SECONDS,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    window_length: int = SSA_WINDOW_LENGTH,
) -> None:
    """Первые n компонент SSA (фрагмент zoom_seconds)."""
    output_path = Path(output_path)
    x = np.asarray(signal, dtype=np.float64)
    n_show = _n_samples_for_seconds(zoom_seconds, sample_rate_hz, x.size)
    t = _time_axis(n_show, sample_rate_hz)

    data = compute_ssa_components(x, window_length, max_components=n_components)
    comps = data["components"]
    sv = data["singular_values"]

    fig, axes = plt.subplots(n_components, 1, figsize=(11, 2.0 * n_components), sharex=True)
    if n_components == 1:
        axes = [axes]
    fig.suptitle(
        f"Компоненты SSA ({zoom_seconds:g} с, L={data['window_length']})",
        fontsize=12,
    )

    for i, ax in enumerate(axes):
        ax.plot(t, comps[i][:n_show], linewidth=1.0)
        ax.set_ylabel(f"i={i}")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.99,
            0.85,
            f"sigma={sv[i]:.0f}",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
        )

    axes[-1].set_xlabel("Время, с")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_batch_overview(
    fourier_freqs: np.ndarray,
    ssa_freqs: np.ndarray,
    output_path: str | Path,
) -> None:
    """Сводка по выборке: scatter + гистограммы + разница частот."""
    output_path = Path(output_path)
    diff = ssa_freqs - fourier_freqs

    fig = plt.figure(figsize=(12, 10))
    g = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax_sc = fig.add_subplot(g[0, :])
    ax_sc.scatter(fourier_freqs, ssa_freqs, alpha=0.75, edgecolors="k", linewidths=0.3)
    max_val = max(float(np.max(fourier_freqs)), float(np.max(ssa_freqs)), 1e-6)
    ax_sc.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="y = x")
    ax_sc.set_xlabel("Частота Фурье, Гц")
    ax_sc.set_ylabel("Частота SSA, Гц")
    ax_sc.set_title("Сравнение по всей выборке")
    ax_sc.legend()
    ax_sc.grid(True, alpha=0.3)
    ax_sc.set_aspect("equal", adjustable="box")

    ax_h1 = fig.add_subplot(g[1, 0])
    ax_h1.hist(fourier_freqs, bins=20, color="#d62728", alpha=0.75, edgecolor="white")
    ax_h1.set_title("Распределение частот Фурье")
    ax_h1.set_xlabel("Гц")
    ax_h1.grid(True, alpha=0.3)

    ax_h2 = fig.add_subplot(g[1, 1])
    ax_h2.hist(diff, bins=20, color="#9467bd", alpha=0.75, edgecolor="white")
    ax_h2.axvline(0, color="k", linestyle="--", linewidth=1)
    ax_h2.set_title("Разница SSA - Фурье")
    ax_h2.set_xlabel("Гц")
    ax_h2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_demo_reports(
    signals: list,
    fourier_freqs: np.ndarray,
    ssa_freqs: np.ndarray,
    output_dir: str | Path,
    demo_indices: list[int] | None = None,
    zoom_seconds: float = PLOT_ZOOM_SECONDS,
) -> None:
    """Сохраняет наглядные графики в output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if demo_indices is None:
        n = len(signals)
        demo_indices = [0, n // 2, n - 1] if n >= 3 else list(range(n))

    for idx in demo_indices:
        if 0 <= idx < len(signals):
            save_signal_demo(
                signals[idx],
                idx,
                output_dir / f"demo_signal_{idx}.png",
                zoom_seconds=zoom_seconds,
            )
            save_ssa_components_plot(
                signals[idx],
                output_dir / f"ssa_components_{idx}.png",
                zoom_seconds=zoom_seconds,
            )

    save_batch_overview(fourier_freqs, ssa_freqs, output_dir / "batch_overview.png")