"""Визуализация сравнения частот Фурье и SSA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_comparison_plot(fourier_freqs, ssa_freqs, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(fourier_freqs, ssa_freqs, alpha=0.75, edgecolors="k", linewidths=0.3)

    max_val = max(
        float(np.max(fourier_freqs)) if fourier_freqs.size else 0.0,
        float(np.max(ssa_freqs)) if ssa_freqs.size else 0.0,
        1e-6,
    )
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="y = x (биссектриса)")

    ax.set_xlabel("Частота главной компоненты Фурье, Гц")
    ax.set_ylabel("Частота главной гармоники SSA, Гц")
    ax.set_title("Сравнение Фурье и SSA на выборке пульсовых сигналов")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_reference_comparison_plot(fourier_freqs, ssa_freqs, reference_bpm, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    valid = ~np.isnan(reference_bpm)
    ref = reference_bpm[valid]
    fourier_bpm = fourier_freqs[valid] * 60.0
    ssa_bpm = ssa_freqs[valid] * 60.0

    max_bpm = max(
        float(np.max(ref)) if ref.size else 0.0,
        float(np.max(fourier_bpm)) if fourier_bpm.size else 0.0,
        float(np.max(ssa_bpm)) if ssa_bpm.size else 0.0,
        1e-6,
    )
    for ax, y, title, color in [
        (axes[0], fourier_bpm, "Фурье vs ЧСС (из XLS)", "#d62728"),
        (axes[1], ssa_bpm, "SSA vs ЧСС (из XLS)", "#9467bd"),
    ]:
        ax.scatter(ref, y, alpha=0.75, edgecolors="k", linewidths=0.3, color=color)
        ax.plot([0, max_bpm], [0, max_bpm], "k--", linewidth=1.2, label="y = x")
        ax.set_xlabel("ЧСС из XLS, уд/мин")
        ax.set_ylabel("Оценка метода, уд/мин")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    fig.suptitle("Сравнение оценок частоты с исходной ЧСС")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_results_table(fourier_freqs, ssa_freqs, output_path, reference_bpm=None):
    data = {
        "signal_index": np.arange(fourier_freqs.size),
        "fourier_freq_hz": fourier_freqs,
        "ssa_freq_hz": ssa_freqs,
        "diff_hz": ssa_freqs - fourier_freqs,
        "fourier_bpm": fourier_freqs * 60.0,
        "ssa_bpm": ssa_freqs * 60.0,
    }
    if reference_bpm is not None:
        data["ref_hr_bpm"] = reference_bpm
        data["fourier_diff_bpm"] = data["fourier_bpm"] - reference_bpm
        data["ssa_diff_bpm"] = data["ssa_bpm"] - reference_bpm
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df
