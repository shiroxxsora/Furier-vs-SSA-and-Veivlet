"""Графики и CSV для лабораторной №1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fourier
import ssa
from lib.io import SAMPLE_RATE_HZ
from lib.signal import hz_to_bpm, save_fig, time_axis

ZOOM_S = 1.0
OVERVIEW_S = 30.0


def draw_bisector(ax, mx, label="y = x", color="r"):
    ax.plot([0, mx], [0, mx], color=color, ls="--", label=label)


def save_all(
    out_dir: Path,
    signals: list[np.ndarray],
    f_hz: np.ndarray,
    s_hz: np.ndarray,
    ref_bpm: np.ndarray | None,
) -> pd.DataFrame:
    out_dir.mkdir(exist_ok=True)

    df = pd.DataFrame({
        "signal_index": np.arange(f_hz.size),
        "fourier_freq_hz": f_hz,
        "ssa_freq_hz": s_hz,
        "diff_hz": s_hz - f_hz,
        "fourier_bpm": hz_to_bpm(f_hz),
        "ssa_bpm": hz_to_bpm(s_hz),
    })
    if ref_bpm is not None:
        df["ref_hr_bpm"] = ref_bpm
        df["fourier_diff_bpm"] = df["fourier_bpm"] - ref_bpm
        df["ssa_diff_bpm"] = df["ssa_bpm"] - ref_bpm
    df.to_csv(out_dir / "fourier_ssa_frequencies.csv", index=False, encoding="utf-8-sig")

    # scatter Фурье vs SSA
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(f_hz, s_hz, alpha=0.75, edgecolors="k", linewidths=0.3)
    mx = max(float(f_hz.max()), float(s_hz.max()), 1e-6)
    draw_bisector(ax, mx, "y = x (биссектриса)")
    ax.set_xlabel("Фурье, Гц"); ax.set_ylabel("SSA, Гц")
    ax.set_title("Сравнение Фурье и SSA"); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.legend(); fig.tight_layout()
    save_fig(fig, out_dir / "fourier_vs_ssa.png")

    # vs эталон
    if ref_bpm is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        ok = ~np.isnan(ref_bpm)
        ref, fb, sb = ref_bpm[ok], hz_to_bpm(f_hz[ok]), hz_to_bpm(s_hz[ok])
        mx = max(ref.max(), fb.max(), sb.max(), 1e-6)
        for ax, y, title, c in [(axes[0], fb, "Фурье vs ЧСС", "#d62728"), (axes[1], sb, "SSA vs ЧСС", "#9467bd")]:
            ax.scatter(ref, y, alpha=0.75, edgecolors="k", linewidths=0.3, color=c)
            draw_bisector(ax, mx, color="k", label="y = x")
            ax.set_xlabel("ЧСС, уд/мин"); ax.set_ylabel("Оценка, уд/мин")
            ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend()
        fig.suptitle("Сравнение с эталонной ЧСС"); fig.tight_layout()
        save_fig(fig, out_dir / "methods_vs_reference_hr.png")

    # demo-сигналы
    n = len(signals)
    for idx in ([0, n // 2, n - 1] if n >= 3 else range(n)):
        save_demo_plot(signals[idx], idx, out_dir / f"demo_signal_{idx}.png")
        save_ssa_components_plot(signals[idx], out_dir / f"ssa_components_{idx}.png")

    # сводка
    diff = s_hz - f_hz
    fig = plt.figure(figsize=(12, 10))
    g = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])
    ax = fig.add_subplot(g[0, :])
    ax.scatter(f_hz, s_hz, alpha=0.75, edgecolors="k", linewidths=0.3)
    mx = max(float(f_hz.max()), float(s_hz.max()), 1e-6)
    draw_bisector(ax, mx); ax.set_xlabel("Фурье, Гц"); ax.set_ylabel("SSA, Гц")
    ax.set_title("Выборка"); ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
    ax = fig.add_subplot(g[1, 0]); ax.hist(f_hz, bins=20, color="#d62728", alpha=0.75)
    ax.set_title("Фурье"); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(g[1, 1]); ax.hist(diff, bins=20, color="#9467bd", alpha=0.75)
    ax.axvline(0, color="k", ls="--"); ax.set_title("SSA − Фурье"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out_dir / "batch_overview.png")

    return df


def save_demo_plot(signal, idx, path):
    fs = SAMPLE_RATE_HZ
    x = np.asarray(signal, dtype=np.float64)
    nz = min(x.size, int(ZOOM_S * fs))
    no = min(x.size, int(OVERVIEW_S * fs))
    tz, to = time_axis(nz, fs), time_axis(no, fs)
    fh = fourier.dominant_freq(x, fs)
    sh = ssa.dominant_freq(x, fs)
    freqs, amps = fourier.spectrum(x, fs)
    comps, sv, L = ssa.components(x)
    di = int(np.argmax(sv ** 2)) if sv.size else 0

    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.75], hspace=0.35, wspace=0.28)
    fig.suptitle(f"Сигнал #{idx} ({ZOOM_S:g} с)")

    ax = fig.add_subplot(gs[0, 0]); ax.plot(tz, x[:nz]); ax.set_title("Пульс"); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(gs[0, 1]); ax.plot(freqs, amps); ax.axvline(fh, c="r", ls="--", label=f"{fh:.2f} Гц")
    ax.set_xlim(0, min(50, freqs[-1])); ax.legend(); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(tz, x[:nz], c="#ccc", label="исходный")
    ax.plot(tz, comps[di][:nz], c="#ff7f0e", label=f"SSA {di}")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(["Фурье", "SSA"], [fh, sh], color=["#d62728", "#9467bd"], width=0.5)
    ax.set_ylabel("Гц"); ax.grid(True, axis="y", alpha=0.3)
    ax = fig.add_subplot(gs[2, :]); ax.plot(to, x[:no]); ax.axvspan(0, ZOOM_S, alpha=0.2, color="y")
    ax.grid(True, alpha=0.3)
    save_fig(fig, path)


def save_ssa_components_plot(signal, path, n=6):
    fs = SAMPLE_RATE_HZ
    ns = min(signal.size, int(ZOOM_S * fs))
    t = time_axis(ns, fs)
    comps, sv, L = ssa.components(signal, k=n)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Компоненты SSA ({ZOOM_S:g} с, L={L})")
    for i, ax in enumerate(axes):
        ax.plot(t, comps[i][:ns]); ax.set_ylabel(f"i={i}"); ax.grid(True, alpha=0.3)
        ax.text(0.99, 0.85, f"σ={sv[i]:.0f}", transform=ax.transAxes, ha="right", fontsize=8)
    axes[-1].set_xlabel("Время, с")
    fig.tight_layout()
    save_fig(fig, path)
