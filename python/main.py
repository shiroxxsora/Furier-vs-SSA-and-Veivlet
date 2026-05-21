"""
Лабораторная работа №1: сравнение Фурье и SSA на пульсовых сигналах.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import SAMPLE_RATE_HZ, SAMPLES_PER_SIGNAL
from fourier_analysis import dominant_frequencies_batch
from io_signals import download_dataset, load_signals_from_dat, resolve_ecg_dat_path
from plot_results import save_comparison_plot, save_results_table
from ssa_analysis import dominant_frequencies_ssa_batch
from visualize import save_demo_reports


def parse_args():
    parser = argparse.ArgumentParser(description="Сравнение Фурье и SSA (100 Гц, uint16).")
    parser.add_argument("--dat-path", type=Path, default=None)
    parser.add_argument("--max-signals", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument(
        "--demo-indices",
        type=int,
        nargs="*",
        default=None,
        help="Индексы сигналов для подробных графиков (по умолчанию 0, N/2, N-1).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Только CSV, без графиков.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dat_path is None:
        print("Загрузка датасета Kaggle...")
        dataset_dir = download_dataset()
        dat_path = resolve_ecg_dat_path(dataset_dir)
    else:
        dat_path = args.dat_path

    print(f"Файл сигналов: {dat_path}")
    signals = load_signals_from_dat(dat_path, max_signals=args.max_signals)
    n = len(signals)
    print(f"Выборка: N = {n}, по {SAMPLES_PER_SIGNAL} отсчётов, fs = {SAMPLE_RATE_HZ} Гц")

    print("Расчёт частот по Фурье...")
    fourier_freqs = dominant_frequencies_batch(signals)

    print("Расчёт частот по SSA...")
    ssa_freqs = dominant_frequencies_ssa_batch(signals)

    csv_path = args.output_dir / "fourier_ssa_frequencies.csv"
    df = save_results_table(fourier_freqs, ssa_freqs, csv_path)

    if not args.no_plots:
        print("Построение графиков...")
        save_comparison_plot(
            fourier_freqs, ssa_freqs, args.output_dir / "fourier_vs_ssa.png"
        )
        save_demo_reports(
            signals,
            fourier_freqs,
            ssa_freqs,
            args.output_dir,
            demo_indices=args.demo_indices,
        )

    print("\nПервые 10 значений:")
    print(df.head(10).to_string(index=False))
    print(f"\nСредняя |разница| (Гц): {np.mean(np.abs(df['diff_hz'])):.4f}")
    print(f"Таблица:  {csv_path}")
    if not args.no_plots:
        print(f"Графики в: {args.output_dir.resolve()}")
        print("  - fourier_vs_ssa.png       (точки + биссектриса)")
        print("  - batch_overview.png       (сводка + гистограммы)")
        print("  - demo_signal_<i>.png      (1 с крупно + обзор 30 с)")
        print("  - ssa_components_<i>.png   (компоненты SSA, 1 с)")


if __name__ == "__main__":
    main()
