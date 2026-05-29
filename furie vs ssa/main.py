"""
Лабораторная работа №1: сравнение Фурье и SSA на пульсовых сигналах.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import HR_MAX_BPM, HR_MIN_BPM, SAMPLE_RATE_HZ, SAMPLES_PER_SIGNAL, SSA_WINDOW_LENGTH
from fourier_analysis import dominant_frequencies_batch
from io_signals import (
    download_dataset,
    extract_dataset_zip,
    load_reference_hr_from_xls,
    resolve_signals_dir,
    resolve_reference_xls,
    load_signals_from_directory,
    load_signals_from_dat,
    resolve_ecg_dat_path,
)
from plot_results import save_comparison_plot, save_reference_comparison_plot, save_results_table
from ssa_analysis import dominant_frequencies_ssa_batch
from visualize import save_demo_reports


def parse_args():
    parser = argparse.ArgumentParser(description="Сравнение Фурье и SSA (100 Гц, uint16).")
    parser.add_argument("--dat-path", type=Path, default=None)
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=None,
        help="Папка с сигналами dNNNN (например, из AD_Dudin).",
    )
    parser.add_argument(
        "--reference-xls",
        type=Path,
        default=None,
        help="XLS-файл с колонками 'Имя файла' и 'Чсс'.",
    )
    parser.add_argument(
        "--dataset-zip",
        type=Path,
        default=None,
        help="Путь к zip-архиву датасета (внутри должен быть ECG.dat).",
    )
    parser.add_argument("--max-signals", type=int, default=None)
    parser.add_argument(
        "--min-bpm",
        type=float,
        default=HR_MIN_BPM,
        help="Нижняя граница физиологического диапазона для поиска пика.",
    )
    parser.add_argument(
        "--max-bpm",
        type=float,
        default=HR_MAX_BPM,
        help="Верхняя граница физиологического диапазона для поиска пика.",
    )
    parser.add_argument(
        "--ssa-window-length",
        type=int,
        default=SSA_WINDOW_LENGTH,
        help="Длина окна SSA (меньше -> быстрее, но грубее).",
    )
    parser.add_argument(
        "--ssa-decimate",
        type=int,
        default=1,
        help="Прореживание для SSA: 2 = каждый 2-й отсчёт, 4 = каждый 4-й.",
    )
    parser.add_argument(
        "--ssa-max-samples",
        type=int,
        default=None,
        help="Ограничить число отсчётов на сигнал только для SSA.",
    )
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

    signal_names = None
    reference_bpm = None
    resolved_signals_dir = None

    if args.signals_dir is not None:
        resolved_signals_dir = args.signals_dir
        print(f"Загрузка сигналов из папки: {resolved_signals_dir}")
        signals, signal_names = load_signals_from_directory(
            resolved_signals_dir,
            max_signals=args.max_signals,
        )
        dat_path = None
    elif args.dat_path is not None:
        dat_path = args.dat_path
        signals = load_signals_from_dat(dat_path, max_signals=args.max_signals)
    elif args.dataset_zip is not None:
        print(f"Распаковка архива: {args.dataset_zip}")
        dataset_dir = extract_dataset_zip(args.dataset_zip)
        try:
            dat_path = resolve_ecg_dat_path(dataset_dir)
            signals = load_signals_from_dat(dat_path, max_signals=args.max_signals)
        except FileNotFoundError:
            resolved_signals_dir = resolve_signals_dir(dataset_dir)
            print(f"Найден формат dNNNN, загрузка из: {resolved_signals_dir}")
            signals, signal_names = load_signals_from_directory(
                resolved_signals_dir,
                max_signals=args.max_signals,
            )
            dat_path = None
    else:
        print("Загрузка датасета Kaggle...")
        dataset_dir = download_dataset()
        dat_path = resolve_ecg_dat_path(dataset_dir)
        signals = load_signals_from_dat(dat_path, max_signals=args.max_signals)

    if dat_path is not None:
        print(f"Файл сигналов: {dat_path}")
    else:
        print(f"Сигналы из каталога, файлов: {len(signals)}")

    reference_xls = args.reference_xls
    if reference_xls is None and resolved_signals_dir is not None:
        reference_xls = resolve_reference_xls(resolved_signals_dir)

    if reference_xls is not None:
        if signal_names is None:
            raise ValueError("--reference-xls поддерживается только для режима dNNNN")
        print(f"Загрузка эталонной ЧСС из XLS: {reference_xls}")
        reference_bpm = load_reference_hr_from_xls(reference_xls, signal_names)
        matched = int(np.sum(~np.isnan(reference_bpm)))
        print(f"Сопоставлено эталонных ЧСС: {matched}/{len(reference_bpm)}")

    n = len(signals)
    print(f"Выборка: N = {n}, по {SAMPLES_PER_SIGNAL} отсчётов, fs = {SAMPLE_RATE_HZ} Гц")

    print("Расчёт частот по Фурье...")
    fourier_freqs = dominant_frequencies_batch(
        signals,
        sample_rate_hz=SAMPLE_RATE_HZ,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
    )

    print("Расчёт частот по SSA...")
    ssa_freqs = dominant_frequencies_ssa_batch(
        signals,
        sample_rate_hz=SAMPLE_RATE_HZ,
        window_length=args.ssa_window_length,
        decimate=args.ssa_decimate,
        max_samples=args.ssa_max_samples,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
    )

    csv_path = args.output_dir / "fourier_ssa_frequencies.csv"
    df = save_results_table(fourier_freqs, ssa_freqs, csv_path, reference_bpm=reference_bpm)

    if not args.no_plots:
        print("Построение графиков...")
        save_comparison_plot(
            fourier_freqs,
            ssa_freqs,
            args.output_dir / "fourier_vs_ssa.png",
        )
        if reference_bpm is not None:
            save_reference_comparison_plot(
                fourier_freqs,
                ssa_freqs,
                reference_bpm,
                args.output_dir / "methods_vs_reference_hr.png",
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
    print(
        "Диапазон оценок ЧСС (Фурье), уд/мин: "
        f"{np.nanmin(df['fourier_bpm']):.1f}..{np.nanmax(df['fourier_bpm']):.1f}"
    )
    print(
        "Диапазон оценок ЧСС (SSA), уд/мин: "
        f"{np.nanmin(df['ssa_bpm']):.1f}..{np.nanmax(df['ssa_bpm']):.1f}"
    )
    if reference_bpm is not None:
        valid = df["ref_hr_bpm"].notna()
        if valid.any():
            n_valid = int(valid.sum())
            print(
                "Средняя |Фурье - ЧСС| (уд/мин), "
                f"по {n_valid} совпадениям: {np.mean(np.abs(df.loc[valid, 'fourier_diff_bpm'])):.2f}"
            )
            print(
                "Средняя |SSA - ЧСС| (уд/мин), "
                f"по {n_valid} совпадениям: {np.mean(np.abs(df.loc[valid, 'ssa_diff_bpm'])):.2f}"
            )
    print(f"Таблица:  {csv_path}")
    if not args.no_plots:
        print(f"Графики в: {args.output_dir.resolve()}")
        print("  - fourier_vs_ssa.png       (точки + биссектриса)")
        if reference_bpm is not None:
            print("  - methods_vs_reference_hr.png (Фурье/SSA против ЧСС из XLS)")
        print("  - batch_overview.png       (сводка + гистограммы)")
        print("  - demo_signal_<i>.png      (1 с крупно + обзор 30 с)")
        print("  - ssa_components_<i>.png   (компоненты SSA, 1 с)")


if __name__ == "__main__":
    main()
