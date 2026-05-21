"""
Лабораторная №2: вейвлет-спектры для K базисных вейвлетов.
Выход: output/<вейвлет>/male|female/ — N спектров в каждом подкаталоге.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import OUTPUT_ROOT, WAVELET_BASES
from gender import load_gender_labels
from io_signals import load_signals_from_dat, resolve_ecg_dat_path
from wavelet_spectrum import save_wavelet_spectrum_png


def parse_args():
    parser = argparse.ArgumentParser(description="Лаб. №2: ДВП (PyWavelets).")
    parser.add_argument("--dat-path", type=Path, default=None)
    parser.add_argument("--max-signals", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / OUTPUT_ROOT,
    )
    parser.add_argument("--wavelets", nargs="*", default=None)
    parser.add_argument("--gender-csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dat_path = args.dat_path or resolve_ecg_dat_path()
    wavelets = args.wavelets or WAVELET_BASES

    print(f"Данные: {dat_path}")
    print(f"Вейвлеты (K={len(wavelets)}): {', '.join(wavelets)}")

    signals = load_signals_from_dat(dat_path, max_signals=args.max_signals)
    n = len(signals)
    genders = load_gender_labels(n, args.gender_csv)
    print(f"Сигналов N={n}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    for wavelet in wavelets:
        wavelet_dir = args.output_root / wavelet
        for gender in ("male", "female"):
            (wavelet_dir / gender).mkdir(parents=True, exist_ok=True)

        print(f"  Вейвлет {wavelet}...")
        for idx, (signal, gender) in enumerate(zip(signals, genders)):
            out_file = wavelet_dir / gender / f"signal_{idx:04d}.png"
            save_wavelet_spectrum_png(signal, out_file, wavelet, idx)

    root = args.output_root.resolve()
    print(f"Готово: {root}")
    for w in wavelets:
        print(f"  {root / w}/male|female/ — по {n} файлов")


if __name__ == "__main__":
    main()
