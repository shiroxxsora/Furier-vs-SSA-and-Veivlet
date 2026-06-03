"""Лабораторная №1: Фурье и SSA."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import fourier
import ssa
from lib.io import find_reference_xls, load_data, load_reference_hr
from plots import save_all


def main() -> None:
    lab = Path(__file__).resolve().parent
    out = lab / "output"
    out.mkdir(exist_ok=True)

    try:
        data = load_data(lab)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    print(f"Данные: {data.dir}  ({len(data.signals)} сигналов)")

    ref_bpm = None
    xls = find_reference_xls(data.dir)
    if xls:
        print(f"Эталон: {xls.name}")
        ref_bpm = load_reference_hr(xls, data.names)

    print("Фурье...")
    f_hz = fourier.batch(data.signals)
    print("SSA...")
    s_hz = ssa.batch(data.signals)

    df = save_all(out, data.signals, f_hz, s_hz, ref_bpm)

    print(f"\n|разница| ср.: {np.mean(np.abs(df['diff_hz'])):.4f} Гц")
    print(f"Результаты: {out.resolve()}")


if __name__ == "__main__":
    main()
