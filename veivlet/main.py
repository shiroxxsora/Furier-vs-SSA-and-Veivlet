"""Лабораторная №2: вейвлет-спектры."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wavelet
from lib.io import load_data


def main() -> None:
    lab = Path(__file__).resolve().parent
    out = lab / "output"
    out.mkdir(exist_ok=True)

    try:
        data = load_data(lab)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e

    print(f"Данные: {data.dir}  ({len(data.signals)} сигналов)")

    for w in wavelet.WAVELETS:
        wavelet_dir = out / w
        wavelet_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {w}...")
        for i, sig in enumerate(data.signals):
            wavelet.save_png(sig, wavelet_dir / f"signal_{i:04d}.png", w, i)

    print(f"Готово: {out.resolve()}")


if __name__ == "__main__":
    main()
