"""Константы, загрузка сигналов dNNNN и эталонной ЧСС из XLS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from struct import unpack

import numpy as np
import pandas as pd

SAMPLE_RATE_HZ = 100
SAMPLES_PER_SIGNAL = 10_000
HR_MIN_BPM = 40
HR_MAX_BPM = 220
DATA_DIR = "data"


@dataclass(frozen=True)
class Dataset:
    signals: list[np.ndarray]
    names: list[str]
    dir: Path


def read_signal(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        raw = f.read(2 * SAMPLES_PER_SIGNAL)
    if len(raw) < 2 * SAMPLES_PER_SIGNAL:
        raise ValueError(f"Файл {path}: недостаточно отсчётов")
    return np.asarray(unpack(f"{SAMPLES_PER_SIGNAL}H", raw), dtype=np.float64)


def find_signals_dir(data_dir: Path) -> Path:
    best: tuple[int, Path] | None = None
    for candidate in [data_dir, *data_dir.rglob("*")]:
        if not candidate.is_dir():
            continue
        count = sum(
            1
            for p in candidate.iterdir()
            if p.is_file() and p.name.lower().startswith("d") and p.suffix.lower() != ".tmp"
        )
        if count and (best is None or count > best[0]):
            best = (count, candidate)
    if best:
        return best[1]
    raise FileNotFoundError(f"Не найдены файлы dNNNN в {data_dir}")


def load_data(lab_dir: Path | None = None) -> Dataset:
    """Загружает сигналы из furie vs ssa/data/."""
    base = (lab_dir or Path.cwd()).resolve()
    repo = base.parent
    for data_dir in (repo / DATA_DIR, repo / "furie vs ssa" / DATA_DIR, base / DATA_DIR):
        if not data_dir.is_dir():
            continue
        try:
            signals_dir = find_signals_dir(data_dir)
            files = sorted(
                p for p in signals_dir.glob("d*")
                if p.is_file() and p.suffix.lower() != ".tmp"
            )
            return Dataset(
                signals=[read_signal(p) for p in files],
                names=[p.name for p in files],
                dir=signals_dir,
            )
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Положите файлы dNNNN в {DATA_DIR}/")


def load_reference_hr(xls_path: Path, names: list[str]) -> np.ndarray:
    df = pd.read_excel(xls_path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    name_col, hr_col = cols.get("имя файла"), cols.get("чсс")
    if not name_col or not hr_col:
        raise KeyError("В xls нужны колонки 'Имя файла' и 'Чсс'")

    mapping = {}
    for _, row in df[[name_col, hr_col]].dropna(subset=[name_col]).iterrows():
        name = str(row[name_col]).strip()
        if name and pd.notna(row[hr_col]):
            mapping[name] = float(row[hr_col])
    return np.array([mapping.get(n, np.nan) for n in names], dtype=np.float64)


def find_reference_xls(signals_dir: Path) -> Path | None:
    files = sorted([*signals_dir.glob("*.xls"), *signals_dir.glob("*.xlsx")])
    return files[0] if files else None
