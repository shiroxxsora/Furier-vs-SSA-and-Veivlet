"""Чтение пульсовых сигналов: бинарные файлы и датасет Kaggle."""

from __future__ import annotations

from pathlib import Path
from struct import unpack
from typing import Iterable

import kagglehub
import numpy as np

from config import ECG_DAT_FILENAME, KAGGLE_DATASET, SAMPLES_PER_SIGNAL


def read_signal_uint16(filename, n_samples=SAMPLES_PER_SIGNAL):
    """Чтение uint16 по методичке: np.array(unpack('10000H', F.read()))."""
    path = Path(filename)
    with open(path, "rb") as handle:
        raw = handle.read(2 * n_samples)
    if len(raw) < 2 * n_samples:
        raise ValueError(f"Файл {path}: недостаточно отсчётов")
    return np.asarray(unpack(f"{n_samples}H", raw), dtype=np.float64)


def download_dataset():
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def resolve_ecg_dat_path(dataset_dir=None):
    root = dataset_dir or download_dataset()
    direct = root / ECG_DAT_FILENAME
    if direct.is_file():
        return direct
    matches = list(root.rglob(ECG_DAT_FILENAME))
    if not matches:
        raise FileNotFoundError(f"{ECG_DAT_FILENAME} не найден в {root}")
    return matches[0]


def load_signals_from_dat(dat_path=None, max_signals=None):
    path = dat_path or resolve_ecg_dat_path()
    total = path.stat().st_size // 2
    n_signals = total // SAMPLES_PER_SIGNAL
    if max_signals is not None:
        n_signals = min(n_signals, max_signals)
    signals = []
    with open(path, "rb") as handle:
        for _ in range(n_signals):
            raw = handle.read(2 * SAMPLES_PER_SIGNAL)
            signals.append(
                np.asarray(unpack(f"{SAMPLES_PER_SIGNAL}H", raw), dtype=np.float64)
            )
    return signals


def iter_signal_files(directory, pattern="*.dat"):
    yield from sorted(Path(directory).glob(pattern))
