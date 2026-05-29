"""Чтение пульсовых сигналов (uint16, 100 Гц)."""

from __future__ import annotations

from pathlib import Path
from struct import unpack
from zipfile import ZipFile

import kagglehub
import numpy as np

from config import ECG_DAT_FILENAME, KAGGLE_DATASET, SAMPLES_PER_SIGNAL


def _maybe_fix_mojibake_name(name: str) -> str:
    try:
        fixed = name.encode("cp437").decode("cp866")
    except UnicodeError:
        return name
    has_cyrillic = any("А" <= ch <= "я" or ch in "Ёё" for ch in fixed)
    return fixed if has_cyrillic else name


def normalize_extracted_names(root_dir):
    root = Path(root_dir)
    paths = sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for path in paths:
        fixed_name = _maybe_fix_mojibake_name(path.name)
        if fixed_name == path.name:
            continue
        target = path.with_name(fixed_name)
        if target.exists():
            if path.is_file():
                path.unlink(missing_ok=True)
                continue
            if path.is_dir() and target.is_dir():
                for child in list(path.iterdir()):
                    child_target = target / child.name
                    if not child_target.exists():
                        child.rename(child_target)
                try:
                    path.rmdir()
                except OSError:
                    pass
            continue
        path.rename(target)


def extract_dataset_zip(zip_path, extract_dir=None):
    archive = Path(zip_path)
    if not archive.is_file():
        raise FileNotFoundError(f"Архив не найден: {archive}")
    target_dir = (
        Path(extract_dir)
        if extract_dir is not None
        else Path(__file__).resolve().parent / "data" / archive.stem
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zip_file:
        zip_file.extractall(target_dir)
    normalize_extracted_names(target_dir)
    return target_dir


def resolve_ecg_dat_path(dataset_dir=None):
    root = (
        Path(dataset_dir)
        if dataset_dir
        else Path(kagglehub.dataset_download(KAGGLE_DATASET))
    )
    direct = root / ECG_DAT_FILENAME
    if direct.is_file():
        return direct
    matches = list(root.rglob(ECG_DAT_FILENAME))
    if not matches:
        raise FileNotFoundError(f"{ECG_DAT_FILENAME} не найден в {root}")
    return matches[0]


def load_signals_from_dat(dat_path=None, max_signals=None):
    path = Path(dat_path) if dat_path else resolve_ecg_dat_path()
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


def read_signal_uint16(filename, n_samples=SAMPLES_PER_SIGNAL):
    """Чтение uint16 по методичке: np.array(unpack('10000H', F.read()))."""
    path = Path(filename)
    with open(path, "rb") as handle:
        raw = handle.read(2 * n_samples)
    if len(raw) < 2 * n_samples:
        raise ValueError(f"Файл {path}: недостаточно отсчётов")
    return np.asarray(unpack(f"{n_samples}H", raw), dtype=np.float64)


def load_signals_from_directory(signals_dir, pattern="d*", max_signals=None):
    """Загрузка сигналов из папки с файлами вида dNNNN (без расширения)."""
    root = Path(signals_dir)
    files = [
        p
        for p in sorted(root.glob(pattern))
        if p.is_file() and p.suffix.lower() != ".tmp"
    ]
    if max_signals is not None:
        files = files[:max_signals]
    signals = [read_signal_uint16(path) for path in files]
    return signals


def resolve_signals_dir(dataset_dir):
    """Находит подпапку с файлами dNNNN внутри dataset_dir."""
    root = Path(dataset_dir)
    candidates = [root, *[p for p in root.iterdir() if p.is_dir()]]
    scored = []
    for candidate in candidates:
        files_count = sum(
            1
            for p in candidate.iterdir()
            if p.is_file() and p.name.lower().startswith("d") and p.suffix.lower() != ".tmp"
        )
        if files_count == 0:
            continue
        is_mojibake = _maybe_fix_mojibake_name(candidate.name) != candidate.name
        scored.append((files_count, not is_mojibake, candidate))
    if scored:
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return scored[0][2]
    raise FileNotFoundError(f"Не найдена папка с файлами dNNNN в {root}")
