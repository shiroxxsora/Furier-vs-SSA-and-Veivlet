"""Распределение сигналов по полу носителя (два подкаталога)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import GENDER_LABELS_FILE


def load_gender_labels(n_signals: int, labels_path: Path | None = None) -> list[str]:
    """
    Метки male/female из CSV или правило по умолчанию.

    В Kaggle-датасете пол не указан: чётный индекс — male, нечётный — female.
    """
    path = labels_path or Path(GENDER_LABELS_FILE)
    if path.is_file():
        df = pd.read_csv(path)
        if "signal_index" not in df.columns or "gender" not in df.columns:
            raise ValueError("CSV: столбцы signal_index, gender")
        mapping = dict(
            zip(
                df["signal_index"].astype(int),
                df["gender"].astype(str).str.lower(),
            )
        )
        return [_normalize_gender(mapping.get(i, "male" if i % 2 == 0 else "female")) for i in range(n_signals)]

    return ["male" if i % 2 == 0 else "female" for i in range(n_signals)]


def _normalize_gender(value: str) -> str:
    v = value.strip().lower()
    if v in ("m", "male", "м", "муж", "мужской"):
        return "male"
    if v in ("f", "female", "ж", "жен", "женский"):
        return "female"
    raise ValueError(f"Неизвестная метка пола: {value}")
