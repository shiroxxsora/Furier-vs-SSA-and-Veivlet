"""Частота главной компоненты через БПФ."""

from __future__ import annotations

import numpy as np

from lib.io import HR_MAX_BPM, HR_MIN_BPM, SAMPLE_RATE_HZ
from lib.signal import center, peak_in_band


def dominant_freq(signal: np.ndarray, fs: float = SAMPLE_RATE_HZ) -> float:
    x = center(signal)
    if x.size < 2:
        return 0.0
    amps = np.abs(np.fft.rfft(x))[1:]
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)[1:]
    return peak_in_band(amps, freqs, HR_MIN_BPM / 60, HR_MAX_BPM / 60)


def batch(signals: list[np.ndarray], fs: float = SAMPLE_RATE_HZ) -> np.ndarray:
    return np.array([dominant_freq(s, fs) for s in signals], dtype=np.float64)


def spectrum(signal: np.ndarray, fs: float = SAMPLE_RATE_HZ) -> tuple[np.ndarray, np.ndarray]:
    x = center(signal)
    amps = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    return freqs[1:], amps[1:]
