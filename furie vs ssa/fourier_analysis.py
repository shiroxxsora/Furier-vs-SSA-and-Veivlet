"""Преобразование Фурье: частота главной спектральной компоненты."""

from __future__ import annotations

import numpy as np

from config import SAMPLE_RATE_HZ, SIGNAL_DURATION_S


def _dft_amplitude_phase(x, k):
    """
    Амплитуда k-й гармоники: f_k = k / T, T — длительность сигнала (см. задание).
    """
    n = x.size
    n_idx = np.arange(n, dtype=np.float64)
    angle = -2.0 * np.pi * k * n_idx / n
    coeff = np.sum(x * np.exp(1j * angle))
    amplitude = (2.0 / n) * np.abs(coeff)
    phase = np.angle(coeff)
    return amplitude, phase


def dominant_frequency_fourier(signal, sample_rate_hz=SAMPLE_RATE_HZ, use_numpy_fft=True):
    """Частота главной компоненты (максимум амплитуды спектра, k >= 1)."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size
    if n < 2:
        return 0.0

    if use_numpy_fft:
        amplitudes = np.abs(np.fft.rfft(x))
        k_peak = int(np.argmax(amplitudes[1:]) + 1)
        return k_peak * sample_rate_hz / n

    k_max = n // 2
    best_k, best_amp = 1, -1.0
    for k in range(1, k_max + 1):
        amp, _ = _dft_amplitude_phase(x, k)
        if amp > best_amp:
            best_amp, best_k = amp, k
    return best_k / SIGNAL_DURATION_S


def dominant_frequencies_batch(signals, sample_rate_hz=SAMPLE_RATE_HZ):
    return np.array(
        [dominant_frequency_fourier(s, sample_rate_hz) for s in signals],
        dtype=np.float64,
    )

def compute_fourier_spectrum(signal, sample_rate_hz=SAMPLE_RATE_HZ):
    """Частоты и амплитуды спектра (без постоянной составляющей)."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    n = x.size
    amplitudes = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    return freqs[1:], amplitudes[1:]
