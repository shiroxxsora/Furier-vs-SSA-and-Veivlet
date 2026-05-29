"""Преобразование Фурье: частота главной спектральной компоненты.

По методичке:
  X_k = sum_{n=0}^{N-1} x_n * exp(-j*2*pi*k*n/N)
  A_k = 2/N * |X_k|  — амплитуда k-й гармоники
  f_k = k / T        — частота k-й гармоники (T = N/fs — длительность сигнала)
"""

from __future__ import annotations

import numpy as np

from config import HR_MAX_BPM, HR_MIN_BPM, SAMPLE_RATE_HZ


def dft_amplitude_phase(x: np.ndarray, k: int) -> tuple[float, float]:
    """Прямая реализация ДПФ по формуле из методички (для k-й гармоники).

    Используется только в демонстрационных/учебных целях.
    В батч-расчётах применяется numpy.fft.rfft (в N log N вместо N^2).
    """
    n = x.size
    n_idx = np.arange(n, dtype=np.float64)
    # Комплексный коэффициент X_k
    coeff = np.sum(x * np.exp(-2j * np.pi * k * n_idx / n))
    amplitude = (2.0 / n) * abs(coeff)
    phase = float(np.angle(coeff))
    return amplitude, phase


def _peak_frequency_in_band(
    amplitudes: np.ndarray,
    freqs: np.ndarray,
    min_hz: float,
    max_hz: float,
) -> float:
    """Возвращает частоту максимального пика в диапазоне [min_hz, max_hz].

    Если в диапазоне нет ни одной частотной точки — берётся глобальный максимум.
    """
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    target_freqs = freqs[mask] if mask.any() else freqs
    target_amps = amplitudes[mask] if mask.any() else amplitudes
    return float(target_freqs[int(np.argmax(target_amps))])


def dominant_frequency_fourier(
    signal,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    min_bpm: float = HR_MIN_BPM,
    max_bpm: float = HR_MAX_BPM,
) -> float:
    """Частота главной компоненты сигнала через БПФ.

    Шаги:
      1. Вычитаем постоянную составляющую (DC-offset).
      2. Вычисляем амплитудный спектр через numpy.fft.rfft.
      3. Ищем пик в физиологическом диапазоне [min_bpm, max_bpm] уд/мин.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()          # убираем постоянную составляющую
    n = x.size
    if n < 2:
        return 0.0

    # rfft даёт только положительные частоты: 0, fs/N, 2*fs/N, ..., fs/2
    amplitudes = np.abs(np.fft.rfft(x))[1:]   # [1:] — пропускаем k=0 (DC)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)[1:]

    return _peak_frequency_in_band(
        amplitudes,
        freqs,
        min_hz=min_bpm / 60.0,
        max_hz=max_bpm / 60.0,
    )


def dominant_frequencies_batch(
    signals,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    min_bpm: float = HR_MIN_BPM,
    max_bpm: float = HR_MAX_BPM,
) -> np.ndarray:
    """Применяет dominant_frequency_fourier ко всей выборке сигналов."""
    return np.array(
        [
            dominant_frequency_fourier(s, sample_rate_hz, min_bpm, max_bpm)
            for s in signals
        ],
        dtype=np.float64,
    )


def compute_fourier_spectrum(
    signal,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (freqs, amplitudes) полного спектра без DC-составляющей.

    Используется для построения графиков в visualize.py.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    amplitudes = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate_hz)
    return freqs[1:], amplitudes[1:]
