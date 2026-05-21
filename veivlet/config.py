"""Параметры лабораторной №2: дискретное вейвлет-преобразование."""

SAMPLE_RATE_HZ = 100
SAMPLES_PER_SIGNAL = 10_000

KAGGLE_DATASET = "ahmadsaeed1007/ecg-signal"
ECG_DAT_FILENAME = "ECG.dat"

# K базисных вейвлетов (PyWavelets)
WAVELET_BASES = ["db4", "sym5", "coif3", "haar"]

# Макс. уровень DWT (масштаб a = 2^j); None — автоматически по длине сигнала
DWT_MAX_LEVEL = None

OUTPUT_ROOT = "output"
GENDER_DIRS = {"male": "male", "female": "female"}
GENDER_LABELS_FILE = "gender_labels.csv"
