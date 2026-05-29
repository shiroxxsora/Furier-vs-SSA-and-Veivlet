# Furier vs SSA and Veivlet

Лабораторные работы по анализу пульсовых сигналов (ECG, 100 Гц, uint16).

**Репозиторий:** https://github.com/shiroxxsora/Furier-vs-SSA-and-Veivlet

| Папка | Работа |
|-------|--------|
| [furie vs ssa](furie%20vs%20ssa/) | №1 — Фурье и SSA |
| [veivlet](veivlet/) | №2 — вейвлет-спектры (PyWavelets) |

## Установка

```bash
git clone https://github.com/shiroxxsora/Furier-vs-SSA-and-Veivlet.git
cd Furier-vs-SSA-and-Veivlet
pip install -r requirements.txt
```

## Запуск

```bash
cd "furie vs ssa"
python main.py

cd veivlet
python main.py
```

Локальный архив датасета (например, `d:\Downloads\AD_Dudin.zip`):

```bash
cd "furie vs ssa"
python main.py --dataset-zip "d:\Downloads\AD_Dudin.zip"

cd ../veivlet
python main.py --dataset-zip "d:\Downloads\AD_Dudin.zip"
```

Датасет по умолчанию: [ECG Signal](https://www.kaggle.com/datasets/ahmadsaeed1007/ecg-signal) (скачивается через `kagglehub`).
