# Furier vs SSA and Veivlet

Анализ пульсовых сигналов (100 Гц, uint16).

## Установка

```bash
pip install -r requirements.txt
```

Данные — файлы `dNNNN` и XLS в `furie vs ssa/data/`.

## Запуск

```bash
cd "furie vs ssa" && python main.py
cd ../veivlet && python main.py
```

## Структура

```
lib/           — общий код (io, signal)
furie vs ssa/  — лаб. №1: fourier.py, ssa.py, plots.py, main.py
veivlet/       — лаб. №2: wavelet.py, main.py
```
