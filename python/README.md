# Лабораторная №1: Фурье и SSA

## Запуск

```bash
pip install -r ../requirements.txt
python main.py
```

Подробные графики для сигналов 0, 5, 9:

```bash
python main.py --demo-indices 0 5 9
```

## Графики (папка output/)

| Файл | Содержание |
|------|------------|
| `demo_signal_<i>.png` | Сигнал, спектр Фурье, главная компонента SSA, столбцы частот |
| `ssa_components_<i>.png` | Первые 6 элементарных компонент SSA |
| `fourier_vs_ssa.png` | Точки + биссектриса y=x |
| `batch_overview.png` | Сводный scatter + гистограммы |
| `fourier_ssa_frequencies.csv` | Таблица N частот |