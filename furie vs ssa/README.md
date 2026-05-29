# Лабораторная №1: Фурье и SSA

## Запуск

```bash
pip install -r ../requirements.txt
python main.py
```

Запуск из zip-архива датасета:

```bash
python main.py --dataset-zip "d:\Downloads\AD_Dudin.zip"
```

Сравнение с исходной ЧСС из XLS (колонки `Имя файла` и `Чсс`, файл ищется автоматически):

```bash
python main.py --dataset-zip "d:\Downloads\AD_Dudin.zip"
```

Быстрый режим (обычно заметно быстрее на SSA):

```bash
python main.py --dataset-zip "d:\Downloads\AD_Dudin.zip" --no-plots --ssa-decimate 4 --ssa-max-samples 2500 --ssa-window-length 120
```

По умолчанию пики ищутся в физиологическом диапазоне `40..220` уд/мин (можно изменить через `--min-bpm` и `--max-bpm`).

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