# Лабораторная №2: вейвлет-преобразование

Дискретное вейвлет-преобразование (DWT, [PyWavelets](https://pywavelets.readthedocs.io/)) пульсовых сигналов.  
Спектр — матрица |коэффициентов| по уровням разложения (масштабы a = 2^j).

## Запуск

```bash
pip install -r ../requirements.txt
cd veivlet
python main.py
```

Свой `ECG.dat`:

```bash
python main.py --dat-path "C:\path\to\ECG.dat"
```

Другие вейвлеты (K на выбор):

```bash
python main.py --wavelets db4 sym4 haar bior2.2
```

## Выход

```
output/
  db4/
    male/   signal_0000.png ... signal_NNNN.png
    female/
  sym5/
    male/
    female/
  ...
```

По умолчанию **K = 4**: `db4`, `sym5`, `coif3`, `haar`.

## Пол носителя

В датасете Kaggle меток нет. По умолчанию: чётный индекс — `male`, нечётный — `female`.

Файл `gender_labels.csv`:

```csv
signal_index,gender
0,male
1,female
```

Запуск: `python main.py --gender-csv gender_labels.csv`
