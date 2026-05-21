import kagglehub
from io_signals import resolve_ecg_dat_path

if __name__ == "__main__":
    path = kagglehub.dataset_download("ahmadsaeed1007/ecg-signal")
    print("Path to dataset files:", path)
    print("ECG.dat:", resolve_ecg_dat_path(path))