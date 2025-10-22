import pandas as pd
import re
import os

def read_txt(file_path):

    try:
        with open(file_path, 'r', encoding='cp1252') as f:
            lines = f.readlines()

        start_index = next((i for i, l in enumerate(lines) if re.match(r'^\s*\d+', l)), None)
        if start_index is None:
            raise ValueError("No numeric data found in file.")

        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            skiprows=start_index - 2 if start_index > 1 else 0,
            encoding='cp1252'
        )

        df.columns = [c.strip().replace('\ufeff', '') for c in df.columns]

        if not {'Ts', 'Weight'} <= set(df.columns):
            raise ValueError(f"Columns found: {df.columns.tolist()} — expected 'Ts' and 'Weight'.")

        temp = pd.to_numeric(df['Ts'], errors='coerce').dropna().values
        weight = pd.to_numeric(df['Weight'], errors='coerce').dropna().values

        return temp, weight

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None


def load_all_txt_files(data_dir):
    
    txt_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.txt')]
    if not txt_files:
        print(f"No .txt files found in {data_dir}")
        return []

    data_list = []
    for f in txt_files:
        path = os.path.join(data_dir, f)
        temp, weight = read_txt(path)
        if temp is not None and weight is not None:
            data_list.append((f, temp, weight))
    return data_list
