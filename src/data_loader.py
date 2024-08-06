# data_loader.py

import pandas as pd

def load_csv(file_path):
    return pd.read_csv(file_path)

def load_document(file_path):
    with open(file_path, 'r') as file:
        return file.read()
