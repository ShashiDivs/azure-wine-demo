import pickle

with open('data/processed/scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(scaler)

import pandas as pd

df = pd.read_csv("data/raw/wine_data.csv")


print(df.head())