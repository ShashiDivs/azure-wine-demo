import pandas as pd
from sklearn.datasets import load_wine
import os

class DataIngestion:

    def __init__(self, data_path='data/raw'):
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)

    
    def load_data(self):

        print("Loading Wine Dataset")
        wine = load_wine()

        #  Create Dataframe
        df = pd.DataFrame(
            data=wine.data,
            columns = wine.feature_names
        )

        df['target'] = wine.target

        print(f"Dataset Loaded {df.shape[0]} samples, {df.shape[1]} features")
        print(f"Target classes: {wine.target_names}")

        return df, wine.feature_names, wine.target_names
    
    def save_raw_data(self, df):
        filepath = os.path.join(self.data_path, 'wine_data.csv')
        df.to_csv(filepath,index=False)
        print(f"Raw data saved to {filepath}")
        return filepath
    

if __name__ == "__main__":

    ingestion = DataIngestion()
    df, features, targets = ingestion.load_data()
    ingestion.save_raw_data(df)
    print("\nDataset info")
    print(df.info())
    print("\n Data head")
    print(df.head())