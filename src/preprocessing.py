import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from data_ingestion import DataIngestion


class DataPreprocessor:

    def __init__(self,processed_path = 'data/processed'):
        self.processed_path = processed_path
        self.scaler = StandardScaler()
        os.makedirs(processed_path,exist_ok=True)

    def check_missing_values(self,df):

        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("Missing values found:")
            print(missing[missing > 0])
        else:
            print("No Missing values found")
        
        return df
    
    def split_features_target(self,df):

        X = df.drop('target', axis=1)
        y = df['target']
        print(f"Fetures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        return X, y
    
    def train_val_test_split(self, X, y, val_size=0.15, test_size=0.15, random_state=42):

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size,random_state=random_state, stratify=y
        )

        val_friction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_friction ,random_state=random_state, stratify=y_temp
        )

        # train:70, val:15, test15
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):

        X_train_scaled = self.scaler.fit_transform(X_train) # learn transform
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def save_scalar(self):

        scalar_path = os.path.join(self.processed_path, 'scalar.pkl')
        with open(scalar_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f" Scalar saved to {scalar_path}")

    def preprocess(self, df):

        # checking missing values
        df  = self.check_missing_values(df)

        # Split features and target
        X, y = self.split_features_target(df)

        #Train/val/test split
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_val_test_split(X, y)

        #Train/val/test split
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(X_train,X_val, X_test)

        # Save scalar

        self.save_scalar()

        return X_train_scaled, X_val_scaled, X_test_scaled, y_train.values, y_val.values, y_test.values
    


if __name__ == "__main__":

    ingestion = DataIngestion()

    df, _, _ = ingestion.load_data()

    #preproces

    preprocessor = DataPreprocessor()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(df)

    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final X_val shape: {X_val.shape}")
    print(f"Final y_train shape: {y_train.shape}")











