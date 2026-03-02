import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
import pickle
import os


class FeatureEngineer:

    def __init__(self, processed_path='data/processed'):

        self.processed_path = processed_path
        self.pca = None
        os.makedirs(processed_path, exist_ok=True)

    
    def create_polynomial_features(self, X, feature_names, degree=2):

        X_poly = X.copy()

        for i in range(min(3, X.shape[1])):
            squared = X[:, i] ** 2
            X_poly = np.column_stack([X_poly, squared])

        return X_poly

    def apply_pca(self, X_train, X_test, n_components=10):

        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train)
        X_test_pca = self.pca.transform(X_test)

        explained_var = sum(self.pca.explained_variance_ratio_)
        print(f"Explained variance: {explained_var:.4f}")

        return X_train_pca, X_test_pca
    
    def create_interaction_features(self, X):

        if X.shape[1] >=2:
            interaction_1 = X[:,0] * X[:, 1]
            X = np.column_stack([X, interaction_1])

        
        if X.shape[1] >=3:
            interaction_2 = X[:,0] * X[:, 2]
            X = np.column_stack([X, interaction_2])

        return X
    
    def save_pca(self):

        if self.pca:
            pca_path = os.path.join(self.processed_path, 'pca.pkl')
            with open(pca_path, "wb") as f:
                pickle.dump(self.pca, f)
            print(f"PCA saved to {pca_path}")

    def engineer_features(self, X_train, X_test, feature_names, use_pca=False):

        if use_pca:
            X_train_eng, X_test_eng = self.apply_pca(X_train, X_test, n_components=10)
            self.save_pca()
        else:
            X_train_eng = self.create_interaction_features(X_train)
            X_test_eng = self.create_interaction_features(X_test)

        
        print(f" Shape after Feature Engineering train {X_train_eng.shape}, test  {X_test_eng.shape}")

        return X_train_eng, X_test_eng
    

if __name__ == "__main__":

    ingestion = DataIngestion()
    df, features, _ = ingestion.load_data()

    preprocessor = DataPreprocessor()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(df)

    engineer = FeatureEngineer()

    X_train_eng, X_test_eng = engineer.engineer_features(
        X_train, X_test, features, use_pca=False
    )

    print(f"Final feature Engineering X_train shape {X_train_eng.shape}")
