import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import WineClassification
import numpy as np
import json
import os
from sklearn.model_selection import KFold
from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer



class HyperparameterTuner:

    def __init__(self, X_train, y_train, models_path='models'):
        self.X_train = X_train
        self.y_train = y_train
        self.models_path = models_path
        self.best_params = None
        self.best_score = 0
        os.makedirs(models_path, exist_ok=True)

    def create_data_loader(self, X, y, batch_size):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def train_and_evaluate_fold(self, X_train_fold, y_train_fold, X_val_fold, y_val_fold, config, epochs=50):
        
        train_loader = self.create_data_loader(X_train_fold, y_train_fold, config['batch_size'])
        val_loader = self.create_data_loader(X_val_fold, y_val_fold, config['batch_size'])

        # Initialize fresh model for each fold
        model = WineClassification(
            input_size=X_train_fold.shape[1],
            hidden_sizes=config['hidden_sizes'],
            num_classes=len(np.unique(self.y_train)),
            dropout_rate=config['dropout_rate']
        )

        criterion = nn.CrossEntropyLoss()

        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

        # Training loop
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate on validation fold
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        return 100 * correct / total

    def cross_validate(self, config, n_folds=3, epochs=50):
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(self.X_train)):
            X_train_fold = self.X_train[train_idx]
            y_train_fold = self.y_train[train_idx]
            X_val_fold = self.X_train[val_idx]
            y_val_fold = self.y_train[val_idx]

            score = self.train_and_evaluate_fold(
                X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                config, epochs=epochs
            )
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        return mean_score, std_score

    def grid_search(self, param_grid, epochs=50):
        
        print("\n=== Starting Hyperparameter Tuning (with Cross-Validation) ===")

        results = []

        # Generate all combinations
        learning_rates = param_grid.get('learning_rate', [0.001])
        batch_sizes = param_grid.get('batch_size', [32])
        hidden_sizes_list = param_grid.get('hidden_sizes', [[64, 32]])
        dropout_rates = param_grid.get('dropout_rate', [0.3])
        optimizers = param_grid.get('optimizer', ['adam'])

        total_configs = (len(learning_rates) * len(batch_sizes) *
                        len(hidden_sizes_list) * len(dropout_rates) * len(optimizers))

        print(f"Testing {total_configs} configurations with 3-fold CV...")

        config_num = 0
        for lr in learning_rates:
            for bs in batch_sizes:
                for hs in hidden_sizes_list:
                    for dr in dropout_rates:
                        for opt in optimizers:
                            config_num += 1

                            config = {
                                'learning_rate': lr,
                                'batch_size': bs,
                                'hidden_sizes': hs,
                                'dropout_rate': dr,
                                'optimizer': opt
                            }

                            print(f"\nConfig {config_num}/{total_configs}: {config}")

                            mean_acc, std_acc = self.cross_validate(
                                config, n_folds=3, epochs=epochs
                            )

                            results.append({
                                'config': config,
                                'mean_accuracy': mean_acc,
                                'std_accuracy': std_acc
                            })

                            print(f"CV Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")

                            # Update best based on cross-validated score
                            if mean_acc > self.best_score:
                                self.best_score = mean_acc
                                self.best_params = config
                                print(f"*** New best CV accuracy: {mean_acc:.2f}%")

        # Save results
        self.save_tuning_results(results)

        print("\n=== Hyperparameter Tuning Complete ===")
        print(f"Best configuration: {self.best_params}")
        print(f"Best CV accuracy: {self.best_score:.2f}%")

        return self.best_params, self.best_score

    def save_tuning_results(self, results):
        """Save tuning results to JSON"""
        results_path = os.path.join(self.models_path, 'tuning_results.json')

        serializable_results = []
        for r in results:
            serializable_results.append({
                'config': r['config'],
                'mean_accuracy': float(r['mean_accuracy']),
                'std_accuracy': float(r['std_accuracy'])
            })

        with open(results_path, 'w') as f:
            json.dump({
                'results': serializable_results,
                'best_params': self.best_params,
                'best_score': float(self.best_score)
            }, f, indent=2)

        print(f"Tuning results saved to {results_path}")


if __name__ == "__main__":

    # Load and preprocess
    ingestion = DataIngestion()
    df, features, _ = ingestion.load_data()

    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(df)

    engineer = FeatureEngineer()
    X_train_eng, X_val_eng = engineer.engineer_features(
        X_train, X_val, features, use_pca=False
    )

    # Tune hyperparameters using cross-validation on training data only
    tuner = HyperparameterTuner(X_train_eng, y_train)

    param_grid = {
        'learning_rate': [0.001, 0.01],
        'batch_size': [16, 32],
        'hidden_sizes': [[32, 16], [64, 32]],
        'dropout_rate': [0.2, 0.3],
        'optimizer': ['adam']
    }

    best_params, best_score = tuner.grid_search(param_grid, epochs=30)
