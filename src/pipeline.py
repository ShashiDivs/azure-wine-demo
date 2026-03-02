import os
import sys
from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from hyperparameter_tuning import HyperparameterTuner
from training import ModelTrainer
import json
import argparse

sys.path.append(os.path.dirname(__file__))

def run_pipeline(tune_hyperparameters=False, use_pca=False):

    # step 1: Data ingestion
    ingestion = DataIngestion()
    df, features, target_names = ingestion.load_data()
    ingestion.save_raw_data(df)

    # step 2: Preprocessing
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(df)

    # step 3: Feature Engineering
    engineer = FeatureEngineer()
    X_train_eng, X_val_eng = engineer.engineer_features(
        X_train, X_val, features, use_pca=use_pca
    )
    if use_pca:
        X_test_eng = engineer.apply_pca.transform(X_test)
    else:
        X_test_eng = engineer.create_interaction_features(X_test)

    # step 4: Hyperparamter Tuning
    best_params = None

    if tune_hyperparameters:
        tuner = HyperparameterTuner(X_train_eng, y_train)

        param_grid = {
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32],
            'hidden_sizes': [[32, 16], [64, 32]],
            'dropout_rate': [0.2, 0.3],
            'optimizer': ['adam']
        }

        best_params, best_score = tuner.grid_search(param_grid, epochs=30)
    else:
        print("Usinf default parameters")

    # step 5: Model Training

    trainer = ModelTrainer()
    model, history = trainer.train(
        X_train_eng, y_train, X_val_eng, y_val, config=best_params, epochs=50
    )

    # step 6: Evaluation
    test_accuracy, confusion_matrix = trainer.evaluate(X_test_eng, y_test, target_names=target_names)

    # step 7: Model Training

    trainer.plot_history()
    trainer.save_model()

    pipeline_config = {
        'use_pca':use_pca,
        'tuned':tune_hyperparameters,
        'best_params': best_params if best_params else 'default',
        'test_accuracy': float(test_accuracy),
        'input_features':int(X_train_eng.shape[1]),
        'num_classes': int(len(target_names)),
        'target_names':list(target_names)
    }

    config_path = 'models/pipeline_config.json'
    with open(config_path, 'w') as f:
        json.dump(pipeline_config, f, indent=2)
    print(f"Pipeline configuration saved to {config_path}")

    return trainer, test_accuracy

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wine Quality classification Pipeline")
    parser.add_argument('--tune', action='store_true',help="Enable Hyperparameter tuning")
    parser.add_argument('--pca', action='store_true',help="Use PCA for feature engineering")

    args = parser.parse_args()
    trainer, accuracy = run_pipeline(
        tune_hyperparameters=args.tune,
        use_pca=args.pca
    )


    
