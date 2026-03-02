"""
Quick Test Script
Tests all modules to ensure everything works
"""
import sys
import os

# Add src to path
sys.path.append('src')

print("="*60)
print("TESTING WINE QUALITY ANN PROJECT")
print("="*60)

# Test 1: Data Ingestion
print("\n[1/6] Testing Data Ingestion...")
try:
    from src.data_ingestion import DataIngestion
    ingestion = DataIngestion()
    df, features, targets = ingestion.load_data()
    print(f"[PASS] Data loaded: {df.shape}")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    sys.exit(1)

# Test 2: Preprocessing (now returns train/val/test)
print("\n[2/6] Testing Preprocessing...")
try:
    from src.preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(df)
    print(f"[PASS] Data preprocessed: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    sys.exit(1)

# Test 3: Feature Engineering
print("\n[3/6] Testing Feature Engineering...")
try:
    from src.feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    X_train_eng, X_val_eng = engineer.engineer_features(
        X_train, X_val, features, use_pca=False
    )
    print(f"[PASS] Features engineered: Train {X_train_eng.shape}, Val {X_val_eng.shape}")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    sys.exit(1)

# Test 4: Model
print("\n[4/6] Testing Model Architecture...")
try:
    import torch
    from src.model import SimpleANN, WineClassifierANN

    model1 = SimpleANN(input_size=X_train_eng.shape[1], num_classes=3)
    model2 = WineClassifierANN(input_size=X_train_eng.shape[1], num_classes=3)

    # Test forward pass
    test_input = torch.randn(10, X_train_eng.shape[1])
    output1 = model1(test_input)
    output2 = model2(test_input)

    print(f"[PASS] Models initialized successfully")
    print(f"  SimpleANN output shape: {output1.shape}")
    print(f"  WineClassifierANN output shape: {output2.shape}")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    sys.exit(1)

# Test 5: Training (quick test with 5 epochs, train on train, validate on val)
print("\n[5/6] Testing Training Pipeline (5 epochs)...")
try:
    from src.training import ModelTrainer

    trainer = ModelTrainer()
    model, history = trainer.train(
        X_train_eng, y_train, X_val_eng, y_val,
        config=None,
        epochs=5
    )

    print(f"[PASS] Training successful")
    print(f"  Final train accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"  Final val accuracy: {history['val_acc'][-1]:.2f}%")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    sys.exit(1)

# Test 6: Model Save/Load
print("\n[6/6] Testing Model Save/Load...")
try:
    trainer.save_model('test_model.pth')

    new_trainer = ModelTrainer()
    loaded_model = new_trainer.load_model(
        'test_model.pth',
        input_size=X_train_eng.shape[1],
        num_classes=3,
        config={'hidden_sizes': [32, 16], 'dropout_rate': 0.2}
    )

    # Clean up test model
    os.remove('models/test_model.pth')

    print(f"[PASS] Model save/load successful")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
print("\nYou can now:")
print("1. Train full model: python src/pipeline.py")
print("2. Run Streamlit app: streamlit run app.py")
print("3. Deploy to Azure: python azure_deploy.py")
print("="*60)
