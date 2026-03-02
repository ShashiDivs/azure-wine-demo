import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import WineClassification, SimpleANN
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend -- safe for servers without a display
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class ModelTrainer:
    """Train and evaluate the ANN model"""

    def __init__(self, models_path='models', plots_path='plots'):
        self.models_path = models_path
        self.plots_path = plots_path
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(plots_path, exist_ok=True)

    def prepare_data(self, X_train, y_train, X_val, y_val, batch_size=32):
      
        # Convert numpy arrays to PyTorch tensors
        # FloatTensor for features (32-bit float), LongTensor for labels (integer)
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)

        # TensorDataset pairs each input with its label
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # shuffle=True for training (randomize batch composition each epoch)
        # shuffle=False for validation (consistency for fair evaluation)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def initialize_model(self, input_size, num_classes, config=None):
        """Initialize the model with given configuration"""
        if config is None:
            self.model = SimpleANN(input_size, num_classes)
        else:
            self.model = WineClassification(
                input_size=input_size,
                hidden_sizes=config.get('hidden_sizes', [64, 32]),
                num_classes=num_classes,
                dropout_rate=config.get('dropout_rate', 0.3)
            )

        print("Model initialized:")
        print(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")

        return self.model

    def train(self, X_train, y_train, X_val, y_val, config=None, epochs=100):

        print("\n=== Starting Model Training ===")

        # Default config
        if config is None:
            config = {
                'learning_rate': 0.001,
                'batch_size': 32,
                'hidden_sizes': [32, 16],
                'dropout_rate': 0.2,
                'optimizer': 'adam'
            }

        # Prepare data
        train_loader, val_loader = self.prepare_data(
            X_train, y_train, X_val, y_val, config['batch_size']
        )

        # Initialize model
        num_classes = len(np.unique(y_train))
        self.initialize_model(X_train.shape[1], num_classes, config)

        criterion = nn.CrossEntropyLoss()


        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        else:
            optimizer = optim.SGD(self.model.parameters(), lr=config['learning_rate'], momentum=0.9)

        # Training loop
        for epoch in range(epochs):
            # --- Training phase ---
            self.model.train()  # Enable dropout and BatchNorm training mode
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()       # Clear gradients from previous batch
                outputs = self.model(batch_X)  # Forward pass
                loss = criterion(outputs, batch_y)  # Compute loss
                loss.backward()             # Backpropagation: compute gradients
                optimizer.step()            # Update weights using gradients

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            train_accuracy = 100 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)

            # --- Validation phase ---
            self.model.eval()  # Disable dropout, use running BatchNorm stats
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():  # No gradients needed for evaluation
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_accuracy = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)

            # Save history for plotting later
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_accuracy)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(val_accuracy)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
                print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        print("\n=== Training Complete ===")
        print(f"Final Train Accuracy: {train_accuracy:.2f}%")
        print(f"Final Validation Accuracy: {val_accuracy:.2f}%")

        return self.model, self.history

    def evaluate(self, X_test, y_test, target_names=None):
        
        print("\n=== Evaluating Model ===")

        self.model.eval()
        X_test_tensor = torch.FloatTensor(X_test)

        with torch.no_grad():
            outputs = self.model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)

        y_pred = predicted.numpy()

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy*100:.2f}%")

        # Classification report: precision, recall, f1-score per class
        # WHY CARE ABOUT MORE THAN ACCURACY?
        #   Accuracy can be misleading with imbalanced classes. If 90% of data
        #   is class A, predicting "A" always gives 90% accuracy but 0% recall
        #   on class B. Precision/recall/F1 tell the full story.
        print("\nClassification Report:")
        report_labels = list(target_names) if target_names is not None else None
        print(classification_report(y_test, y_pred, target_names=report_labels))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Plot confusion matrix as a heatmap
        self._plot_confusion_matrix(cm, target_names)

        return accuracy, cm

    def _plot_confusion_matrix(self, cm, target_names=None):
        
        labels = list(target_names) if target_names is not None else [f"Class {i}" for i in range(len(cm))]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax
        )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()

        plot_path = os.path.join(self.plots_path, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"Confusion matrix saved to {plot_path}")
        plt.close()

    def plot_history(self):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_path, 'training_history.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"\nTraining history plot saved to {plot_path}")
        plt.close()

    def save_model(self, filename='wine_classifier.pth'):
        """Save the trained model"""
        model_path = os.path.join(self.models_path, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model.__class__.__name__,
            'history': self.history
        }, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, filename='wine_classifier.pth', input_size=13, num_classes=3, config=None):
        """Load a trained model"""
        model_path = os.path.join(self.models_path, filename)

        # Load checkpoint
        checkpoint = torch.load(model_path, weights_only=True)

        # Determine model type from checkpoint or config
        model_type = checkpoint.get('model_type', 'SimpleANN')

        if model_type == 'WineClassification' or config is not None:
            self.model = WineClassification(
                input_size=input_size,
                hidden_sizes=config.get('hidden_sizes', [64, 32]) if config else [64, 32],
                num_classes=num_classes,
                dropout_rate=config.get('dropout_rate', 0.3) if config else 0.3
            )
        else:
            self.model = SimpleANN(input_size, num_classes)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint['history']

        print(f"Model loaded from {model_path}")
        return self.model


if __name__ == "__main__":
    from data_ingestion import DataIngestion
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer

    # Load and preprocess
    ingestion = DataIngestion()
    df, features, target_names = ingestion.load_data()

    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(df)

    engineer = FeatureEngineer()
    X_train_eng, X_val_eng = engineer.engineer_features(
        X_train, X_val, features, use_pca=False
    )

    # Train model (using train set, validate on val set)
    trainer = ModelTrainer()
    model, history = trainer.train(X_train_eng, y_train, X_val_eng, y_val, epochs=50)

    # Final evaluation on held-out test set
    X_test_eng = engineer.create_interaction_features(X_test)
    trainer.evaluate(X_test_eng, y_test, target_names=target_names)

    # Plot and save
    trainer.plot_history()
    trainer.save_model()
