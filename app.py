"""
Streamlit App for Wine Quality Classification
"""
import streamlit as st
import torch
import numpy as np
import pandas as pd
import pickle
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add src to path
sys.path.append('src')

from model import SimpleANN, WineClassification
from data_ingestion import DataIngestion
from feature_engineering import FeatureEngineer


# Page config
st.set_page_config(
    page_title="Wine Quality Classifier",
    page_icon="🍷",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #8B0000;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    """Load trained model and configuration"""
    try:
        # Load config
        with open('models/pipeline_config.json', 'r') as f:
            config = json.load(f)

        # Load scaler
        with open('data/processed/scalar.pkl', 'rb') as f:
            scaler = pickle.load(f)

        # Initialize and load model
        checkpoint = torch.load('models/wine_classifier.pth', map_location=torch.device('cpu'), weights_only=True)
        model_type = checkpoint.get('model_type', 'SimpleANN')

        if model_type == 'WineClassification':
            best_params = config.get('best_params', {})
            if isinstance(best_params, dict):
                hidden_sizes = best_params.get('hidden_sizes', [32, 16])
                dropout_rate = best_params.get('dropout_rate', 0.2)
            else:
                hidden_sizes = [32, 16]
                dropout_rate = 0.2
            model = WineClassification(
                input_size=config['input_features'],
                hidden_sizes=hidden_sizes,
                num_classes=config['num_classes'],
                dropout_rate=dropout_rate
            )
        else:
            model = SimpleANN(
                input_size=config['input_features'],
                num_classes=config['num_classes']
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model, scaler, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def predict_wine_quality(model, scaler, features):
    """Make prediction"""
    # Scale features
    features_scaled = scaler.transform([features])

    # Apply the same feature engineering as the training pipeline
    engineer = FeatureEngineer()
    features_eng = engineer.create_interaction_features(features_scaled)

    # Convert to tensor
    features_tensor = torch.FloatTensor(features_eng)

    # Predict
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.softmax(output, dim=1).numpy()[0]
        predicted_class = np.argmax(probabilities)

    return predicted_class, probabilities


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🍷 Wine Quality Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Deep Learning Model for Wine Classification</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Predict", "📊 Model Info", "📈 Training History"])
    
    # Load model
    model, scaler, config = load_model_and_config()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first by running: `python src/pipeline.py`")
        return
    
    # Wine class names
    wine_classes = ['Class 0', 'Class 1', 'Class 2']
    if config and 'target_names' in config:
        wine_classes = config['target_names']
    
    # Pages
    if page == "🏠 Home":
        show_home_page(config)
    
    elif page == "🔮 Predict":
        show_prediction_page(model, scaler, wine_classes)
    
    elif page == "📊 Model Info":
        show_model_info_page(model, config)
    
    elif page == "📈 Training History":
        show_training_history_page()


def show_home_page(config):
    """Display home page"""
    st.header("Welcome to Wine Quality Classifier")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "ANN (PyTorch)")
    
    with col2:
        if config:
            st.metric("Test Accuracy", f"{config['test_accuracy']*100:.2f}%")
        else:
            st.metric("Test Accuracy", "N/A")
    
    with col3:
        if config:
            st.metric("Number of Classes", config['num_classes'])
        else:
            st.metric("Number of Classes", "3")
    
    st.markdown("---")
    
    st.subheader("About This Project")
    st.write("""
    This is a complete end-to-end machine learning project for wine quality classification using:
    
    - **Data Ingestion**: Loading the Wine dataset from sklearn
    - **Preprocessing**: Data cleaning, splitting, and scaling
    - **Feature Engineering**: Creating additional features and dimensionality reduction
    - **Hyperparameter Tuning**: Grid search for optimal parameters
    - **Model Training**: PyTorch ANN with customizable architecture
    - **Deployment**: Streamlit web interface
    
    Navigate to the **Predict** page to classify wine samples!
    """)
    
    st.markdown("---")
    
    st.subheader("Dataset Information")
    
    # Load dataset info
    ingestion = DataIngestion()
    df, features, target_names = ingestion.load_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Shape:**", df.shape)
        st.write("**Number of Features:**", len(features))
        st.write("**Number of Classes:**", len(target_names))
    
    with col2:
        st.write("**Features:**")
        for i, feat in enumerate(features[:5], 1):
            st.write(f"{i}. {feat}")
        if len(features) > 5:
            st.write(f"... and {len(features)-5} more")


def show_prediction_page(model, scaler, wine_classes):
    """Display prediction page"""
    st.header("🔮 Make Predictions")
    
    st.write("Enter the wine characteristics below to predict its quality class:")
    
    # Feature names from Wine dataset
    feature_names = [
        'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
        'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 
        'proanthocyanins', 'color_intensity', 'hue', 
        'od280/od315_of_diluted_wines', 'proline'
    ]
    
    # Create input fields in columns
    col1, col2, col3 = st.columns(3)
    
    features = []
    
    for i, name in enumerate(feature_names):
        col = [col1, col2, col3][i % 3]
        
        with col:
            # Default values (approximate means)
            default_values = [13.0, 2.3, 2.4, 19.5, 99.7, 2.3, 2.0, 0.36, 1.6, 5.1, 1.0, 2.6, 746.0]
            
            value = st.number_input(
                name.replace('_', ' ').title(),
                value=float(default_values[i]),
                step=0.1,
                format="%.2f"
            )
            features.append(value)
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Wine Class", type="primary"):
        with st.spinner("Making prediction..."):
            predicted_class, probabilities = predict_wine_quality(model, scaler, features)
        
        # Display results
        st.success("Prediction Complete!")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Predicted Class", wine_classes[predicted_class])
            st.metric("Confidence", f"{probabilities[predicted_class]*100:.2f}%")
        
        with col2:
            st.subheader("Class Probabilities")
            
            # Create probability chart
            prob_df = pd.DataFrame({
                'Class': wine_classes,
                'Probability': probabilities * 100
            })
            
            st.bar_chart(prob_df.set_index('Class'))
        
        # Show probability details
        st.markdown("---")
        st.subheader("Detailed Probabilities")
        
        for i, (wine_class, prob) in enumerate(zip(wine_classes, probabilities)):
            st.progress(float(prob), text=f"{wine_class}: {prob*100:.2f}%")


def show_model_info_page(model, config):
    """Display model information page"""
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Architecture")
        st.write(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        st.metric("Total Parameters", f"{total_params:,}")
        st.metric("Trainable Parameters", f"{trainable_params:,}")
    
    with col2:
        st.subheader("Configuration")
        
        if config:
            st.json(config)
        else:
            st.write("No configuration found")
    
    st.markdown("---")
    
    st.subheader("Model Features")
    
    features_info = [
        "✅ Multi-layer Perceptron (MLP) architecture",
        "✅ ReLU activation functions",
        "✅ Dropout for regularization",
        "✅ Batch Normalization",
        "✅ Cross-Entropy loss",
        "✅ Adam optimizer"
    ]
    
    for feature in features_info:
        st.write(feature)


def show_training_history_page():
    """Display training history page"""
    st.header("📈 Training History")
    
    # Load training history
    try:
        checkpoint = torch.load('models/wine_classifier.pth', map_location=torch.device('cpu'), weights_only=True)
        history = checkpoint['history']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loss Over Epochs")
            
            loss_df = pd.DataFrame({
                'Epoch': range(1, len(history['train_loss']) + 1),
                'Train Loss': history['train_loss'],
                'Validation Loss': history['val_loss']
            })
            
            st.line_chart(loss_df.set_index('Epoch'))
        
        with col2:
            st.subheader("Accuracy Over Epochs")
            
            acc_df = pd.DataFrame({
                'Epoch': range(1, len(history['train_acc']) + 1),
                'Train Accuracy': history['train_acc'],
                'Validation Accuracy': history['val_acc']
            })
            
            st.line_chart(acc_df.set_index('Epoch'))
        
        st.markdown("---")
        
        # Final metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
        
        with col2:
            st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
        
        with col3:
            st.metric("Final Train Acc", f"{history['train_acc'][-1]:.2f}%")
        
        with col4:
            st.metric("Final Val Acc", f"{history['val_acc'][-1]:.2f}%")
        
        # Show training plot if exists
        if os.path.exists('plots/training_history.png'):
            st.markdown("---")
            st.subheader("Training Visualization")
            st.image('plots/training_history.png')
        
    except Exception as e:
        st.error(f"Could not load training history: {e}")


if __name__ == "__main__":
    main()
