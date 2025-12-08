"""
Model Predictions and Visualization
====================================

This script loads the trained CNN and MLP models and makes predictions
on test samples, displaying the results with visualizations.

Usage:
------
    python predictions.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_preprocessed_data_with_augmentation

sns.set_style("whitegrid")

# Constants
IMG_HEIGHT = 112
IMG_WIDTH = 92
IMG_CHANNELS = 1
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'predictions')
MLP_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'mlp')


def load_all_data():
    """Load data and prepare for both CNN and MLP models."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "umist_cropped.mat"
    )
    
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = (
        load_preprocessed_data_with_augmentation(dataset_path=path)
    )
    
    num_classes = len(np.unique(y_train))
    print(f"\nTest samples: {X_test.shape[0]}")
    print(f"Features: {X_test.shape[1]}, Classes: {num_classes}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


def prepare_cnn_data(X_test):
    """Reshape data for CNN input."""
    return X_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)


def prepare_mlp_data(X_test, num_classes):
    """
    Prepare PCA-reduced data and clustering features for MLP models.
    
    Loads the SAME PCA, K-means, and GMM models that were used during training
    to ensure features match what the MLP models expect.
    """
    print("\n" + "-" * 50)
    print("Preparing MLP features (loading saved transformers)...")
    print("-" * 50)
    
    # Load saved PCA model
    pca_path = os.path.join(MLP_OUTPUT_DIR, 'pca_model.joblib')
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA model not found at {pca_path}. Run mlp.py first!")
    pca = joblib.load(pca_path)
    print(f"✓ Loaded PCA model from: {pca_path}")
    
    # Transform test data with PCA
    X_test_pca = pca.transform(X_test)
    print(f"  PCA features: {X_test_pca.shape[1]}")
    
    # Load saved K-means model
    kmeans_path = os.path.join(MLP_OUTPUT_DIR, 'kmeans_model.joblib')
    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(f"K-means model not found at {kmeans_path}. Run mlp.py first!")
    kmeans = joblib.load(kmeans_path)
    print(f"✓ Loaded K-means model from: {kmeans_path}")
    
    # Transform with K-means
    X_test_dist = kmeans.transform(X_test_pca)
    X_test_kmeans = np.hstack([X_test_pca, X_test_dist])
    print(f"  PCA + K-Means features: {X_test_kmeans.shape[1]}")
    
    # Load saved GMM model
    gmm_path = os.path.join(MLP_OUTPUT_DIR, 'gmm_model.joblib')
    if not os.path.exists(gmm_path):
        raise FileNotFoundError(f"GMM model not found at {gmm_path}. Run mlp.py first!")
    gmm = joblib.load(gmm_path)
    print(f"✓ Loaded GMM model from: {gmm_path}")
    
    # Transform with GMM
    X_test_probs = gmm.predict_proba(X_test_pca)
    X_test_gmm = np.hstack([X_test_pca, X_test_probs])
    print(f"  PCA + GMM features: {X_test_gmm.shape[1]}")
    
    return X_test_pca, X_test_kmeans, X_test_gmm


def load_models():
    """Load all trained models."""
    print("\n" + "=" * 70)
    print("LOADING MODELS")
    print("=" * 70)
    
    models = {}
    base_dir = os.path.dirname(__file__)
    
    # CNN model
    cnn_path = os.path.join(base_dir, 'outputs', 'cnn', 'best_cnn.keras')
    if os.path.exists(cnn_path):
        models['CNN'] = keras.models.load_model(cnn_path)
        print(f"✓ CNN loaded from: {cnn_path}")
    else:
        print(f"✗ CNN not found at: {cnn_path}")
    
    # MLP models
    mlp_models = {
        'MLP (PCA)': 'mlp_pca.keras',
        'MLP (PCA+KMeans)': 'mlp_pca_kmeans.keras',
        'MLP (PCA+GMM)': 'mlp_pca_gmm.keras'
    }
    
    for name, filename in mlp_models.items():
        path = os.path.join(base_dir, 'outputs', 'mlp', filename)
        if os.path.exists(path):
            models[name] = keras.models.load_model(path)
            print(f"✓ {name} loaded from: {path}")
        else:
            print(f"✗ {name} not found at: {path}")
    
    return models


def predict_samples(models, X_cnn, X_pca, X_kmeans, X_gmm, y_true, n_samples=10):
    """Make predictions on random test samples."""
    print("\n" + "=" * 70)
    print(f"PREDICTIONS ON {n_samples} RANDOM TEST SAMPLES")
    print("=" * 70)
    
    # Select random indices
    np.random.seed(42)
    indices = np.random.choice(len(y_true), n_samples, replace=False)
    
    predictions = {}
    
    for name, model in models.items():
        if name == 'CNN':
            X = X_cnn[indices]
        elif name == 'MLP (PCA)':
            X = X_pca[indices]
        elif name == 'MLP (PCA+KMeans)':
            X = X_kmeans[indices]
        elif name == 'MLP (PCA+GMM)':
            X = X_gmm[indices]
        else:
            continue
        
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        predictions[name] = y_pred
    
    # Print predictions table
    print(f"\n{'Sample':<8} {'True':<8}", end='')
    for name in predictions.keys():
        print(f"{name:<18}", end='')
    print()
    print("-" * (16 + 18 * len(predictions)))
    
    for i, idx in enumerate(indices):
        true_label = y_true[idx]
        print(f"{i+1:<8} {true_label:<8}", end='')
        for name, preds in predictions.items():
            pred = preds[i]
            match = "✓" if pred == true_label else "✗"
            print(f"{pred} {match:<16}", end='')
        print()
    
    # Accuracy summary
    print("\n" + "-" * 50)
    print("Accuracy on these samples:")
    for name, preds in predictions.items():
        acc = np.mean(preds == y_true[indices])
        print(f"  {name}: {acc:.1%}")
    
    return indices, predictions


def plot_prediction_samples(X_test_flat, y_true, indices, predictions):
    """Visualize predicted samples with model predictions."""
    n_samples = len(indices)
    n_models = len(predictions)
    
    fig, axes = plt.subplots(2, n_samples // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Reshape and display image
        img = X_test_flat[idx].reshape(IMG_HEIGHT, IMG_WIDTH)
        ax.imshow(img, cmap='gray')
        
        true_label = y_true[idx]
        
        # Build title with predictions
        title_lines = [f"True: {true_label}"]
        for name, preds in predictions.items():
            pred = preds[i]
            short_name = name.replace('MLP ', '').replace('(', '').replace(')', '')
            title_lines.append(f"{short_name}: {pred}")
        
        ax.set_title('\n'.join(title_lines), fontsize=8)
        ax.axis('off')
    
    plt.suptitle('Model Predictions on Test Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_samples.png'), dpi=150)
    plt.show()


def plot_confusion_matrices(models, X_cnn, X_pca, X_kmeans, X_gmm, y_true):
    """Plot confusion matrices for all models."""
    print("\n" + "=" * 70)
    print("GENERATING CONFUSION MATRICES")
    print("=" * 70)
    
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, model) in zip(axes, models.items()):
        if name == 'CNN':
            X = X_cnn
        elif name == 'MLP (PCA)':
            X = X_pca
        elif name == 'MLP (PCA+KMeans)':
            X = X_kmeans
        elif name == 'MLP (PCA+GMM)':
            X = X_gmm
        else:
            continue
        
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        
        sns.heatmap(cm_norm, ax=ax, cmap='Blues', vmin=0, vmax=1,
                    square=True, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{name}\nAcc: {np.mean(y_pred == y_true):.1%}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.suptitle('Confusion Matrices (Normalized)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrices.png'), dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(models, X_cnn, X_pca, X_kmeans, X_gmm, y_true):
    """Plot accuracy comparison bar chart."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    accuracies = {}
    
    for name, model in models.items():
        if name == 'CNN':
            X = X_cnn
        elif name == 'MLP (PCA)':
            X = X_pca
        elif name == 'MLP (PCA+KMeans)':
            X = X_kmeans
        elif name == 'MLP (PCA+GMM)':
            X = X_gmm
        else:
            continue
        
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        accuracies[name] = np.mean(y_pred == y_true)
    
    # Print results
    print(f"\n{'Model':<25} {'Test Accuracy':<15}")
    print("-" * 40)
    for name, acc in accuracies.items():
        print(f"{name:<25} {acc:.4f} ({acc:.1%})")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = ax.bar(accuracies.keys(), accuracies.values(), color=colors[:len(accuracies)], 
                  alpha=0.8, edgecolor='black')
    
    for bar, acc in zip(bars, accuracies.values()):
        ax.annotate(f'{acc:.1%}', 
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Model Comparison - Test Set Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=150)
    plt.show()
    
    # Best model
    best_model = max(accuracies, key=accuracies.get)
    print(f"\nBest Model: {best_model} ({accuracies[best_model]:.1%})")
    
    return accuracies


def main():
    """Main execution function."""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = load_all_data()
    
    # Prepare data for different models
    X_cnn = prepare_cnn_data(X_test)
    X_pca, X_kmeans, X_gmm = prepare_mlp_data(X_test, num_classes)
    
    # Load models
    models = load_models()
    
    if not models:
        print("\n❌ No models found! Please train models first using cnn.py and mlp.py")
        return
    
    # Make predictions on random samples
    indices, predictions = predict_samples(
        models, X_cnn, X_pca, X_kmeans, X_gmm, y_test, n_samples=10
    )
    
    # Visualize predictions
    plot_prediction_samples(X_test, y_test, indices, predictions)
    
    # Model comparison
    accuracies = plot_model_comparison(models, X_cnn, X_pca, X_kmeans, X_gmm, y_test)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
