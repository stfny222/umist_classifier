"""
MLP Classifier with PCA Dimensionality Reduction and Keras Tuner Hyperband
===========================================================================

This script implements a Multi-Layer Perceptron (MLP) classifier for the UMIST
facial recognition dataset with the following pipeline:

1. Load augmented UMIST data
2. Apply PCA for dimensionality reduction
3. Optionally apply K-means clustering and use distances to cluster centers as features
4. Fine-tune MLP hyperparameters using Keras Tuner Hyperband
5. Train and evaluate the best model with and without K-means features

Usage:
------
    python mlp.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import keras_tuner as kt

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_preprocessed_data_with_augmentation
from dimensionality_reduction.pca import determine_pca_components, fit_and_transform_pca

sns.set_style("whitegrid")

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', 'mlp')


def load_data():
    """Load and prepare data for MLP."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "umist_cropped.mat"
    )
    
    print("=" * 70)
    print("LOADING DATA (AUGMENTED)")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = (
        load_preprocessed_data_with_augmentation(dataset_path=path)
    )
    
    num_classes = len(np.unique(y_train))
    print(f"\nTrain: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}, Classes: {num_classes}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes


def apply_pca_reduction(X_train, X_val, X_test, variance_threshold=0.95):
    """Apply PCA dimensionality reduction."""
    print("\n" + "=" * 70)
    print("PCA DIMENSIONALITY REDUCTION")
    print("=" * 70)
    
    # Determine optimal number of components
    n_components, _, _, _ = determine_pca_components(
        X_train, variance_threshold=variance_threshold, plot=True
    )
    
    # Fit and transform
    X_train_pca, X_val_pca, X_test_pca, pca = fit_and_transform_pca(
        X_train, X_val, X_test, n_components
    )
    
    print(f"\nReduced dimensions: {X_train.shape[1]} -> {X_train_pca.shape[1]}")
    
    # Save PCA model
    pca_path = os.path.join(OUTPUT_DIR, 'pca_model.joblib')
    joblib.dump(pca, pca_path)
    print(f"PCA model saved to: {pca_path}")
    
    return X_train_pca, X_val_pca, X_test_pca, pca, n_components


def apply_kmeans_features(X_train, X_val, X_test, n_clusters=20, n_init=10):
    """
    Apply K-means clustering and compute distance features to cluster centers.
    
    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Feature matrices (PCA-reduced)
    n_clusters : int
        Number of clusters (typically = number of classes)
    n_init : int
        Number of K-means initializations
        
    Returns
    -------
    X_train_dist, X_val_dist, X_test_dist : np.ndarray
        Distance features (n_samples, n_clusters)
    kmeans : KMeans
        Fitted KMeans model
    """
    print("\n" + "=" * 70)
    print(f"K-MEANS CLUSTERING (k={n_clusters})")
    print("=" * 70)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=42,
        verbose=0
    )
    
    # Fit on training data
    kmeans.fit(X_train)
    print(f"K-means fitted with {n_clusters} clusters")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    # Compute distances to cluster centers for all sets
    X_train_dist = kmeans.transform(X_train)  # (n_samples, n_clusters)
    X_val_dist = kmeans.transform(X_val)
    X_test_dist = kmeans.transform(X_test)
    
    print(f"Distance features shape: {X_train_dist.shape}")
    
    # Save K-means model
    kmeans_path = os.path.join(OUTPUT_DIR, 'kmeans_model.joblib')
    joblib.dump(kmeans, kmeans_path)
    print(f"K-means model saved to: {kmeans_path}")
    
    return X_train_dist, X_val_dist, X_test_dist, kmeans


def apply_gmm_features(X_train, X_val, X_test, n_components=20, n_init=5):
    """
    Apply Gaussian Mixture Model and compute soft probability features.
    
    Unlike K-means which gives hard cluster assignments, GMM provides
    soft probabilities of belonging to each cluster (component).
    
    Parameters
    ----------
    X_train, X_val, X_test : np.ndarray
        Feature matrices (PCA-reduced)
    n_components : int
        Number of Gaussian components (typically = number of classes)
    n_init : int
        Number of GMM initializations
        
    Returns
    -------
    X_train_probs, X_val_probs, X_test_probs : np.ndarray
        Soft probability features (n_samples, n_components)
    gmm : GaussianMixture
        Fitted GMM model
    """
    print("\n" + "=" * 70)
    print(f"GAUSSIAN MIXTURE MODEL (n_components={n_components})")
    print("=" * 70)
    
    gmm = GaussianMixture(
        n_components=n_components,
        n_init=n_init,
        covariance_type='full',
        random_state=42,
        verbose=0
    )
    
    # Fit on training data
    gmm.fit(X_train)
    print(f"GMM fitted with {n_components} components")
    print(f"Converged: {gmm.converged_}")
    print(f"Log-likelihood (train): {gmm.score(X_train):.4f}")
    
    # Compute soft probabilities for all sets
    X_train_probs = gmm.predict_proba(X_train)  # (n_samples, n_components)
    X_val_probs = gmm.predict_proba(X_val)
    X_test_probs = gmm.predict_proba(X_test)
    
    print(f"Soft probability features shape: {X_train_probs.shape}")
    
    # Save GMM model
    gmm_path = os.path.join(OUTPUT_DIR, 'gmm_model.joblib')
    joblib.dump(gmm, gmm_path)
    print(f"GMM model saved to: {gmm_path}")
    
    return X_train_probs, X_val_probs, X_test_probs, gmm


def combine_features(*feature_arrays):
    """Concatenate multiple feature arrays horizontally."""
    return np.hstack(feature_arrays)


def build_mlp_model(hp, input_dim, num_classes):
    """
    Build MLP with tunable hyperparameters.
    
    Hyperparameters tuned:
    - Number of hidden layers (1-3)
    - Units per layer (64, 128, 256)
    - Dropout rate (0.2-0.5)
    - L2 regularization (1e-4, 1e-3)
    - Learning rate (1e-3, 1e-4)
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    # HP: number of hidden layers
    n_layers = hp.Int('n_layers', min_value=1, max_value=3, default=2)
    
    # HP: units per layer
    units = hp.Choice('units', values=[64, 128, 256], default=128)
    
    # HP: dropout rate
    dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1, default=0.3)
    
    # HP: L2 regularization
    l2 = hp.Choice('l2_reg', values=[1e-4, 1e-3], default=1e-4)
    
    # Add hidden layers
    for i in range(n_layers):
        model.add(layers.Dense(
            units,
            activation='relu',
            kernel_regularizer=regularizers.l2(l2),
            name=f'dense_{i}'
        ))
        model.add(layers.BatchNormalization(name=f'bn_{i}'))
        model.add(layers.Dropout(dropout, name=f'dropout_{i}'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    # HP: learning rate
    lr = hp.Choice('learning_rate', values=[1e-3, 1e-4], default=1e-3)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def run_hyperband_tuning(X_train, X_val, y_train_oh, y_val_oh, num_classes, 
                         tuner_name='mlp_pca', max_epochs=50):
    """
    Run Hyperband hyperparameter search.
    
    Parameters
    ----------
    X_train, X_val : np.ndarray
        Training and validation features
    y_train_oh, y_val_oh : np.ndarray
        One-hot encoded labels
    num_classes : int
        Number of output classes
    tuner_name : str
        Name for the tuner project (to differentiate runs)
    max_epochs : int
        Maximum epochs for Hyperband
        
    Returns
    -------
    best_model : keras.Model
        Best model from tuning
    best_hps : kt.HyperParameters
        Best hyperparameters
    """
    print("\n" + "=" * 70)
    print(f"KERAS TUNER - HYPERBAND ({tuner_name})")
    print("=" * 70)
    
    input_dim = X_train.shape[1]
    
    tuner_dir = os.path.join(OUTPUT_DIR, 'tuner')
    os.makedirs(tuner_dir, exist_ok=True)
    
    tuner = kt.Hyperband(
        lambda hp: build_mlp_model(hp, input_dim, num_classes),
        objective='val_accuracy',
        max_epochs=max_epochs,
        factor=3,
        directory=tuner_dir,
        project_name=tuner_name,
        overwrite=True
    )
    
    print(f"\nInput dimension: {input_dim}")
    print(f"Running Hyperband search (max_epochs={max_epochs})...")
    
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    tuner.search(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )
    
   
    best_hps = tuner.get_best_hyperparameters()[0]
    
    # Get best validation accuracy from the oracle
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_val_acc = best_trial.metrics.get_best_value('val_accuracy')
    
    print("\n" + "-" * 50)
    print("BEST HYPERPARAMETERS")
    print("-" * 50)
    print(f"  Val Accuracy: {best_val_acc:.4f}")
    print(f"  Number of layers: {best_hps.get('n_layers')}")
    print(f"  Units per layer: {best_hps.get('units')}")
    print(f"  Dropout: {best_hps.get('dropout')}")
    print(f"  L2 regularization: {best_hps.get('l2_reg')}")
    print(f"  Learning rate: {best_hps.get('learning_rate')}")
    
    return tuner.get_best_models()[0], best_hps


def train_final_model(best_hps, X_train, X_val, y_train_oh, y_val_oh, 
                      num_classes, model_name='mlp_best'):
    """Train final model with best hyperparameters."""
    print("\n" + "=" * 70)
    print(f"TRAINING FINAL MODEL ({model_name})")
    print("=" * 70)
    
    input_dim = X_train.shape[1]
    model = build_mlp_model(best_hps, input_dim, num_classes)
    model.summary()
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True, 
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, f'{model_name}.keras')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, history


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test, model_name='MLP'):
    """Evaluate model and print metrics."""
    print("\n" + "=" * 70)
    print(f"EVALUATION: {model_name}")
    print("=" * 70)
    
    results = {}
    
    for name, X, y in [('Train', X_train, y_train), 
                       ('Val', X_val, y_val), 
                       ('Test', X_test, y_test)]:
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
        
        results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    
    # Classification report for test
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n" + "-" * 50)
    print("TEST SET CLASSIFICATION REPORT")
    print("-" * 50)
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    return results, y_test_pred


def plot_training_history(history, title='MLP', save_name='training_mlp.png'):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val', linewidth=2)
    ax1.set_title(f'{title} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val', linewidth=2)
    ax2.set_title(f'{title} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150)
    plt.show()


def plot_metrics_comparison(results_dict, title='MLP Comparison'):
    """
    Plot metrics comparison between different models.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping model name -> results dict
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    n_models = len(results_dict)
    x = np.arange(len(metrics))
    width = 0.35 / n_models
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, (model_name, results) in enumerate(results_dict.items()):
        # Plot test results
        test_vals = [results['Test'][m] for m in metrics]
        offset = (i - n_models/2 + 0.5) * width * 2
        bars = ax.bar(x + offset, test_vals, width * 2, label=model_name, 
                      color=colors[i % len(colors)], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, v in zip(bars, test_vals):
            ax.annotate(f'{v:.3f}', 
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', 
                       ha='center', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{title} - Test Set Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'], fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mlp_comparison.png'), dpi=150)
    plt.show()


def plot_kmeans_clusters(X_pca, y, kmeans, title='K-Means Clusters'):
    """Plot first 2 PCA dimensions with K-means cluster centers."""
    if X_pca.shape[1] < 2:
        print("Cannot visualize: need at least 2 dimensions")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: True labels
    scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab20', 
                           alpha=0.6, s=20)
    ax1.set_title('True Labels (PCA)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter1, ax=ax1, label='Subject ID')
    
    # Plot 2: K-means clusters
    cluster_labels = kmeans.predict(X_pca)
    scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                           cmap='tab20', alpha=0.6, s=20)
    # Plot cluster centers
    centers = kmeans.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, 
                edgecolors='black', linewidths=2, label='Centroids')
    ax2.set_title('K-Means Clusters (PCA)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend()
    plt.colorbar(scatter2, ax=ax2, label='Cluster')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'kmeans_clusters.png'), dpi=150)
    plt.show()


def print_comparison_summary(results_dict):
    """Print summary comparison table for all models."""
    print("\n" + "=" * 70)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 70)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    print(f"\n{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 78)
    
    for name, results in results_dict.items():
        test = results['Test']
        print(f"{name:<30} {test['accuracy']:<12.4f} {test['precision']:<12.4f} "
              f"{test['recall']:<12.4f} {test['f1']:<12.4f}")
    
    # Find baseline (first model) and compute improvements
    baseline_name = list(results_dict.keys())[0]
    baseline_results = results_dict[baseline_name]
    
    print("\n" + "-" * 78)
    print(f"Improvements over {baseline_name}:")
    
    for name, results in list(results_dict.items())[1:]:
        print(f"\n  {name}:")
        for metric in metrics:
            diff = results['Test'][metric] - baseline_results['Test'][metric]
            sign = '+' if diff >= 0 else ''
            print(f"    {metric.capitalize()}: {sign}{diff:.4f} ({sign}{diff*100:.2f}%)")
    
    # Best model per metric
    print("\n" + "-" * 78)
    print("Best model per metric:")
    for metric in metrics:
        best_model = max(results_dict.keys(), key=lambda k: results_dict[k]['Test'][metric])
        best_val = results_dict[best_model]['Test'][metric]
        print(f"  {metric.capitalize()}: {best_model} ({best_val:.4f})")


def main():
    """Main execution function."""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = load_data()
    
    # One-hot encode labels
    y_train_oh = to_categorical(y_train, num_classes)
    y_val_oh = to_categorical(y_val, num_classes)
    y_test_oh = to_categorical(y_test, num_classes)
    
    # =========================================================================
    # Step 2: Apply PCA Dimensionality Reduction
    # =========================================================================
    X_train_pca, X_val_pca, X_test_pca, pca, n_pca_components = apply_pca_reduction(
        X_train, X_val, X_test, variance_threshold=0.95
    )
    
    # =========================================================================
    # Step 3: Apply K-Means and Get Distance Features
    # =========================================================================
    X_train_dist, X_val_dist, X_test_dist, kmeans = apply_kmeans_features(
        X_train_pca, X_val_pca, X_test_pca, n_clusters=num_classes
    )
    
    # Combine PCA features with K-means distance features
    X_train_kmeans = combine_features(X_train_pca, X_train_dist)
    X_val_kmeans = combine_features(X_val_pca, X_val_dist)
    X_test_kmeans = combine_features(X_test_pca, X_test_dist)
    
    print(f"\nPCA + K-means features shape: {X_train_kmeans.shape}")
    print(f"  PCA features: {X_train_pca.shape[1]}")
    print(f"  K-means distance features: {X_train_dist.shape[1]}")
    
    # Visualize K-means clusters
    plot_kmeans_clusters(X_train_pca, y_train, kmeans)
    
    # =========================================================================
    # Step 4: Apply GMM and Get Soft Probability Features
    # =========================================================================
    X_train_probs, X_val_probs, X_test_probs, gmm = apply_gmm_features(
        X_train_pca, X_val_pca, X_test_pca, n_components=num_classes
    )
    
    # Combine PCA features with GMM soft probability features
    X_train_gmm = combine_features(X_train_pca, X_train_probs)
    X_val_gmm = combine_features(X_val_pca, X_val_probs)
    X_test_gmm = combine_features(X_test_pca, X_test_probs)
    
    print(f"\nPCA + GMM features shape: {X_train_gmm.shape}")
    print(f"  PCA features: {X_train_pca.shape[1]}")
    print(f"  GMM probability features: {X_train_probs.shape[1]}")
    
    # =========================================================================
    # Step 5: Train MLP with PCA features only
    # =========================================================================
    print("\n" + "#" * 70)
    print("# MODEL 1: MLP with PCA features only")
    print("#" * 70)
    
    # Hyperparameter tuning
    best_model_pca, best_hps_pca = run_hyperband_tuning(
        X_train_pca, X_val_pca, y_train_oh, y_val_oh, num_classes,
        tuner_name='mlp_pca', max_epochs=50
    )
    
    # Train final model
    model_pca, history_pca = train_final_model(
        best_hps_pca, X_train_pca, X_val_pca, y_train_oh, y_val_oh,
        num_classes, model_name='mlp_pca'
    )
    
    # Evaluate
    results_pca, _ = evaluate_model(
        model_pca, X_train_pca, X_val_pca, X_test_pca, 
        y_train, y_val, y_test, model_name='MLP + PCA'
    )
    
    plot_training_history(history_pca, title='MLP + PCA', save_name='training_mlp_pca.png')
    
    # =========================================================================
    # Step 6: Train MLP with PCA + K-means distance features
    # =========================================================================
    print("\n" + "#" * 70)
    print("# MODEL 2: MLP with PCA + K-Means distance features")
    print("#" * 70)
    
    # Hyperparameter tuning
    best_model_kmeans, best_hps_kmeans = run_hyperband_tuning(
        X_train_kmeans, X_val_kmeans, y_train_oh, y_val_oh, num_classes,
        tuner_name='mlp_pca_kmeans', max_epochs=50
    )
    
    # Train final model
    model_kmeans, history_kmeans = train_final_model(
        best_hps_kmeans, X_train_kmeans, X_val_kmeans, y_train_oh, y_val_oh,
        num_classes, model_name='mlp_pca_kmeans'
    )
    
    # Evaluate
    results_kmeans, _ = evaluate_model(
        model_kmeans, X_train_kmeans, X_val_kmeans, X_test_kmeans,
        y_train, y_val, y_test, model_name='MLP + PCA + K-Means'
    )
    
    plot_training_history(history_kmeans, title='MLP + PCA + K-Means', 
                         save_name='training_mlp_pca_kmeans.png')
    
    # =========================================================================
    # Step 7: Train MLP with PCA + GMM soft probability features
    # =========================================================================
    print("\n" + "#" * 70)
    print("# MODEL 3: MLP with PCA + GMM soft probability features")
    print("#" * 70)
    
    # Hyperparameter tuning
    best_model_gmm, best_hps_gmm = run_hyperband_tuning(
        X_train_gmm, X_val_gmm, y_train_oh, y_val_oh, num_classes,
        tuner_name='mlp_pca_gmm', max_epochs=50
    )
    
    # Train final model
    model_gmm, history_gmm = train_final_model(
        best_hps_gmm, X_train_gmm, X_val_gmm, y_train_oh, y_val_oh,
        num_classes, model_name='mlp_pca_gmm'
    )
    
    # Evaluate
    results_gmm, _ = evaluate_model(
        model_gmm, X_train_gmm, X_val_gmm, X_test_gmm,
        y_train, y_val, y_test, model_name='MLP + PCA + GMM'
    )
    
    plot_training_history(history_gmm, title='MLP + PCA + GMM', 
                         save_name='training_mlp_pca_gmm.png')
    
    # =========================================================================
    # Step 8: Compare Results
    # =========================================================================
    all_results = {
        'MLP + PCA': results_pca,
        'MLP + PCA + K-Means': results_kmeans,
        'MLP + PCA + GMM': results_gmm
    }
    
    plot_metrics_comparison(all_results, title='MLP Models')
    
    print_comparison_summary(all_results)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
