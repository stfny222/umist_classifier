"""
SVM Classifier on LDA-Reduced Data
===================================

This script trains a Support Vector Machine (SVM) classifier on LDA-reduced
UMIST face recognition data. It includes:

1. Data loading and preprocessing
2. PCA -> LDA dimensionality reduction
3. SVM hyperparameter tuning via GridSearchCV
4. Model evaluation with comprehensive metrics
5. Visualization of results

Usage:
------
    python svm_lda.py

Dependencies:
    pip install scikit-learn numpy matplotlib seaborn
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_preprocessed_data_with_augmentation
from dimensionality_reduction.pca import (
    determine_pca_components_for_lda,
    fit_and_transform_pca_lda,
)

sns.set_style("whitegrid")


def load_and_reduce_data():
    """Load data and apply PCA->LDA dimensionality reduction.
    
    Returns
    -------
    tuple
        LDA-reduced train/val/test sets and labels
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "umist_cropped.mat"
    )
    
    print("=" * 70)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 70)
    
    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = (
        load_preprocessed_data_with_augmentation(
            dataset_path=path,
        )
    )
    
    print(f"\nOriginal data shapes:")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Feature dimensionality: {X_train.shape[1]}")
    print(f"  Number of classes: {len(np.unique(y_train))}")
    
    # Apply PCA -> LDA dimensionality reduction
    print("\n" + "=" * 70)
    print("APPLYING PCA -> LDA DIMENSIONALITY REDUCTION")
    print("=" * 70)
    
    n_pca_components, _, _, _ = determine_pca_components_for_lda(
        X_train, y_train, plot=True
    )
    
    X_train_lda, X_val_lda, X_test_lda, pca, lda = fit_and_transform_pca_lda(
        X_train, X_val, X_test, y_train, n_pca_components
    )
    
    print(f"\nDimensionality reduction summary:")
    print(f"  Original -> PCA -> LDA: {X_train.shape[1]} -> {n_pca_components} -> {X_train_lda.shape[1]}")
    
    return X_train_lda, X_val_lda, X_test_lda, y_train, y_val, y_test


def grid_search_svm(X_train, y_train):
    """Perform GridSearchCV to find optimal SVM hyperparameters.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features (LDA-reduced)
    y_train : np.ndarray
        Training labels
    
    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object with best model
    """
    print("\n" + "=" * 70)
    print("SVM HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 70)
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear', 'poly'],
        'degree': [2, 3, 4],  # Only used for poly kernel
    }
    
    # Note: degree is only relevant for poly kernel, but sklearn handles this
    # We can also use a more targeted approach
    param_grid_refined = [
        # RBF kernel
        {
            'kernel': ['rbf'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        },
        # Linear kernel
        {
            'kernel': ['linear'],
            'C': [0.1, 1, 10, 100],
        },
        # Polynomial kernel
        {
            'kernel': ['poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'degree': [2, 3, 4],
        },
    ]
    
    print("Parameter grid:")
    for i, params in enumerate(param_grid_refined):
        print(f"  Config {i+1}: {params}")
    
    # Initialize SVM
    svm = SVC(random_state=42)

    
    grid_search = GridSearchCV(
        svm,
        param_grid_refined,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    
    grid_search.fit(X_train, y_train)
    
    # Display results
    print("\n" + "-" * 50)
    print("GRID SEARCH RESULTS")
    print("-" * 50)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    # Display top 5 configurations
    results = grid_search.cv_results_
    sorted_indices = np.argsort(results['rank_test_score'])[:5]
    
    print("\nTop 5 configurations:")
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"  {rank}. Score: {results['mean_test_score'][idx]:.4f} "
              f"(+/- {results['std_test_score'][idx]:.4f}) - "
              f"{results['params'][idx]}")
    
    return grid_search


def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate the trained SVM model on all splits.
    
    Parameters
    ----------
    model : SVC
        Trained SVM model
    X_train, X_val, X_test : np.ndarray
        Feature matrices
    y_train, y_val, y_test : np.ndarray
        Labels
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for each split
    splits = {
        'Train': (y_train, y_train_pred),
        'Validation': (y_val, y_val_pred),
        'Test': (y_test, y_test_pred),
    }
    
    metrics_summary = {}
    
    for split_name, (y_true, y_pred) in splits.items():
        print(f"\n{'-' * 50}")
        print(f"{split_name.upper()} SET METRICS")
        print(f"{'-' * 50}")
        
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Precision (macro):  {precision_macro:.4f}")
        print(f"Precision (weighted): {precision_weighted:.4f}")
        print(f"Recall (macro):     {recall_macro:.4f}")
        print(f"Recall (weighted):  {recall_weighted:.4f}")
        print(f"F1-Score (macro):   {f1_macro:.4f}")
        print(f"F1-Score (weighted): {f1_weighted:.4f}")
        
        metrics_summary[split_name] = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
        }
    
    # Detailed classification report for test set
    print(f"\n{'=' * 70}")
    print("DETAILED CLASSIFICATION REPORT (TEST SET)")
    print("=" * 70)
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    return metrics_summary, y_test_pred



def plot_metrics_comparison(metrics_summary):
    """Plot comparison of metrics across train/val/test splits.
    
    Parameters
    ----------
    metrics_summary : dict
        Dictionary containing metrics for each split
    """
    splits = list(metrics_summary.keys())
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    
    for i, (split, color) in enumerate(zip(splits, colors)):
        values = [metrics_summary[split][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=split, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Score')
    ax.set_title('SVM Performance Metrics Comparison (LDA-Reduced Data)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    # Load and reduce data
    X_train_lda, X_val_lda, X_test_lda, y_train, y_val, y_test = load_and_reduce_data()
    
    
    print(f"\nCombined train+val for grid search: {X_train_lda.shape}")
    
    # Perform grid search
    grid_search = grid_search_svm(X_train_lda, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on all splits
    metrics_summary, y_test_pred = evaluate_model(
        best_model,
        X_train_lda, X_val_lda, X_test_lda,
        y_train, y_val, y_test
    )
    
    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Plot metrics comparison
    plot_metrics_comparison(metrics_summary)
    
    # Cross-validation scores on combined train+val
    print("\n" + "=" * 70)
    print("FINAL CROSS-VALIDATION SCORES")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("SVM CLASSIFICATION COMPLETE")
    print("=" * 70)
    print(f"\nBest model parameters: {grid_search.best_params_}")
    print(f"Test set accuracy: {metrics_summary['Test']['accuracy']:.4f}")
    print(f"Test set F1-score (weighted): {metrics_summary['Test']['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
