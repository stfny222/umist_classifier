"""
Compare All Dimensionality Reduction Methods: PCA vs Autoencoder vs UMAP
=========================================================================

This script provides a comprehensive comparison of three dimensionality
reduction techniques on the UMIST facial recognition dataset:

1. PCA (Principal Component Analysis)
   - Linear, fast, deterministic
   - Supports exact reconstruction

2. Autoencoder (Convolutional)
   - Non-linear, learns complex patterns
   - Supports reconstruction via decoder

3. UMAP (Uniform Manifold Approximation and Projection)
   - Non-linear, preserves local/global structure
   - Best for visualization and clustering
   - Reconstruction via inverse_transform (approximate)

Comparison Metrics:
- Reconstruction Error (MSE, MAE) - for PCA, Autoencoder, and UMAP
- Clustering Performance (Silhouette, NMI, ARI)
- Classification Accuracy (k-NN on reduced features)
- 2D Visualization Quality

Usage:
------
    python compare_all_methods.py

Dependencies:
    pip install numpy scikit-learn tensorflow umap-learn matplotlib seaborn
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from time import time

import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score,
)

from data_preprocessing import load_preprocessed_data_with_augmentation
from dimensionality_reduction.pca import determine_pca_components, fit_and_transform_pca, reconstruct_from_pca
from dimensionality_reduction.autoencoding import build_autoencoder, train_autoencoder, reconstruct_images
from dimensionality_reduction.umap_reduction import fit_and_transform_umap

sns.set_style("whitegrid")

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'comparison_all')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(train_split=0.30, val_split=0.35, test_split=0.35, augmentation_factor=5):
    """Load and prepare the dataset."""
    cache_dir = f'processed_data_{int(train_split*100)}_{int(val_split*100)}_{int(test_split*100)}'
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data_with_augmentation(
        dataset_path='umist_cropped.mat',
        cache_dir=cache_dir,
        augmentation_factor=augmentation_factor,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def apply_pca(X_train, X_val, X_test, n_components=None, variance_threshold=0.95):
    """Apply PCA dimensionality reduction."""
    print("\n" + "-" * 50)
    print("Applying PCA...")
    
    start_time = time()
    
    if n_components is None:
        n_components, _, cum_var, _ = determine_pca_components(
            X_train, variance_threshold=variance_threshold, plot=False
        )
        print(f"  Auto-selected {n_components} components (variance: {cum_var[n_components-1]:.2%})")
    
    X_train_pca, X_val_pca, X_test_pca, pca = fit_and_transform_pca(
        X_train, X_val, X_test, n_components
    )
    
    elapsed = time() - start_time
    print(f"  Time: {elapsed:.2f}s")
    
    return X_train_pca, X_val_pca, X_test_pca, pca, n_components, elapsed


def apply_autoencoder(X_train, X_val, X_test, latent_dim, epochs=50, batch_size=32):
    """Apply Autoencoder dimensionality reduction."""
    print("\n" + "-" * 50)
    print("Applying Autoencoder...")
    
    start_time = time()
    
    autoencoder, encoder, history = train_autoencoder(
        X_train, X_val,
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Get encoded representations
    X_train_reshaped = X_train.reshape(-1, 112, 92, 1)
    X_val_reshaped = X_val.reshape(-1, 112, 92, 1)
    X_test_reshaped = X_test.reshape(-1, 112, 92, 1)
    
    X_train_encoded = encoder.predict(X_train_reshaped, verbose=0)
    X_val_encoded = encoder.predict(X_val_reshaped, verbose=0)
    X_test_encoded = encoder.predict(X_test_reshaped, verbose=0)
    
    elapsed = time() - start_time
    print(f"  Time: {elapsed:.2f}s")
    
    return X_train_encoded, X_val_encoded, X_test_encoded, autoencoder, encoder, elapsed


def apply_umap(X_train, X_val, X_test, n_components, n_neighbors=15, min_dist=0.1):
    """Apply UMAP dimensionality reduction."""
    print("\n" + "-" * 50)
    print("Applying UMAP...")
    
    start_time = time()
    
    X_train_umap, X_val_umap, X_test_umap, umap_model = fit_and_transform_umap(
        X_train, X_val, X_test,
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        verbose=False
    )
    
    elapsed = time() - start_time
    print(f"  Time: {elapsed:.2f}s")
    
    return X_train_umap, X_val_umap, X_test_umap, umap_model, elapsed


def compute_reconstruction_error(X_original, X_reconstructed):
    """Compute MSE and MAE reconstruction errors."""
    mse = np.mean((X_original - X_reconstructed) ** 2)
    mae = np.mean(np.abs(X_original - X_reconstructed))
    return mse, mae


def evaluate_clustering(X, y_true, n_clusters=20):
    """Evaluate clustering performance on reduced features."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    
    silhouette = silhouette_score(X, y_pred) if n_clusters > 1 else 0
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    
    return {'silhouette': silhouette, 'nmi': nmi, 'ari': ari}


def evaluate_classification(X_train, y_train, X_test, y_test, n_neighbors=5):
    """Evaluate k-NN classification on reduced features."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def plot_2d_embeddings(embeddings_dict, y, save_path=None):
    """
    Plot 2D embeddings for all methods side by side.
    
    Parameters
    ----------
    embeddings_dict : dict
        Dictionary mapping method name -> 2D embedding array
    y : np.ndarray
        Labels for coloring
    save_path : str, optional
        Path to save the figure
    """
    n_methods = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (method_name, X_2d) in zip(axes, embeddings_dict.items()):
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=y, cmap='tab20', alpha=0.7, s=30
        )
        ax.set_xlabel("Component 1", fontsize=11)
        ax.set_ylabel("Component 2", fontsize=11)
        ax.set_title(method_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=axes[-1], label="Subject ID")
    plt.suptitle("2D Embeddings Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_reconstruction_comparison(images_dict, n_samples=5, save_path=None):
    """
    Plot reconstruction comparison for all methods.
    
    Parameters
    ----------
    images_dict : dict
        Dictionary mapping method name -> reconstructed images array
    n_samples : int
        Number of samples to display
    save_path : str, optional
        Path to save the figure
    """
    n_methods = len(images_dict)
    fig, axes = plt.subplots(n_methods, n_samples, figsize=(2 * n_samples, 2.2 * n_methods))
    
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    # Use same indices for all methods
    np.random.seed(42)
    indices = np.random.choice(list(images_dict.values())[0].shape[0], n_samples, replace=False)
    
    for row, (method_name, images) in enumerate(images_dict.items()):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            img = images[idx].reshape(112, 92)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if col == 0:
                ax.set_ylabel(method_name, fontsize=10, fontweight='bold')
    
    plt.suptitle("Reconstruction Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_metrics_comparison(results_df, save_path=None):
    """
    Plot bar chart comparing all metrics across methods.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with metrics for each method
    save_path : str, optional
        Path to save the figure
    """
    # Separate reconstruction and performance metrics
    recon_metrics = ['MSE', 'MAE']
    perf_metrics = ['Silhouette', 'NMI', 'ARI', 'k-NN Accuracy']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reconstruction metrics (lower is better)
    recon_data = results_df[results_df['Metric'].isin(recon_metrics)]
    if not recon_data.empty:
        recon_pivot = recon_data.pivot(index='Metric', columns='Method', values='Value')
        recon_pivot.plot(kind='bar', ax=axes[0], rot=0, colormap='Set2')
        axes[0].set_title("Reconstruction Error (Lower is Better)", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Error")
        axes[0].legend(title="Method")
        axes[0].grid(True, alpha=0.3, axis='y')
    
    # Performance metrics (higher is better)
    perf_data = results_df[results_df['Metric'].isin(perf_metrics)]
    if not perf_data.empty:
        perf_pivot = perf_data.pivot(index='Metric', columns='Method', values='Value')
        perf_pivot.plot(kind='bar', ax=axes[1], rot=0, colormap='Set2')
        axes[1].set_title("Performance Metrics (Higher is Better)", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Score")
        axes[1].legend(title="Method")
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].set_ylim(0, 1)
    
    plt.suptitle("Dimensionality Reduction Methods Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def compare_all_methods(
    train_split=0.30,
    val_split=0.35,
    test_split=0.35,
    augmentation_factor=5,
    n_components=None,
    variance_threshold=0.95,
    autoencoder_epochs=50,
):
    """
    Compare PCA, Autoencoder, and UMAP on the UMIST dataset.
    
    Parameters
    ----------
    train_split, val_split, test_split : float
        Data split ratios
    augmentation_factor : int
        Factor to augment training data
    n_components : int, optional
        Number of components (if None, auto-determined by PCA)
    variance_threshold : float
        Variance threshold for PCA component selection
    autoencoder_epochs : int
        Number of epochs for autoencoder training
    
    Returns
    -------
    dict
        Comprehensive results dictionary
    """
    print("=" * 70)
    print("COMPREHENSIVE COMPARISON: PCA vs AUTOENCODER vs UMAP")
    print("=" * 70)
    
    # Load data
    print("\n[1/6] Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_data(
        train_split, val_split, test_split, augmentation_factor
    )
    
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Classes: {len(np.unique(y_train))}")
    
    n_classes = len(np.unique(y_train))
    
    # =========================================================================
    # Apply PCA
    # =========================================================================
    print("\n[2/6] Applying PCA...")
    X_train_pca, X_val_pca, X_test_pca, pca, n_comp, pca_time = apply_pca(
        X_train, X_val, X_test, n_components, variance_threshold
    )
    
    # PCA reconstruction
    X_test_pca_recon = reconstruct_from_pca(X_test_pca, pca, scaler=None)
    pca_mse, pca_mae = compute_reconstruction_error(X_test, X_test_pca_recon)
    print(f"  Reconstruction MSE: {pca_mse:.6f}, MAE: {pca_mae:.6f}")
    
    # =========================================================================
    # Apply Autoencoder
    # =========================================================================
    print("\n[3/6] Applying Autoencoder...")
    X_train_ae, X_val_ae, X_test_ae, autoencoder, encoder, ae_time = apply_autoencoder(
        X_train, X_val, X_test, latent_dim=n_comp, epochs=autoencoder_epochs
    )
    
    # Autoencoder reconstruction
    X_test_ae_recon = reconstruct_images(autoencoder, X_test)
    ae_mse, ae_mae = compute_reconstruction_error(X_test, X_test_ae_recon)
    print(f"  Reconstruction MSE: {ae_mse:.6f}, MAE: {ae_mae:.6f}")
    
    # =========================================================================
    # Apply UMAP
    # =========================================================================
    print("\n[4/6] Applying UMAP...")
    X_train_umap, X_val_umap, X_test_umap, umap_model, umap_time = apply_umap(
        X_train, X_val, X_test, n_components=n_comp
    )
    
    # =========================================================================
    # Evaluate Clustering
    # =========================================================================
    print("\n[5/6] Evaluating clustering performance...")
    
    pca_clustering = evaluate_clustering(X_test_pca, y_test, n_clusters=n_classes)
    ae_clustering = evaluate_clustering(X_test_ae, y_test, n_clusters=n_classes)
    umap_clustering = evaluate_clustering(X_test_umap, y_test, n_clusters=n_classes)
    
    print(f"  PCA:         Silhouette={pca_clustering['silhouette']:.3f}, NMI={pca_clustering['nmi']:.3f}, ARI={pca_clustering['ari']:.3f}")
    print(f"  Autoencoder: Silhouette={ae_clustering['silhouette']:.3f}, NMI={ae_clustering['nmi']:.3f}, ARI={ae_clustering['ari']:.3f}")
    print(f"  UMAP:        Silhouette={umap_clustering['silhouette']:.3f}, NMI={umap_clustering['nmi']:.3f}, ARI={umap_clustering['ari']:.3f}")
    
    # =========================================================================
    # Evaluate Classification
    # =========================================================================
    print("\n[6/6] Evaluating classification performance (k-NN)...")
    
    pca_acc = evaluate_classification(X_train_pca, y_train, X_test_pca, y_test)
    ae_acc = evaluate_classification(X_train_ae, y_train, X_test_ae, y_test)
    umap_acc = evaluate_classification(X_train_umap, y_train, X_test_umap, y_test)
    
    print(f"  PCA:         Accuracy={pca_acc:.3f}")
    print(f"  Autoencoder: Accuracy={ae_acc:.3f}")
    print(f"  UMAP:        Accuracy={umap_acc:.3f}")
    
    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    summary_data = {
        'Metric': ['Dimensions', 'Time (s)', 'MSE', 'MAE', 'Silhouette', 'NMI', 'ARI', 'k-NN Accuracy'],
        'PCA': [n_comp, f"{pca_time:.2f}", f"{pca_mse:.6f}", f"{pca_mae:.6f}",
                f"{pca_clustering['silhouette']:.3f}", f"{pca_clustering['nmi']:.3f}",
                f"{pca_clustering['ari']:.3f}", f"{pca_acc:.3f}"],
        'Autoencoder': [n_comp, f"{ae_time:.2f}", f"{ae_mse:.6f}", f"{ae_mae:.6f}",
                        f"{ae_clustering['silhouette']:.3f}", f"{ae_clustering['nmi']:.3f}",
                        f"{ae_clustering['ari']:.3f}", f"{ae_acc:.3f}"],
        'UMAP': [n_comp, f"{umap_time:.2f}", "N/A", "N/A",
                 f"{umap_clustering['silhouette']:.3f}", f"{umap_clustering['nmi']:.3f}",
                 f"{umap_clustering['ari']:.3f}", f"{umap_acc:.3f}"],
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'comparison_summary.csv'), index=False)
    print(f"\n✓ Summary saved to: {os.path.join(OUTPUT_DIR, 'comparison_summary.csv')}")
    
    # =========================================================================
    # Visualizations
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # 2D embeddings for visualization
    print("\nComputing 2D embeddings for visualization...")
    
    # PCA 2D
    _, _, X_test_pca_2d, _ = fit_and_transform_pca(X_train, X_val, X_test, n_components=2)
    
    # UMAP 2D
    _, _, X_test_umap_2d, _ = fit_and_transform_umap(
        X_train, X_val, X_test, n_components=2, verbose=False
    )
    
    # Autoencoder doesn't directly give 2D, apply PCA on encoded features
    from sklearn.decomposition import PCA as skPCA
    pca_2d = skPCA(n_components=2)
    X_test_ae_2d = pca_2d.fit_transform(X_test_ae)
    
    # Plot 2D embeddings
    embeddings_2d = {
        'PCA': X_test_pca_2d,
        'Autoencoder + PCA': X_test_ae_2d,
        'UMAP': X_test_umap_2d,
    }
    plot_2d_embeddings(
        embeddings_2d, y_test,
        save_path=os.path.join(OUTPUT_DIR, '2d_embeddings_comparison.png')
    )
    
    # Plot reconstruction comparison
    recon_images = {'Original': X_test, 'PCA': X_test_pca_recon, 'Autoencoder': X_test_ae_recon}
    
    plot_reconstruction_comparison(
        recon_images, n_samples=8,
        save_path=os.path.join(OUTPUT_DIR, 'reconstruction_comparison.png')
    )
    
    # Plot metrics comparison
    metrics_data = []
    for method in ['PCA', 'Autoencoder', 'UMAP']:
        if method == 'PCA':
            metrics = {'MSE': pca_mse, 'MAE': pca_mae, 'Silhouette': pca_clustering['silhouette'],
                      'NMI': pca_clustering['nmi'], 'ARI': pca_clustering['ari'], 'k-NN Accuracy': pca_acc}
        elif method == 'Autoencoder':
            metrics = {'MSE': ae_mse, 'MAE': ae_mae, 'Silhouette': ae_clustering['silhouette'],
                      'NMI': ae_clustering['nmi'], 'ARI': ae_clustering['ari'], 'k-NN Accuracy': ae_acc}
        else:
            metrics = {'MSE': 0, 'MAE': 0,
                      'Silhouette': umap_clustering['silhouette'], 'NMI': umap_clustering['nmi'],
                      'ARI': umap_clustering['ari'], 'k-NN Accuracy': umap_acc}
        
        for metric_name, value in metrics.items():
            metrics_data.append({'Method': method, 'Metric': metric_name, 'Value': value})
    
    metrics_df = pd.DataFrame(metrics_data)
    plot_metrics_comparison(
        metrics_df,
        save_path=os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
    )
    
    # =========================================================================
    # Return Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("   - comparison_summary.csv")
    print("   - 2d_embeddings_comparison.png")
    print("   - reconstruction_comparison.png")
    print("   - metrics_comparison.png")
    
    return {
        'pca': {
            'reduced': (X_train_pca, X_val_pca, X_test_pca),
            'reconstructed': X_test_pca_recon,
            'model': pca,
            'mse': pca_mse, 'mae': pca_mae,
            'clustering': pca_clustering,
            'classification_acc': pca_acc,
            'time': pca_time,
        },
        'autoencoder': {
            'reduced': (X_train_ae, X_val_ae, X_test_ae),
            'reconstructed': X_test_ae_recon,
            'model': (autoencoder, encoder),
            'mse': ae_mse, 'mae': ae_mae,
            'clustering': ae_clustering,
            'classification_acc': ae_acc,
            'time': ae_time,
        },
        'umap': {
            'reduced': (X_train_umap, X_val_umap, X_test_umap),
            'model': umap_model,
            'clustering': umap_clustering,
            'classification_acc': umap_acc,
            'time': umap_time,
        },
        'n_components': n_comp,
        'summary': summary_df,
    }


if __name__ == "__main__":
    results = compare_all_methods(
        train_split=0.30,
        val_split=0.35,
        test_split=0.35,
        augmentation_factor=5,
        autoencoder_epochs=50,
    )
    
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("""
1. RECONSTRUCTION QUALITY:
   - PCA: Linear, exact reconstruction
   - Autoencoder: Non-linear, learns complex patterns
   - UMAP: Approximate inverse (if available)

2. CLUSTERING (Unsupervised):
   - Look at Silhouette, NMI, and ARI scores
   - UMAP typically excels at preserving cluster structure

3. CLASSIFICATION (Supervised):
   - k-NN accuracy shows discriminative power
   - Compare which features separate classes best

4. TRADE-OFFS:
   - PCA: Fast, interpretable, linear
   - Autoencoder: Slower training, non-linear, flexible
   - UMAP: Good for clustering/visualization, non-linear
    """)
