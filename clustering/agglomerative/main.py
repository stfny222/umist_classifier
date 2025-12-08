"""
Agglomerative Clustering Comparison: PCA vs UMAP vs Autoencoder
================================================================

This script performs agglomerative hierarchical clustering on the UMIST facial 
recognition dataset using three unsupervised dimensionality reduction methods:

1. PCA (Principal Component Analysis)
   - Linear dimensionality reduction
   - Preserves global variance structure
   - Fast and deterministic

2. UMAP (Uniform Manifold Approximation and Projection)
   - Non-linear dimensionality reduction
   - Preserves local and global structure
   - Better at preserving cluster structure

3. Autoencoder (Convolutional)
   - Non-linear dimensionality reduction
   - Learns hierarchical features
   - Can capture complex patterns

All methods are fully UNSUPERVISED, making this a valid unsupervised learning
pipeline for clustering evaluation.

Evaluation Metrics:
- Silhouette Score: Cluster cohesion vs separation (-1 to 1, higher is better)
- Cluster Purity: Homogeneity of clusters w.r.t. true labels (0 to 1)
- Normalized Mutual Information (NMI): Information shared between clusters and labels
- Adjusted Rand Index (ARI): Similarity between clustering and true labels

Usage:
    python main.py

Outputs are saved to the 'outputs/' directory.
"""

import os
import sys
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering

# Add parent directories to path BEFORE imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_preprocessing.pipeline import load_preprocessed_data_with_augmentation
from dimensionality_reduction.pca import determine_pca_components, fit_and_transform_pca
from dimensionality_reduction.umap_reduction import fit_and_transform_umap
from dimensionality_reduction.autoencoding import train_autoencoder, reconstruct_images

from shared import (
    evaluate_clustering, 
    print_metrics,
    evaluate_dimensionality_reduction,
    print_dimred_metrics,
    plot_metric_comparison,
    plot_clustering_2d,
    plot_summary_table,
    plot_dimred_comparison,
    plot_cluster_images_comparison,
)
from visualization import plot_dendrogram
import matplotlib.pyplot as plt


def plot_cluster_purity_comparison(labels_dict, y_true, n_classes, save_path=None):
    """
    Plot cluster purity distribution for each method side by side.
    
    Parameters
    ----------
    labels_dict : dict
        Dictionary mapping method name -> cluster labels array
    y_true : np.ndarray
        True labels
    n_classes : int
        Number of clusters
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, len(labels_dict), figsize=(5 * len(labels_dict), 5))
    
    if len(labels_dict) == 1:
        axes = [axes]
    
    colors = ['steelblue', 'darkorange', 'green']
    
    for idx, (method_name, cluster_labels) in enumerate(labels_dict.items()):
        purities = []
        cluster_sizes = []
        
        for cluster_id in range(n_classes):
            cluster_mask = cluster_labels == cluster_id
            if np.sum(cluster_mask) > 0:
                true_labels_in_cluster = y_true[cluster_mask]
                most_common = np.bincount(true_labels_in_cluster).argmax()
                purity = np.sum(true_labels_in_cluster == most_common) / len(true_labels_in_cluster)
                purities.append(purity)
                cluster_sizes.append(np.sum(cluster_mask))
        
        ax = axes[idx]
        bars = ax.bar(range(len(purities)), purities, color=colors[idx % len(colors)], 
                      edgecolor='black', alpha=0.8)
        ax.set_xlabel('Cluster ID', fontsize=11)
        ax.set_ylabel('Purity', fontsize=11)
        ax.set_title(f'{method_name}\nMean Purity: {np.mean(purities):.3f}', 
                     fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add size labels on top of bars
        for bar, size in zip(bars, cluster_sizes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'n={size}', ha='center', va='bottom', fontsize=7, rotation=45)
    
    plt.suptitle('Per-Cluster Purity Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def run_agglomerative_clustering(X, y, k_values, linkage_method="ward", method_name=""):
    """
    Run agglomerative clustering for different k values and evaluate performance.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        True labels for evaluation
    k_values : list
        List of cluster numbers to try
    linkage_method : str
        Linkage criterion: 'ward', 'complete', 'average', 'single'
    method_name : str
        Name for display purposes
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with metrics for each k value
    """
    results = []
    
    print(f"\n{'='*70}")
    print(f"Agglomerative Clustering on {method_name}")
    print(f"{'='*70}")
    print(f"Linkage: {linkage_method}, Feature shape: {X.shape}")
    print("-" * 70)
    
    for k in k_values:
        agg = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage_method,
        )
        cluster_labels = agg.fit_predict(X)

        # Calculate metrics
        metrics = evaluate_clustering(X, y, cluster_labels)
        
        results.append({
            "k": k,
            "silhouette": metrics["silhouette"],
            "purity": metrics["purity"],
            "nmi": metrics["nmi"],
            "ari": metrics["ari"],
        })
        
        print(
            f"k={k:2d} | Silhouette={metrics['silhouette']:.3f} | "
            f"Purity={metrics['purity']:.3f} | NMI={metrics['nmi']:.3f} | "
            f"ARI={metrics['ari']:.3f}"
        )
    
    return pd.DataFrame(results)


def print_summary(results_dict, n_classes):
    """
    Print summary comparison of results at k = n_classes.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method name -> results DataFrame
    n_classes : int
        True number of classes
    """
    print("\n" + "=" * 70)
    print(f"SUMMARY: Results at k={n_classes} (true number of classes)")
    print("=" * 70)
    
    metrics = ["silhouette", "purity", "nmi", "ari"]
    
    print(f"\n{'Method':<15} {'Silhouette':<12} {'Purity':<12} {'NMI':<12} {'ARI':<12}")
    print("-" * 63)
    
    for method_name, results in results_dict.items():
        row = results[results["k"] == n_classes]
        if not row.empty:
            print(
                f"{method_name:<15} "
                f"{row['silhouette'].values[0]:<12.4f} "
                f"{row['purity'].values[0]:<12.4f} "
                f"{row['nmi'].values[0]:<12.4f} "
                f"{row['ari'].values[0]:<12.4f}"
            )
    
    # Determine winner for each metric
    print("\n" + "-" * 63)
    print("Best method per metric:")
    
    for metric in metrics:
        best_method = None
        best_value = -np.inf
        
        for method_name, results in results_dict.items():
            row = results[results["k"] == n_classes]
            if not row.empty:
                value = row[metric].values[0]
                if value > best_value:
                    best_value = value
                    best_method = method_name
        
        print(f"  {metric.upper()}: {best_method} ({best_value:.4f})")


def main():
    """Main execution function."""
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data_path = os.path.join(project_root, "umist_cropped.mat")
    
    print("=" * 70)
    print("AGGLOMERATIVE CLUSTERING: PCA vs UMAP vs Autoencoder (Unsupervised Comparison)")
    print("=" * 70)
    
    print("\nLoading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data_with_augmentation(
        dataset_path=data_path
    )
    
    # Combine train and val for clustering
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    
    n_classes = len(np.unique(y_combined))
    print(f"Combined data shape: {X_combined.shape}")
    print(f"Number of classes: {n_classes}")
    
    # =========================================================================
    # Step 1: PCA Dimensionality Reduction
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: PCA Dimensionality Reduction")
    print("=" * 70)
    
    n_pca_components, _, _, _ = determine_pca_components(
        X_train, variance_threshold=0.95, plot=True
    )
    
    X_train_pca, X_val_pca, X_test_pca, pca = fit_and_transform_pca(
        X_train, X_val, X_test, n_pca_components
    )
    
    X_combined_pca = np.vstack([X_train_pca, X_val_pca])
    
    # =========================================================================
    # Step 2: UMAP Dimensionality Reduction
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: UMAP Dimensionality Reduction")
    print("=" * 70)
    
    # Use similar number of components as PCA for fair comparison
    n_umap_components = min(n_pca_components, 50)
    
    X_train_umap, X_val_umap, X_test_umap, umap_model = fit_and_transform_umap(
        X_train, X_val, X_test,
        n_components=n_umap_components,
        n_neighbors=15,
        min_dist=0.1,
    )
    
    X_combined_umap = np.vstack([X_train_umap, X_val_umap])
    
    # =========================================================================
    # Step 3: Autoencoder Dimensionality Reduction
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Autoencoder Dimensionality Reduction")
    print("=" * 70)
    
    # Use similar number of components as PCA for fair comparison
    n_ae_components = n_pca_components
    
    # Train autoencoder
    autoencoder, encoder, history = train_autoencoder(
        X_train, X_val,
        latent_dim=n_ae_components,
        epochs=50,
        batch_size=32
    )
    
    # Get encoded representations
    X_train_reshaped = X_train.reshape(-1, 112, 92, 1)
    X_val_reshaped = X_val.reshape(-1, 112, 92, 1)
    X_test_reshaped = X_test.reshape(-1, 112, 92, 1)
    
    X_train_ae = encoder.predict(X_train_reshaped, verbose=0)
    X_val_ae = encoder.predict(X_val_reshaped, verbose=0)
    X_test_ae = encoder.predict(X_test_reshaped, verbose=0)
    
    X_combined_ae = np.vstack([X_train_ae, X_val_ae])
    
    print(f"Autoencoder encoded shape: {X_combined_ae.shape}")
    
    # =========================================================================
    # Step 4: Evaluate Dimensionality Reduction Quality
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Dimensionality Reduction Quality Evaluation")
    print("=" * 70)
    
    # Evaluate PCA
    print("\nEvaluating PCA...")
    pca_dimred_metrics = evaluate_dimensionality_reduction(
        X_combined, X_combined_pca,
        n_neighbors=10,
        pca_model=pca,
        method_name="PCA"
    )
    print_dimred_metrics(pca_dimred_metrics)
    
    # Evaluate UMAP (no reconstruction error - non-linear)
    print("\nEvaluating UMAP...")
    umap_dimred_metrics = evaluate_dimensionality_reduction(
        X_combined, X_combined_umap,
        n_neighbors=10,
        pca_model=None,  # UMAP is non-invertible
        method_name="UMAP"
    )
    print_dimred_metrics(umap_dimred_metrics)
    
    # Evaluate Autoencoder
    print("\nEvaluating Autoencoder...")
    # Compute reconstruction error for autoencoder
    X_combined_reshaped = X_combined.reshape(-1, 112, 92, 1)
    X_combined_ae_recon = autoencoder.predict(X_combined_reshaped, verbose=0).reshape(X_combined.shape[0], -1)
    ae_recon_error = np.mean((X_combined - X_combined_ae_recon) ** 2)
    ae_relative_recon_error = ae_recon_error / np.var(X_combined)
    
    ae_dimred_metrics = evaluate_dimensionality_reduction(
        X_combined, X_combined_ae,
        n_neighbors=10,
        pca_model=None,  # Autoencoder uses its own reconstruction
        method_name="Autoencoder"
    )
    # Add reconstruction metrics manually
    ae_dimred_metrics['reconstruction_error'] = ae_recon_error
    ae_dimred_metrics['relative_recon_error'] = ae_relative_recon_error
    
    # Print autoencoder metrics
    print(f"  Trustworthiness: {ae_dimred_metrics['trustworthiness']:.4f}")
    print(f"  Reconstruction Error: {ae_recon_error:.4f}")
    print(f"  Relative Recon Error: {ae_relative_recon_error*100:.2f}%")
    print(f"  N Components: {ae_dimred_metrics['n_components']}")
    
    # Summary comparison
    print("\n" + "-" * 70)
    print("Dimensionality Reduction Comparison:")
    print("-" * 70)
    print(f"{'Metric':<25} {'PCA':<15} {'UMAP':<15} {'Autoencoder':<15}")
    print("-" * 70)
    print(f"{'Trustworthiness':<25} {pca_dimred_metrics['trustworthiness']:<15.4f} {umap_dimred_metrics['trustworthiness']:<15.4f} {ae_dimred_metrics['trustworthiness']:<15.4f}")
    print(f"{'Reconstruction Error':<25} {pca_dimred_metrics['reconstruction_error']:<15.4f} {'N/A':<15} {ae_dimred_metrics['reconstruction_error']:<15.4f}")
    print(f"{'Relative Recon Error':<25} {pca_dimred_metrics['relative_recon_error']*100:<14.2f}% {'N/A':<15} {ae_dimred_metrics['relative_recon_error']*100:<14.2f}%")
    
    # =========================================================================
    # Step 5: Agglomerative Clustering
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Agglomerative Clustering")
    print("=" * 70)
    
    # Define k values to test
    k_values = list(range(5, min(n_classes + 15, 35), 5))
    if n_classes not in k_values:
        k_values.append(n_classes)
    k_values = sorted(set(k_values))
    
    print(f"\nTesting k values: {k_values}")
    
    # Clustering on PCA features
    results_pca = run_agglomerative_clustering(
        X_combined_pca, y_combined, k_values,
        linkage_method="ward", method_name="PCA Features"
    )
    
    # Clustering on UMAP features
    results_umap = run_agglomerative_clustering(
        X_combined_umap, y_combined, k_values,
        linkage_method="ward", method_name="UMAP Features"
    )
    
    # Clustering on Autoencoder features
    results_ae = run_agglomerative_clustering(
        X_combined_ae, y_combined, k_values,
        linkage_method="ward", method_name="Autoencoder Features"
    )
    
    # Store results
    results_dict = {
        "PCA": results_pca,
        "UMAP": results_umap,
        "Autoencoder": results_ae,
    }
    
    # =========================================================================
    # Step 6: Visualizations
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Visualizations")
    print("=" * 70)
    
    # Dimensionality reduction comparison
    print("\nGenerating dimensionality reduction comparison plot...")
    plot_dimred_comparison(
        pca_dimred_metrics, umap_dimred_metrics, ae_dimred_metrics,
        save_path=os.path.join(output_dir, "dimred_comparison.png"),
        algorithm_name="Agglomerative"
    )
    
    # Dendrograms
    print("\nGenerating dendrograms...")
    plot_dendrogram(
        X_combined_pca, y_combined,
        n_clusters=n_classes,
        title="Dendrogram - PCA Features",
        linkage_method='ward',
        distance_metric='euclidean',
        save_path=os.path.join(output_dir, "dendrogram_pca.png")
    )
    
    plot_dendrogram(
        X_combined_umap, y_combined,
        n_clusters=n_classes,
        title="Dendrogram - UMAP Features",
        linkage_method='ward',
        distance_metric='euclidean',
        save_path=os.path.join(output_dir, "dendrogram_umap.png")
    )
    
    plot_dendrogram(
        X_combined_ae, y_combined,
        n_clusters=n_classes,
        title="Dendrogram - Autoencoder Features",
        linkage_method='ward',
        distance_metric='euclidean',
        save_path=os.path.join(output_dir, "dendrogram_autoencoder.png")
    )
    
    # Metric comparison
    print("\nGenerating metric comparison plot...")
    plot_metric_comparison(
        results_dict,
        save_path=os.path.join(output_dir, "metric_comparison.png"),
        algorithm_name="Agglomerative"
    )
    
    # 2D visualizations at k = n_classes
    print("\nGenerating 2D clustering visualizations...")
    
    # For 2D visualization, we need 2D features
    X_train_umap_2d, X_val_umap_2d, _, _ = fit_and_transform_umap(
        X_train, X_val, X_test,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        verbose=False,
    )
    X_combined_umap_2d = np.vstack([X_train_umap_2d, X_val_umap_2d])
    
    # Cluster and visualize
    agg_pca = AgglomerativeClustering(n_clusters=n_classes, linkage="ward")
    labels_pca = agg_pca.fit_predict(X_combined_pca)
    
    agg_umap = AgglomerativeClustering(n_clusters=n_classes, linkage="ward")
    labels_umap = agg_umap.fit_predict(X_combined_umap)
    
    agg_ae = AgglomerativeClustering(n_clusters=n_classes, linkage="ward")
    labels_ae = agg_ae.fit_predict(X_combined_ae)
    
    # For autoencoder 2D, use UMAP on the encoded features
    from sklearn.decomposition import PCA as skPCA
    pca_2d_ae = skPCA(n_components=2)
    X_combined_ae_2d = pca_2d_ae.fit_transform(X_combined_ae)
    
    plot_clustering_2d(
        X_combined_pca[:, :2], labels_pca, y_combined,
        title=f"Clustering - PCA Features (k={n_classes})",
        save_path=os.path.join(output_dir, "clustering_pca_2d.png"),
        algorithm_name="Agglomerative"
    )
    
    plot_clustering_2d(
        X_combined_umap_2d, labels_umap, y_combined,
        title=f"Clustering - UMAP Features (k={n_classes})",
        save_path=os.path.join(output_dir, "clustering_umap_2d.png"),
        algorithm_name="Agglomerative"
    )
    
    plot_clustering_2d(
        X_combined_ae_2d, labels_ae, y_combined,
        title=f"Clustering - Autoencoder Features (k={n_classes})",
        save_path=os.path.join(output_dir, "clustering_autoencoder_2d.png"),
        algorithm_name="Agglomerative"
    )
    
    # =========================================================================
    # Cluster Images Comparison - Show actual face images in clusters
    # =========================================================================
    print("\nGenerating cluster images comparison...")
    
    labels_dict = {
        "PCA": labels_pca,
        "UMAP": labels_umap,
        "Autoencoder": labels_ae,
    }
    
    # Plot actual images in clusters (showing a subset of clusters)
    plot_cluster_images_comparison(
        X_combined, labels_dict, y_combined,
        n_clusters_to_show=5,
        n_images_per_cluster=6,
        save_path=os.path.join(output_dir, "cluster_images_comparison.png")
    )
    
    # Plot per-cluster purity comparison
    print("\nGenerating per-cluster purity comparison...")
    plot_cluster_purity_comparison(
        labels_dict, y_combined, n_classes,
        save_path=os.path.join(output_dir, "cluster_purity_comparison.png")
    )
    
    # Summary table
    print("\nGenerating summary table...")
    summary_df = plot_summary_table(
        results_dict, n_classes,
        save_path=os.path.join(output_dir, "summary_table.png"),
        algorithm_name="Agglomerative"
    )
    
    # =========================================================================
    # Step 7: Summary
    # =========================================================================
    print_summary(results_dict, n_classes)
    
    # Save results to CSV
    results_pca["method"] = "PCA"
    results_umap["method"] = "UMAP"
    results_ae["method"] = "Autoencoder"
    all_results = pd.concat([results_pca, results_umap, results_ae], ignore_index=True)
    results_path = os.path.join(output_dir, "clustering_results.csv")
    all_results.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    return results_dict


if __name__ == "__main__":
    results = main()
