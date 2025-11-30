"""
Agglomerative Clustering Comparison: PCA vs UMAP
=================================================

This script performs agglomerative hierarchical clustering on the UMIST facial 
recognition dataset using two unsupervised dimensionality reduction methods:

1. PCA (Principal Component Analysis)
   - Linear dimensionality reduction
   - Preserves global variance structure
   - Fast and deterministic

2. UMAP (Uniform Manifold Approximation and Projection)
   - Non-linear dimensionality reduction
   - Preserves local and global structure
   - Better at preserving cluster structure

Both methods are fully UNSUPERVISED, making this a valid unsupervised learning
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

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_preprocessing import load_preprocessed_data
from dimensionality_reduction.pca import determine_pca_components, fit_and_transform_pca
from dimensionality_reduction.umap_reduction import fit_and_transform_umap

from evaluation import (
    evaluate_clustering, 
    print_metrics,
    evaluate_dimensionality_reduction,
    print_dimred_metrics,
)
from visualization import (
    plot_dendrogram,
    plot_metric_comparison,
    plot_clustering_2d,
    plot_summary_table,
    plot_dimred_comparison,
)


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
    print("AGGLOMERATIVE CLUSTERING: PCA vs UMAP (Unsupervised Comparison)")
    print("=" * 70)
    
    print("\nLoading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data(
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
    # Step 3: Evaluate Dimensionality Reduction Quality
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Dimensionality Reduction Quality Evaluation")
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
    
    # Summary comparison
    print("\n" + "-" * 70)
    print("Dimensionality Reduction Comparison:")
    print("-" * 70)
    print(f"{'Metric':<25} {'PCA':<15} {'UMAP':<15}")
    print("-" * 55)
    print(f"{'Trustworthiness':<25} {pca_dimred_metrics['trustworthiness']:<15.4f} {umap_dimred_metrics['trustworthiness']:<15.4f}")
    print(f"{'Reconstruction Error':<25} {pca_dimred_metrics['reconstruction_error']:<15.4f} {'N/A':<15}")
    print(f"{'Relative Recon Error':<25} {pca_dimred_metrics['relative_recon_error']*100:<14.2f}% {'N/A':<15}")
    
    # =========================================================================
    # Step 4: Agglomerative Clustering
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Agglomerative Clustering")
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
    
    # Store results
    results_dict = {
        "PCA": results_pca,
        "UMAP": results_umap,
    }
    
    # =========================================================================
    # Step 5: Visualizations
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Visualizations")
    print("=" * 70)
    
    # Dimensionality reduction comparison
    print("\nGenerating dimensionality reduction comparison plot...")
    plot_dimred_comparison(
        pca_dimred_metrics, umap_dimred_metrics,
        save_path=os.path.join(output_dir, "dimred_comparison.png")
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
    
    # Metric comparison
    print("\nGenerating metric comparison plot...")
    plot_metric_comparison(
        results_dict,
        save_path=os.path.join(output_dir, "metric_comparison.png")
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
    
    plot_clustering_2d(
        X_combined_pca[:, :2], labels_pca, y_combined,
        title=f"PCA Clustering (k={n_classes})",
        save_path=os.path.join(output_dir, "clustering_pca_2d.png")
    )
    
    plot_clustering_2d(
        X_combined_umap_2d, labels_umap, y_combined,
        title=f"UMAP Clustering (k={n_classes})",
        save_path=os.path.join(output_dir, "clustering_umap_2d.png")
    )
    
    # Summary table
    print("\nGenerating summary table...")
    summary_df = plot_summary_table(
        results_dict, n_classes,
        save_path=os.path.join(output_dir, "summary_table.png")
    )
    
    # =========================================================================
    # Step 6: Summary
    # =========================================================================
    print_summary(results_dict, n_classes)
    
    # Save results to CSV
    results_pca["method"] = "PCA"
    results_umap["method"] = "UMAP"
    all_results = pd.concat([results_pca, results_umap], ignore_index=True)
    results_path = os.path.join(output_dir, "clustering_results.csv")
    all_results.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to {results_path}")
    
    return results_dict


if __name__ == "__main__":
    results = main()
