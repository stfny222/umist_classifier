"""
K-Means Clustering Comparison: PCA vs UMAP vs Autoencoding
===========================================================

This script performs K-means clustering on the UMIST facial recognition dataset
using four unsupervised dimensionality reduction methods:

1. PCA (Principal Component Analysis)
   - Linear dimensionality reduction
   - Preserves global variance structure
   - Fast and deterministic

2. UMAP (Uniform Manifold Approximation and Projection)
   - Non-linear dimensionality reduction
   - Preserves local and global structure
   - Better at preserving cluster structure

3. Standard Convolutional Autoencoder
   - Neural network-based dimensionality reduction
   - Learns non-linear feature representations
   - ReLU activations

4. Improved Autoencoder with SSIM Loss
   - Enhanced autoencoder with perceptual loss
   - LeakyReLU activations
   - L2 regularization and dropout
   - Optimized for structural similarity

All methods are fully UNSUPERVISED, making this a valid unsupervised learning
pipeline for clustering evaluation.

Evaluation Metrics:
- Silhouette Score: Cluster cohesion vs separation (-1 to 1, higher is better)
- Cluster Purity: Homogeneity of clusters w.r.t. true labels (0 to 1)
- Normalized Mutual Information (NMI): Information shared between clusters and labels
- Adjusted Rand Index (ARI): Similarity between clustering and true labels
- Inertia: Sum of squared distances to nearest cluster center (lower is better)
- Trustworthiness: Local neighborhood preservation in reduced space
- Continuity: Global structure preservation in reduced space

Usage:
    python main.py

Outputs are saved to the 'outputs/' directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_preprocessing import load_preprocessed_data
from dimensionality_reduction.pca import determine_pca_components, fit_and_transform_pca
from dimensionality_reduction.umap_reduction import fit_and_transform_umap

from shared import (
    evaluate_dimensionality_reduction,
    print_dimred_metrics,
)
from shared import (
    plot_metric_comparison,
    plot_clustering_2d,
    plot_summary_table,
    plot_dimred_comparison,
)

# Import utilities from clustering_utils
from clustering_utils import (
    find_optimal_k,
    print_optimal_k_summary,
    extract_autoencoder_features,
    run_clustering_pipeline,
)


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
    print("K-MEANS CLUSTERING: PCA vs UMAP vs Autoencoding (Unsupervised Comparison)")
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
    # Step 3: Autoencoder Dimensionality Reduction
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Autoencoder Dimensionality Reduction")
    print("=" * 70)
    
    # Use n_pca_components as latent dimension for fair comparison
    latent_dim = n_pca_components
    
    # Standard Autoencoder
    X_train_ae, X_val_ae, X_test_ae, encoder_ae = extract_autoencoder_features(
        X_train, X_val, X_test,
        latent_dim=latent_dim,
        improved=False,
        epochs=50,
        batch_size=32,
        method_name=f"Standard Autoencoder (latent_dim={latent_dim})"
    )
    X_combined_ae = np.vstack([X_train_ae, X_val_ae])
    
    # Improved Autoencoder with SSIM loss
    print("\n" + "-" * 70)
    X_train_ae_improved, X_val_ae_improved, X_test_ae_improved, encoder_ae_improved = extract_autoencoder_features(
        X_train, X_val, X_test,
        latent_dim=latent_dim,
        improved=True,
        epochs=50,
        batch_size=32,
        method_name=f"Improved Autoencoder with SSIM (latent_dim={latent_dim})"
    )
    X_combined_ae_improved = np.vstack([X_train_ae_improved, X_val_ae_improved])
    
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
    
    # Evaluate Standard Autoencoder
    print("\nEvaluating Standard Autoencoder...")
    ae_dimred_metrics = evaluate_dimensionality_reduction(
        X_combined, X_combined_ae,
        n_neighbors=10,
        pca_model=None,  # Autoencoder is non-linear
        method_name="Standard Autoencoder"
    )
    print_dimred_metrics(ae_dimred_metrics)
    
    # Evaluate Improved Autoencoder
    print("\nEvaluating Improved Autoencoder (SSIM)...")
    ae_improved_dimred_metrics = evaluate_dimensionality_reduction(
        X_combined, X_combined_ae_improved,
        n_neighbors=10,
        pca_model=None,
        method_name="Improved Autoencoder (SSIM)"
    )
    print_dimred_metrics(ae_improved_dimred_metrics)
    
    # Summary comparison
    print("\n" + "-" * 70)
    print("Dimensionality Reduction Comparison:")
    print("-" * 70)
    print(f"{'Method':<25} {'Trustworthiness':<15} {'Continuity':<15}")
    print("-" * 55)
    print(f"{'PCA':<25} {pca_dimred_metrics['trustworthiness']:<15.4f} {pca_dimred_metrics.get('continuity', np.nan):<15.4f}")
    print(f"{'UMAP':<25} {umap_dimred_metrics['trustworthiness']:<15.4f} {umap_dimred_metrics.get('continuity', np.nan):<15.4f}")
    print(f"{'Autoencoder':<25} {ae_dimred_metrics['trustworthiness']:<15.4f} {ae_dimred_metrics.get('continuity', np.nan):<15.4f}")
    print(f"{'Autoencoder (SSIM)':<25} {ae_improved_dimred_metrics['trustworthiness']:<15.4f} {ae_improved_dimred_metrics.get('continuity', np.nan):<15.4f}")
    
    # =========================================================================
    # Step 5: K-Means Clustering with Optimal K
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: K-Means Clustering with Optimal K")
    print("=" * 70)
    
    # First, find optimal k by testing range
    k_values_test = list(range(5, min(n_classes + 15, 35), 5))
    if n_classes not in k_values_test:
        k_values_test.append(n_classes)
    k_values_test = sorted(set(k_values_test))
    
    # Prepare feature dictionaries for clustering
    X_combined_dict = {
        "PCA": X_combined_pca,
        "UMAP": X_combined_umap,
        "Autoencoder": X_combined_ae,
        "Autoencoder (SSIM)": X_combined_ae_improved,
    }
    
    # Run clustering to get all k results
    results_all = run_clustering_pipeline(
        X_combined_dict, y_combined, k_values_test, "ground_truth (all k)", output_dir
    )
    
    # Find optimal k
    optimal_k_silhouette = find_optimal_k(results_all, method='silhouette')
    
    print("\n" + "=" * 70)
    print("OPTIMAL K VALUES (by Silhouette Score)")
    print("=" * 70)
    for method_name, opt_k in optimal_k_silhouette.items():
        print(f"  {method_name:<25} k={opt_k}")
    
    # Create 2D features for visualization
    X_combined_pca_2d = X_combined_pca[:, :2]
    
    X_train_umap_2d, X_val_umap_2d, _, _ = fit_and_transform_umap(
        X_train, X_val, X_test,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        verbose=False,
    )
    X_combined_umap_2d = np.vstack([X_train_umap_2d, X_val_umap_2d])
    
    # Autoencoders: use first and last components for 2D visualization
    X_combined_ae_2d = X_combined_ae[:, [0, -1]]
    X_combined_ae_improved_2d = X_combined_ae_improved[:, [0, -1]]
    
    X_combined_2d_dict = {
        "PCA": X_combined_pca_2d,
        "UMAP": X_combined_umap_2d,
        "Autoencoder": X_combined_ae_2d,
        "Autoencoder (SSIM)": X_combined_ae_improved_2d,
    }
    
    # =========================================================================
    # Step 6: Visualizations at Optimal K
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Visualizations at Optimal K")
    print("=" * 70)
    
    run_clustering_pipeline(
        X_combined_dict, y_combined, optimal_k_silhouette, "optimal", output_dir,
        X_combined_2d_dict=X_combined_2d_dict,
        X_original=X_combined, y_original=y_combined
    )
    
    # =========================================================================
    # Step 7: Visualizations at Ground Truth K
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: Visualizations at Ground Truth K (k={})".format(n_classes))
    print("=" * 70)
    
    run_clustering_pipeline(
        X_combined_dict, y_combined, k_values_test, "ground_truth", output_dir,
        X_combined_2d_dict=X_combined_2d_dict,
        X_original=X_combined, y_original=y_combined
    )
    
    # =========================================================================
    # Step 8: Generate Summary Tables
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: Generate Summary Tables")
    print("=" * 70)
    
    # Summary table at ground truth k
    print("\nGenerating summary table at ground truth k...")
    summary_df = plot_summary_table(
        results_all, n_classes,
        save_path=os.path.join(output_dir, "summary_table_ground_truth_k.png"),
        algorithm_name="K-Means"
    )
    
    # Summary table at optimal k
    print("Generating summary table at optimal k...")
    optimal_metrics_list = []
    for method_name, opt_k in optimal_k_silhouette.items():
        row = results_all[method_name][results_all[method_name]["k"] == opt_k]
        if not row.empty:
            optimal_metrics_list.append({
                "Method": method_name,
                "k": opt_k,
                "Silhouette": row["silhouette"].values[0],
                "Purity": row["purity"].values[0],
                "NMI": row["nmi"].values[0],
                "ARI": row["ari"].values[0],
            })
    
    # Create summary table figure for optimal k
    if optimal_metrics_list:
        summary_df_optimal = pd.DataFrame(optimal_metrics_list)
        
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare data for table display
        table_data = []
        for _, row in summary_df_optimal.iterrows():
            table_data.append([
                row["Method"],
                f"k={int(row['k'])}",
                f"{row['Silhouette']:.4f}",
                f"{row['Purity']:.4f}",
                f"{row['NMI']:.4f}",
                f"{row['ARI']:.4f}",
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=["Method", "k", "Silhouette", "Purity", "NMI", "ARI"],
            cellLoc='center',
            loc='center',
            colColours=['lightsteelblue'] * 6
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        plt.title("Clustering Results at Optimal k (by Silhouette Score)", fontsize=12, pad=20)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, "summary_table_optimal_k.png"), 
                   bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(output_dir, 'summary_table_optimal_k.png')}")
    
    # =========================================================================
    # Step 9: Metric Comparison Plot
    # =========================================================================
    print("\nGenerating metric comparison plot...")
    plot_metric_comparison(
        results_all,
        save_path=os.path.join(output_dir, "metric_comparison.png"),
        algorithm_name="K-Means"
    )
    
    # =========================================================================
    # Step 10: Optimal K Analysis & Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 10: Optimal K Analysis & Comparison")
    print("=" * 70)
    
    # Find optimal k using elbow method
    optimal_k_elbow = find_optimal_k(results_all, method='elbow')
    
    # Print optimal k summary
    print_optimal_k_summary(results_all, n_classes, optimal_k_silhouette, optimal_k_elbow)
    
    # =========================================================================
    # Step 11: Save All Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 11: Saving Results")
    print("=" * 70)
    
    # Save full clustering results
    results_pca = results_all["PCA"].copy()
    results_umap = results_all["UMAP"].copy()
    results_ae = results_all["Autoencoder"].copy()
    results_ae_improved = results_all["Autoencoder (SSIM)"].copy()
    
    results_pca["method"] = "PCA"
    results_umap["method"] = "UMAP"
    results_ae["method"] = "Autoencoder"
    results_ae_improved["method"] = "Autoencoder (SSIM)"
    
    all_results = pd.concat(
        [results_pca, results_umap, results_ae, results_ae_improved],
        ignore_index=True
    )
    results_path = os.path.join(output_dir, "clustering_results.csv")
    all_results.to_csv(results_path, index=False)
    print(f"✓ Results saved to {results_path}")
    
    # Save optimal k summary
    optimal_k_summary = pd.DataFrame({
        "Method": list(optimal_k_silhouette.keys()),
        "Optimal_k_Silhouette": list(optimal_k_silhouette.values()),
        "Optimal_k_Elbow": list(optimal_k_elbow.values()),
        "Ground_Truth_k": n_classes
    })
    optimal_k_path = os.path.join(output_dir, "optimal_k_summary.csv")
    optimal_k_summary.to_csv(optimal_k_path, index=False)
    print(f"✓ Optimal k summary saved to {optimal_k_path}")
    
    # Save metrics at optimal k
    optimal_metrics_list = []
    for method_name, opt_k in optimal_k_silhouette.items():
        row = results_all[method_name][results_all[method_name]["k"] == opt_k]
        if not row.empty:
            row_dict = row.iloc[0].to_dict()
            row_dict['method'] = method_name
            optimal_metrics_list.append(row_dict)
    
    optimal_metrics_df = pd.DataFrame(optimal_metrics_list)
    optimal_metrics_path = os.path.join(output_dir, "metrics_at_optimal_k.csv")
    optimal_metrics_df.to_csv(optimal_metrics_path, index=False)
    print(f"✓ Metrics at optimal k saved to {optimal_metrics_path}")
    
    # Save dimensionality reduction metrics
    dimred_comparison = pd.DataFrame({
        "Method": ["PCA", "UMAP", "Autoencoder", "Autoencoder (SSIM)"],
        "Trustworthiness": [
            pca_dimred_metrics['trustworthiness'],
            umap_dimred_metrics['trustworthiness'],
            ae_dimred_metrics['trustworthiness'],
            ae_improved_dimred_metrics['trustworthiness']
        ],
        "Continuity": [
            pca_dimred_metrics.get('continuity', np.nan),
            umap_dimred_metrics.get('continuity', np.nan),
            ae_dimred_metrics.get('continuity', np.nan),
            ae_improved_dimred_metrics.get('continuity', np.nan)
        ]
    })
    dimred_path = os.path.join(output_dir, "dimred_metrics.csv")
    dimred_comparison.to_csv(dimred_path, index=False)
    print(f"✓ Dimensionality reduction metrics saved to {dimred_path}")
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    
    return results_all


if __name__ == "__main__":
    results = main()
