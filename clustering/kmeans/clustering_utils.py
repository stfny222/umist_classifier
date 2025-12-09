"""
K-Means Clustering Utility Functions
=====================================

Helper functions for K-means clustering pipeline including:
- Optimal k detection
- Clustering execution
- Results summarization
- Feature extraction from autoencoders
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared import evaluate_clustering


def find_optimal_k(results_dict, k_range=None):
    """
    Find optimal k using silhouette score.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method name -> results DataFrame
    k_range : list, optional
        K values to consider. If None, uses all available k values.
        
    Returns
    -------
    optimal_k_dict : dict
        Dictionary mapping method name -> optimal k value
    """
    optimal_k_dict = {}
    
    for method_name, results_df in results_dict.items():
        # Find k with maximum silhouette score
        best_idx = results_df['silhouette'].idxmax()
        optimal_k = results_df.loc[best_idx, 'k']
        optimal_k_dict[method_name] = int(optimal_k)
    
    return optimal_k_dict


def print_optimal_k_summary(results_dict, n_classes, optimal_k_silhouette):
    """
    Print summary comparison of results at optimal k values.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method name -> results DataFrame
    n_classes : int
        True number of classes
    optimal_k_silhouette : dict
        Optimal k by silhouette method
    """
    metrics = ["silhouette", "purity", "nmi", "ari"]
    
    print("\n" + "=" * 80)
    print("OPTIMAL K COMPARISON: Silhouette Method")
    print("=" * 80)
    print(f"\nOptimal k values (by silhouette score):")
    for method_name, opt_k in optimal_k_silhouette.items():
        print(f"  {method_name:<25} k={opt_k}")
    
    print(f"\n{'Method':<25} {'k':<4} {'Silhouette':<12} {'Purity':<12} {'NMI':<12} {'ARI':<12}")
    print("-" * 80)
    
    for method_name, results in results_dict.items():
        opt_k = optimal_k_silhouette[method_name]
        row = results[results["k"] == opt_k]
        if not row.empty:
            print(
                f"{method_name:<25} {int(opt_k):<4} "
                f"{row['silhouette'].values[0]:<12.4f} "
                f"{row['purity'].values[0]:<12.4f} "
                f"{row['nmi'].values[0]:<12.4f} "
                f"{row['ari'].values[0]:<12.4f}"
            )
    
    # Determine winner for each metric at optimal k
    print("\n" + "-" * 80)
    print("Best method per metric (at optimal silhouette k):")
    
    for metric in metrics:
        best_method = None
        best_value = -np.inf
        
        for method_name, results in results_dict.items():
            opt_k = optimal_k_silhouette[method_name]
            row = results[results["k"] == opt_k]
            if not row.empty:
                value = row[metric].values[0]
                if value > best_value:
                    best_value = value
                    best_method = method_name
        
        print(f"  {metric.upper()}: {best_method} ({best_value:.4f})")
    
    # Comparison: Ground Truth k vs Optimal k
    print("\n" + "=" * 80)
    print(f"COMPARISON: Ground Truth k={n_classes} vs. Optimal k (Silhouette)")
    print("=" * 80)
    print(f"\n{'Method':<25} {'Metric':<15} {'At k={}':<15} {'At Optimal k':<15} {'Δ':<10}")
    print("-" * 80)
    
    for method_name, results in results_dict.items():
        opt_k = optimal_k_silhouette[method_name]
        row_ground = results[results["k"] == n_classes]
        row_optimal = results[results["k"] == opt_k]
        
        for metric in metrics:
            val_ground = row_ground[metric].values[0]
            val_optimal = row_optimal[metric].values[0]
            delta = val_optimal - val_ground
            delta_pct = (delta / val_ground * 100) if val_ground != 0 else 0
            
            metric_label = f"{method_name} - {metric}" if metric == metrics[0] else ""
            print(
                f"{method_name if metric == metrics[0] else '':<25} {metric:<15} "
                f"{val_ground:<15.4f} {val_optimal:<15.4f} "
                f"{delta:+.4f} ({delta_pct:+.1f}%)"
            )
        print("-" * 80)


def extract_autoencoder_features(X_train, X_val, X_test, latent_dim=137, 
                                 improved=False, epochs=50, batch_size=32,
                                 method_name="Autoencoder"):
    """
    Train autoencoder and extract features from bottleneck layer.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training data (flattened)
    X_val : np.ndarray
        Validation data (flattened)
    X_test : np.ndarray
        Test data (flattened)
    latent_dim : int
        Size of bottleneck layer
    improved : bool
        Whether to use improved autoencoder (with SSIM loss)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    method_name : str
        Name for display purposes
        
    Returns
    -------
    X_train_ae : np.ndarray
        Encoded training features
    X_val_ae : np.ndarray
        Encoded validation features
    X_test_ae : np.ndarray
        Encoded test features
    encoder : tf.keras.Model
        Trained encoder model
    """
    # Import here to avoid circular imports
    from dimensionality_reduction.autoencoding import train_autoencoder
    from dimensionality_reduction.autoencoding_improved import train_autoencoder_improved
    
    print(f"\n{'='*70}")
    print(f"Training {method_name} (latent_dim={latent_dim})")
    print(f"{'='*70}")
    
    # Force CPU execution due to potential CuDNN version issues
    with tf.device('/CPU:0'):
        if improved:
            autoencoder, encoder, history = train_autoencoder_improved(
                X_train, X_val,
                latent_dim=latent_dim,
                epochs=epochs,
                batch_size=batch_size,
                dropout_rate=0.2,
                l2_reg=1e-4,
                loss_type='combined'
            )
        else:
            autoencoder, encoder, history = train_autoencoder(
                X_train, X_val,
                latent_dim=latent_dim,
                epochs=epochs,
                batch_size=batch_size
            )
        
        # Extract features using encoder
        print(f"\nExtracting {method_name} features...")
        X_train_ae = encoder.predict(X_train.reshape(-1, 112, 92, 1), verbose=0)
        X_val_ae = encoder.predict(X_val.reshape(-1, 112, 92, 1), verbose=0)
        X_test_ae = encoder.predict(X_test.reshape(-1, 112, 92, 1), verbose=0)
    
    print(f"Extracted features shape: {X_train_ae.shape}")
    
    return X_train_ae, X_val_ae, X_test_ae, encoder


def run_kmeans_clustering(X, y, k_values, n_init=10, method_name=""):
    """
    Run K-means clustering for different k values and evaluate performance.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        True labels for evaluation
    k_values : list
        List of cluster numbers to try
    n_init : int
        Number of times K-means will be run with different centroid seeds
    method_name : str
        Name for display purposes
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with metrics for each k value
    """
    results = []
    
    print(f"\n{'='*70}")
    print(f"K-Means Clustering on {method_name}")
    print(f"{'='*70}")
    print(f"Feature shape: {X.shape}, n_init: {n_init}")
    print("-" * 70)
    
    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=42,
            verbose=0,
        )
        cluster_labels = kmeans.fit_predict(X)

        # Calculate metrics
        metrics = evaluate_clustering(X, y, cluster_labels)
        
        results.append({
            "k": k,
            "silhouette": metrics["silhouette"],
            "purity": metrics["purity"],
            "nmi": metrics["nmi"],
            "ari": metrics["ari"],
            "inertia": kmeans.inertia_,
        })
        
        print(
            f"k={k:2d} | Silhouette={metrics['silhouette']:.3f} | "
            f"Purity={metrics['purity']:.3f} | NMI={metrics['nmi']:.3f} | "
            f"ARI={metrics['ari']:.3f} | Inertia={kmeans.inertia_:.1f}"
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


def run_clustering_pipeline(X_combined_dict, y_combined, k_values, k_type, output_dir, 
                            X_combined_2d_dict=None, X_original=None, y_original=None):
    """
    Run complete clustering pipeline: clustering, visualization, and summary.
    
    Parameters
    ----------
    X_combined_dict : dict
        Dictionary mapping method name -> combined feature array
    y_combined : np.ndarray
        Combined labels
    k_values : list or dict
        If list: test all k values. If dict: optimal k values per method
    k_type : str
        'ground_truth' or 'optimal' for naming outputs
    output_dir : str
        Directory to save outputs
    X_combined_2d_dict : dict, optional
        Dictionary mapping method name -> 2D feature array for visualization
    X_original : np.ndarray, optional
        Original (flattened but not reduced) image data for cluster visualization
    y_original : np.ndarray, optional
        Original labels matching X_original
        
    Returns
    -------
    results_dict : dict
        Results from clustering (or subset for single k if dict provided)
    """
    # Import here to avoid circular imports
    from shared import plot_clustering_2d, plot_cluster_images_comparison
    
    print("\n" + "=" * 70)
    print(f"K-Means Clustering Pipeline ({k_type})")
    print("=" * 70)
    
    # Determine if we're testing multiple k or single k per method
    if isinstance(k_values, dict):
        is_single_k = True
        test_k_values = sorted(set(k_values.values()))
    else:
        is_single_k = False
        test_k_values = k_values
    
    print(f"\nTesting k values: {test_k_values}")
    
    # Run clustering for all methods
    results_dict = {}
    for method_name in X_combined_dict.keys():
        X = X_combined_dict[method_name]
        print(f"\nClustering {method_name}...")
        results_dict[method_name] = run_kmeans_clustering(
            X, y_combined, test_k_values,
            n_init=10, method_name=f"{method_name} Features"
        )
    
    # If single k per method, print summary at that k
    if is_single_k:
        print("\n" + "=" * 70)
        print(f"SUMMARY: Results at Optimal k ({k_type})")
        print("=" * 70)
        
        metrics = ["silhouette", "purity", "nmi", "ari"]
        print(f"\n{'Method':<25} {'k':<4} {'Silhouette':<12} {'Purity':<12} {'NMI':<12} {'ARI':<12}")
        print("-" * 80)
        
        for method_name, opt_k in k_values.items():
            results = results_dict[method_name]
            row = results[results["k"] == opt_k]
            if not row.empty:
                print(
                    f"{method_name:<25} {int(opt_k):<4} "
                    f"{row['silhouette'].values[0]:<12.4f} "
                    f"{row['purity'].values[0]:<12.4f} "
                    f"{row['nmi'].values[0]:<12.4f} "
                    f"{row['ari'].values[0]:<12.4f}"
                )
        
        # Determine winners
        print("\n" + "-" * 80)
        print("Best method per metric:")
        
        for metric in metrics:
            best_method = None
            best_value = -np.inf
            
            for method_name, opt_k in k_values.items():
                results = results_dict[method_name]
                row = results[results["k"] == opt_k]
                if not row.empty:
                    value = row[metric].values[0]
                    if value > best_value:
                        best_value = value
                        best_method = method_name
            
            print(f"  {metric.upper()}: {best_method} ({best_value:.4f})")
    else:
        # Standard summary at ground truth k (k=20 for UMIST dataset)
        ground_truth_k = 20
        print_summary(results_dict, ground_truth_k)
    
    # Generate visualizations if 2D features provided
    if X_combined_2d_dict is not None:
        print("\n" + "=" * 70)
        print(f"VISUALIZATIONS ({k_type})")
        print("=" * 70)
        
        if is_single_k and k_type == "optimal":
            # Visualizations at optimal k
            for method_name, opt_k in k_values.items():
                X_2d = X_combined_2d_dict[method_name]
                kmeans = KMeans(n_clusters=opt_k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(X_combined_dict[method_name])
                
                filename = f"clustering_{method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_2d_optimal_k.png"
                plot_clustering_2d(
                    X_2d, labels, y_combined,
                    title=f"Clustering - {method_name} (Optimal k={opt_k})",
                    save_path=os.path.join(output_dir, filename),
                    algorithm_name="K-Means"
                )
        else:
            # Visualizations at ground truth k (k=20 for UMIST dataset)
            k = 20
            for method_name in X_combined_dict.keys():
                X_2d = X_combined_2d_dict[method_name]
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(X_combined_dict[method_name])
                
                filename = f"clustering_{method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_2d.png"
                plot_clustering_2d(
                    X_2d, labels, y_combined,
                    title=f"Clustering - {method_name} (k={k})",
                    save_path=os.path.join(output_dir, filename),
                    algorithm_name="K-Means"
                )
        
        print("✓ 2D visualizations saved")
    
    # Generate cluster images comparison if original data is provided
    if X_original is not None and y_original is not None:
        print("\n" + "=" * 70)
        print(f"CLUSTER IMAGES COMPARISON ({k_type})")
        print("=" * 70)
        
        if is_single_k and k_type == "optimal":
            # Cluster images at optimal k for each method
            for method_name, opt_k in k_values.items():
                print(f"\nGenerating cluster images for {method_name} at optimal k={opt_k}...")
                
                # Re-fit kmeans on reduced features to get labels matching X_original
                kmeans = KMeans(n_clusters=opt_k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(X_combined_dict[method_name])
                
                labels_dict = {method_name: labels}
                
                filename = f"cluster_images_{method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_optimal_k.png"
                plot_cluster_images_comparison(
                    X_original, labels_dict, y_original,
                    n_clusters_to_show=min(5, opt_k),
                    n_images_per_cluster=6,
                    save_path=os.path.join(output_dir, filename)
                )
        else:
            # Cluster images at ground truth k for all methods (k=20 for UMIST dataset)
            k = 20
            print(f"\nGenerating cluster images for all methods at ground truth k={k}...")
            
            labels_dict = {}
            for method_name in X_combined_dict.keys():
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = kmeans.fit_predict(X_combined_dict[method_name])
                labels_dict[method_name] = labels
            
            filename = f"cluster_images_ground_truth_k.png"
            plot_cluster_images_comparison(
                X_original, labels_dict, y_original,
                n_clusters_to_show=min(5, k),
                n_images_per_cluster=6,
                save_path=os.path.join(output_dir, filename)
            )
        
        print("✓ Cluster images comparison saved")
    
    return results_dict
