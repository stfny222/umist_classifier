"""
Shared visualization functions for clustering results.

These are generic visualization functions that work with any clustering algorithm.
Algorithm-specific visualizations (e.g., dendrograms) are kept in their respective modules.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def plot_metric_comparison(results_dict, save_path=None, algorithm_name=""):
    """
    Plot comparison of clustering metrics across different methods.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method name -> results DataFrame
        Each DataFrame should have columns: k, silhouette, purity, nmi, ari
    save_path : str, optional
        Path to save the figure
    algorithm_name : str, optional
        Name of clustering algorithm (e.g., "K-Means", "Agglomerative")
    """
    metrics = ["silhouette", "purity", "nmi", "ari"]
    metric_names = ["Silhouette Score", "Purity", "NMI", "ARI"]
    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax, metric, name in zip(axes, metrics, metric_names):
        for i, (method_name, results) in enumerate(results_dict.items()):
            ax.plot(
                results["k"], results[metric],
                marker=markers[i % len(markers)],
                linestyle='-',
                linewidth=2,
                markersize=6,
                color=colors[i % len(colors)],
                label=method_name
            )
        
        ax.set_xlabel("Number of Clusters (k)", fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    title = "Clustering: Method Comparison"
    if algorithm_name:
        title = f"{algorithm_name}: Method Comparison"
    plt.suptitle(title, 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_clustering_2d(X, cluster_labels, y_true, title="Clustering Results", save_path=None, algorithm_name=""):
    """
    Visualize clustering results in 2D using first two features/components.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (uses first 2 dimensions for plotting)
    cluster_labels : np.ndarray
        Cluster assignments
    y_true : np.ndarray
        True labels for comparison
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    algorithm_name : str, optional
        Name of clustering algorithm (e.g., "K-Means", "Agglomerative")
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Cluster assignments
    scatter1 = axes[0].scatter(
        X[:, 0], X[:, 1], 
        c=cluster_labels,
        cmap='tab20', alpha=0.7, s=50
    )
    axes[0].set_xlabel("Component 1", fontsize=11)
    axes[0].set_ylabel("Component 2", fontsize=11)
    axes[0].set_title("Cluster Assignments", fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")
    
    # Plot 2: True labels
    scatter2 = axes[1].scatter(
        X[:, 0], X[:, 1], 
        c=y_true,
        cmap='tab20', alpha=0.7, s=50
    )
    axes[1].set_xlabel("Component 1", fontsize=11)
    axes[1].set_ylabel("Component 2", fontsize=11)
    axes[1].set_title("True Labels", fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label="Subject ID")
    
    full_title = title
    if algorithm_name:
        full_title = f"{algorithm_name}: {title}"
    plt.suptitle(full_title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_summary_table(results_dict, n_classes, save_path=None, algorithm_name=""):
    """
    Create a summary table comparing methods at k = n_classes.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method name -> results DataFrame
    n_classes : int
        True number of classes
    save_path : str, optional
        Path to save the figure
    algorithm_name : str, optional
        Name of clustering algorithm (e.g., "K-Means", "Agglomerative")
    """
    metrics = ["silhouette", "purity", "nmi", "ari"]
    
    # Extract results at k = n_classes
    summary_data = []
    for method_name, results in results_dict.items():
        row = results[results["k"] == n_classes]
        if not row.empty:
            summary_data.append({
                "Method": method_name,
                "Silhouette": row["silhouette"].values[0],
                "Purity": row["purity"].values[0],
                "NMI": row["nmi"].values[0],
                "ARI": row["ari"].values[0],
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create figure with table
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=summary_df.round(4).values,
        colLabels=summary_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['lightsteelblue'] * len(summary_df.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    table_title = f"Clustering Results at k={n_classes} (True Number of Classes)"
    if algorithm_name:
        table_title = f"{algorithm_name}: {table_title}"
    plt.title(table_title, 
              fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return summary_df


def plot_dimred_comparison(pca_metrics, umap_metrics, ae_metrics=None, save_path=None, algorithm_name=""):
    """
    Plot comparison of dimensionality reduction quality metrics.
    
    Parameters
    ----------
    pca_metrics : dict
        Metrics from PCA dimensionality reduction
    umap_metrics : dict
        Metrics from UMAP dimensionality reduction
    ae_metrics : dict, optional
        Metrics from Autoencoder dimensionality reduction
    save_path : str, optional
        Path to save the figure
    algorithm_name : str, optional
        Name of clustering algorithm (e.g., "K-Means", "Agglomerative")
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Trustworthiness comparison (bar chart)
    if ae_metrics is not None:
        methods = ['PCA', 'UMAP', 'Autoencoder']
        trust_values = [pca_metrics['trustworthiness'], umap_metrics['trustworthiness'], ae_metrics['trustworthiness']]
        colors = ['steelblue', 'darkorange', 'green']
    else:
        methods = ['PCA', 'UMAP']
        trust_values = [pca_metrics['trustworthiness'], umap_metrics['trustworthiness']]
        colors = ['steelblue', 'darkorange']
    
    bars = axes[0].bar(methods, trust_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Trustworthiness Score', fontsize=12)
    axes[0].set_title('Trustworthiness\n(Local Neighborhood Preservation)', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    
    # Add value labels on bars
    for bar, val in zip(bars, trust_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                     f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[0].legend(loc='lower right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Summary metrics table
    axes[1].axis('off')
    
    if ae_metrics is not None:
        table_data = [
            ['Metric', 'PCA', 'UMAP', 'Autoencoder'],
            ['Trustworthiness', f"{pca_metrics['trustworthiness']:.4f}", f"{umap_metrics['trustworthiness']:.4f}", f"{ae_metrics['trustworthiness']:.4f}"],
            ['Reconstruction Error', f"{pca_metrics['reconstruction_error']:.4f}", 'N/A', f"{ae_metrics.get('reconstruction_error', 'N/A'):.4f}" if isinstance(ae_metrics.get('reconstruction_error'), (int, float)) else 'N/A'],
            ['Relative Recon Error', f"{pca_metrics['relative_recon_error']*100:.2f}%", 'N/A', f"{ae_metrics.get('relative_recon_error', 0)*100:.2f}%" if isinstance(ae_metrics.get('relative_recon_error'), (int, float)) else 'N/A'],
            ['Components', str(pca_metrics['n_components']), str(umap_metrics['n_components']), str(ae_metrics['n_components'])],
        ]
        col_colors = ['lightsteelblue'] * 4
    else:
        table_data = [
            ['Metric', 'PCA', 'UMAP'],
            ['Trustworthiness', f"{pca_metrics['trustworthiness']:.4f}", f"{umap_metrics['trustworthiness']:.4f}"],
            ['Reconstruction Error', f"{pca_metrics['reconstruction_error']:.4f}", 'N/A'],
            ['Relative Recon Error', f"{pca_metrics['relative_recon_error']*100:.2f}%", 'N/A'],
            ['Components', str(pca_metrics['n_components']), str(umap_metrics['n_components'])],
        ]
        col_colors = ['lightsteelblue'] * 3
    
    table = axes[1].table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        colColours=col_colors
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    axes[1].set_title('Dimensionality Reduction Metrics', fontsize=12, fontweight='bold', pad=20)
    
    dimred_title = 'Dimensionality Reduction Quality Comparison'
    if algorithm_name:
        dimred_title = f"{algorithm_name}: {dimred_title}"
    plt.suptitle(dimred_title, 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_cluster_images_comparison(X_original, labels_dict, y_true, n_clusters_to_show=5, 
                                    n_images_per_cluster=6, save_path=None):
    """
    Plot actual face images grouped by cluster assignments for multiple methods.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original flattened images (n_samples, n_features)
    labels_dict : dict
        Dictionary mapping method name -> cluster labels array
    y_true : np.ndarray
        True labels for reference
    n_clusters_to_show : int
        Number of clusters to display
    n_images_per_cluster : int
        Number of images to show per cluster
    save_path : str, optional
        Path to save the figure
    """
    n_methods = len(labels_dict)
    img_height, img_width = 112, 92
    
    fig, axes = plt.subplots(
        n_methods * n_clusters_to_show, n_images_per_cluster + 1,
        figsize=(2 * (n_images_per_cluster + 1), 2 * n_methods * n_clusters_to_show)
    )
    
    row_idx = 0
    for method_idx, (method_name, cluster_labels) in enumerate(labels_dict.items()):
        unique_clusters = np.unique(cluster_labels)[:n_clusters_to_show]
        
        for cluster_id in unique_clusters:
            # Get indices of images in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Get true labels for this cluster to show purity
            true_labels_in_cluster = y_true[cluster_mask]
            most_common_label = np.bincount(true_labels_in_cluster).argmax()
            purity = np.sum(true_labels_in_cluster == most_common_label) / len(true_labels_in_cluster)
            
            # Sample images from this cluster
            n_to_show = min(n_images_per_cluster, len(cluster_indices))
            sample_indices = np.random.choice(cluster_indices, n_to_show, replace=False)
            
            # First column: cluster info
            ax_info = axes[row_idx, 0]
            ax_info.axis('off')
            ax_info.text(0.5, 0.5, 
                        f"{method_name}\nCluster {cluster_id}\n({len(cluster_indices)} imgs)\nPurity: {purity:.0%}",
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        transform=ax_info.transAxes,
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Show images
            for img_idx, sample_idx in enumerate(sample_indices):
                ax = axes[row_idx, img_idx + 1]
                img = X_original[sample_idx].reshape(img_height, img_width)
                ax.imshow(img, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                # Show true label as small text
                ax.set_title(f"ID:{y_true[sample_idx]}", fontsize=7)
            
            # Hide unused subplots
            for img_idx in range(n_to_show, n_images_per_cluster):
                axes[row_idx, img_idx + 1].axis('off')
            
            row_idx += 1
    
    plt.suptitle("Cluster Contents Comparison: Actual Face Images by Cluster\n(ID = True Subject Label)", 
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
