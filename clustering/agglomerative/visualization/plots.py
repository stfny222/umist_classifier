"""
Visualization functions for agglomerative clustering results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score

sns.set_style("whitegrid")


def plot_dendrogram(X, y, n_clusters=None, title="Dendrogram", max_display=50, 
                    linkage_method='ward', distance_metric='euclidean', save_path=None):
    """
    Plot hierarchical clustering dendrogram with cut line and annotations.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        True labels (for labeling leaves)
    n_clusters : int, optional
        Number of clusters to highlight with cut line. If None, uses number of unique labels.
    title : str
        Plot title (method info will be appended)
    max_display : int
        Maximum number of samples to display (for readability)
    linkage_method : str
        Linkage method used ('ward', 'complete', 'average', 'single')
    distance_metric : str
        Distance metric used ('euclidean', 'cosine', etc.)
    save_path : str, optional
        Path to save the figure
    """
    # Subsample if too many points
    if len(X) > max_display:
        np.random.seed(42)
        indices = np.random.choice(len(X), max_display, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]
    else:
        X_sub = X
        y_sub = y
    
    # Determine number of clusters
    if n_clusters is None:
        n_clusters = len(np.unique(y))
    
    # Compute linkage matrix
    Z = linkage(X_sub, method=linkage_method)
    
    # Create figure with extra space for annotations
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create labels with "Subject X" format for clarity
    labels = [f"S{label}" for label in y_sub]
    
    # Plot dendrogram with colors
    dendro = dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
        color_threshold=0,  # Will be set after we find cut distance
    )
    
    # Find the cut distance for n_clusters
    # The cut distance is between the (n-k)th and (n-k+1)th merge distances
    # where n is number of samples and k is number of clusters
    n_samples = len(X_sub)
    if n_clusters < n_samples:
        # Get merge distances (4th column of Z)
        merge_distances = Z[:, 2]
        
        # To get k clusters, we need to cut after (n - k) merges
        # The cut should be between merge (n-k-1) and (n-k)
        cut_idx = n_samples - n_clusters
        if cut_idx > 0 and cut_idx <= len(merge_distances):
            # Cut between these two merge distances
            lower_dist = merge_distances[cut_idx - 1]
            upper_dist = merge_distances[cut_idx] if cut_idx < len(merge_distances) else merge_distances[-1] * 1.1
            cut_distance = (lower_dist + upper_dist) / 2
        else:
            cut_distance = merge_distances[-1] * 0.7  # Default fallback
        
        # Draw cut line
        ax.axhline(y=cut_distance, color='red', linestyle='--', linewidth=2, 
                   label=f'Cut for {n_clusters} clusters')
        
        # Add annotation for cut line
        ax.annotate(
            f'Cut → {n_clusters} clusters',
            xy=(ax.get_xlim()[1] * 0.02, cut_distance),
            xytext=(ax.get_xlim()[1] * 0.02, cut_distance + (ax.get_ylim()[1] * 0.05)),
            fontsize=10, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5)
        )
        
        # Calculate silhouette score at this cut
        cluster_labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        if len(np.unique(cluster_labels)) > 1:
            sil_score = silhouette_score(X_sub, cluster_labels)
            
            # Add silhouette score annotation
            ax.text(
                ax.get_xlim()[1] * 0.98, ax.get_ylim()[1] * 0.95,
                f'Silhouette @ k={n_clusters}: {sil_score:.3f}',
                fontsize=11, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )
    
    # Enhanced title with method information
    full_title = f"{title}\n(Linkage: {linkage_method.capitalize()}, Distance: {distance_metric.capitalize()})"
    ax.set_title(full_title, fontsize=14, fontweight='bold')
    
    # Better axis labels
    ax.set_xlabel("Samples (S = Subject ID)", fontsize=12)
    ax.set_ylabel(f"Distance ({distance_metric.capitalize()})", fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add info box with clear explanation
    n_samples_shown = len(X_sub)
    n_subjects_in_sample = len(np.unique(y_sub))
    n_total_subjects = len(np.unique(y)) if len(y) > len(y_sub) else n_subjects_in_sample
    
    info_text = (
        f"Displaying {n_samples_shown} of {len(X)} samples\n"
        f"Subjects in view: {n_subjects_in_sample} of {n_total_subjects}"
    )
    ax.text(
        0.02, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_metric_comparison(results_dict, save_path=None):
    """
    Plot comparison of clustering metrics across different methods.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping method name -> results DataFrame
        Each DataFrame should have columns: k, silhouette, purity, nmi, ari
    save_path : str, optional
        Path to save the figure
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
    
    plt.suptitle("Agglomerative Clustering: Method Comparison", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_clustering_2d(X, cluster_labels, y_true, title="Clustering Results", save_path=None):
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
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_summary_table(results_dict, n_classes, save_path=None):
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
    
    plt.title(f"Clustering Results at k={n_classes} (True Number of Classes)", 
              fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return summary_df


def plot_dimred_comparison(pca_metrics, umap_metrics, save_path=None):
    """
    Plot comparison of dimensionality reduction quality metrics.
    
    Parameters
    ----------
    pca_metrics : dict
        Metrics from PCA dimensionality reduction
    umap_metrics : dict
        Metrics from UMAP dimensionality reduction
    save_path : str, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Trustworthiness comparison (bar chart)
    methods = ['PCA', 'UMAP']
    trust_values = [pca_metrics['trustworthiness'], umap_metrics['trustworthiness']]
    colors = ['steelblue', 'darkorange']
    
    bars = axes[0].bar(methods, trust_values, color=colors, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Trustworthiness Score', fontsize=12)
    axes[0].set_title('Trustworthiness\n(Local Neighborhood Preservation)', fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    
    # Add value labels on bars
    for bar, val in zip(bars, trust_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                     f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    axes[0].legend(loc='lower right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Summary metrics table
    axes[1].axis('off')
    
    table_data = [
        ['Metric', 'PCA', 'UMAP'],
        ['Trustworthiness', f"{pca_metrics['trustworthiness']:.4f}", f"{umap_metrics['trustworthiness']:.4f}"],
        ['Reconstruction Error', f"{pca_metrics['reconstruction_error']:.4f}", 'N/A'],
        ['Relative Recon Error', f"{pca_metrics['relative_recon_error']*100:.2f}%", 'N/A'],
        ['Components', str(pca_metrics['n_components']), str(umap_metrics['n_components'])],
    ]
    
    table = axes[1].table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        colColours=['lightsteelblue'] * 3
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    axes[1].set_title('Dimensionality Reduction Metrics', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Dimensionality Reduction Quality Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()

