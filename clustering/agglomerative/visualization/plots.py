"""
Visualization functions for agglomerative clustering - hierarchical specific.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score


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


