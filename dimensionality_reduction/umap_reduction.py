"""
Dimensionality Reduction via UMAP
==================================

This module provides UMAP (Uniform Manifold Approximation and Projection)
dimensionality reduction for the UMIST facial recognition dataset.

UMAP is a non-linear dimensionality reduction technique that:
- Preserves both local and global structure
- Is faster than t-SNE for large datasets
- Can be used for visualization (2-3D) or as preprocessing for clustering

Key Parameters:
- n_neighbors: Controls local vs global structure preservation
- min_dist: Controls how tightly points are packed
- n_components: Target dimensionality
- metric: Distance metric (default: euclidean)

Usage:
------
    from dimensionality_reduction.umap_reduction import fit_and_transform_umap

    X_train_umap, X_val_umap, X_test_umap, umap_model = fit_and_transform_umap(
        X_train, X_val, X_test, n_components=50
    )

Dependencies:
    pip install umap-learn

Notes:
------
- UMAP is fit ONLY on training data to avoid leakage.
- Validation and test sets are transformed using the fitted model.
- For clustering, use higher n_components (e.g., 50); for visualization, use 2-3.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import umap
# Import pipeline loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing.pipeline import load_preprocessed_data_with_augmentation

sns.set_style("whitegrid")


def fit_and_transform_umap(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 50,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, umap.UMAP]:
    """
    Fit UMAP on training set and transform all splits.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (n_samples, n_features).
    X_val : np.ndarray
        Validation feature matrix.
    X_test : np.ndarray
        Test feature matrix.
    n_components : int, optional
        Target dimensionality. Default is 50 for clustering.
    n_neighbors : int, optional
        Number of neighbors for local structure. Default is 15.
        - Lower values focus on local structure
        - Higher values capture more global structure
    min_dist : float, optional
        Minimum distance between points. Default is 0.1.
        - Lower values create tighter clusters
        - Higher values spread points more evenly
    metric : str, optional
        Distance metric. Default is "euclidean".
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    X_train_umap : np.ndarray
        Transformed training features.
    X_val_umap : np.ndarray
        Transformed validation features.
    X_test_umap : np.ndarray
        Transformed test features.
    umap_model : umap.UMAP
        Fitted UMAP model.
    """
    if verbose:
        print(f"\nFitting UMAP with {n_components} components...")
        print(f"  n_neighbors: {n_neighbors}")
        print(f"  min_dist: {min_dist}")
        print(f"  metric: {metric}")

    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=verbose,
    )

    X_train_umap = umap_model.fit_transform(X_train)
    X_val_umap = umap_model.transform(X_val)
    X_test_umap = umap_model.transform(X_test)

    if verbose:
        print(
            f"Shapes after UMAP -> Train: {X_train_umap.shape} | "
            f"Val: {X_val_umap.shape} | Test: {X_test_umap.shape}"
        )

    return X_train_umap, X_val_umap, X_test_umap, umap_model


def plot_umap_embedding(
    X_umap: np.ndarray,
    y: np.ndarray,
    title: str = "UMAP Embedding",
    save_path: Optional[str] = None,
):
    """
    Plot 2D UMAP embedding colored by labels.

    Parameters
    ----------
    X_umap : np.ndarray
        UMAP-transformed data (must have at least 2 components).
    y : np.ndarray
        Labels for coloring.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save the figure.
    """
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        X_umap[:, 0], X_umap[:, 1],
        c=y, cmap='tab20', alpha=0.7, s=30
    )
    plt.colorbar(scatter, label="Subject ID")
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def explore_umap_parameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors_range: list = [5, 15, 30, 50],
    min_dist_range: list = [0.0, 0.1, 0.25, 0.5],
    n_components: int = 2,
    save_path: Optional[str] = None,
):
    """
    Explore different UMAP parameter combinations visually.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    n_neighbors_range : list
        List of n_neighbors values to try.
    min_dist_range : list
        List of min_dist values to try.
    n_components : int
        Number of components (should be 2 for visualization).
    save_path : str, optional
        Path to save the figure.
    """
    n_rows = len(n_neighbors_range)
    n_cols = len(min_dist_range)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    for i, n_neighbors in enumerate(n_neighbors_range):
        for j, min_dist in enumerate(min_dist_range):
            print(f"Fitting UMAP: n_neighbors={n_neighbors}, min_dist={min_dist}")
            
            umap_model = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42,
                verbose=False,
            )
            X_umap = umap_model.fit_transform(X_train)

            ax = axes[i, j] if n_rows > 1 else axes[j]
            scatter = ax.scatter(
                X_umap[:, 0], X_umap[:, 1],
                c=y_train, cmap='tab20', alpha=0.7, s=15
            )
            ax.set_title(f"n_neighbors={n_neighbors}\nmin_dist={min_dist}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("UMAP Parameter Exploration", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_trustworthiness_vs_components(
    X_train: np.ndarray,
    component_range: list = [2, 5, 10, 20, 30, 50, 75, 100],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    save_path: Optional[str] = None,
):
    """
    Plot trustworthiness score vs number of UMAP components.
    
    This is analogous to PCA's explained variance curve - it shows
    how well local structure is preserved as we increase dimensionality.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    component_range : list
        List of n_components values to test.
    n_neighbors : int
        Number of neighbors for UMAP.
    min_dist : float
        Minimum distance for UMAP.
    save_path : str, optional
        Path to save the figure.
    """
    from sklearn.manifold import trustworthiness
    
    trustworthiness_scores = []
    
    print("Computing trustworthiness for different component counts...")
    for n_comp in component_range:
        print(f"  n_components={n_comp}...", end=" ")
        
        umap_model = umap.UMAP(
            n_components=n_comp,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
            verbose=False,
        )
        X_umap = umap_model.fit_transform(X_train)
        
        # Compute trustworthiness
        trust = trustworthiness(X_train, X_umap, n_neighbors=10)
        trustworthiness_scores.append(trust)
        print(f"trustworthiness={trust:.4f}")
    
    # Find knee point (diminishing returns)
    from kneed import KneeLocator
    knee = KneeLocator(
        component_range, trustworthiness_scores,
        curve='concave', direction='increasing'
    )
    knee_point = knee.knee if knee.knee else component_range[-1]
    knee_value = trustworthiness_scores[component_range.index(knee_point)] if knee_point in component_range else trustworthiness_scores[-1]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Trustworthiness curve
    ax1 = axes[0]
    ax1.plot(component_range, trustworthiness_scores, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect (1.0)')
    ax1.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='Threshold (0.95)')
    
    # Mark knee point
    if knee_point:
        ax1.axvline(x=knee_point, color='red', linestyle='--', alpha=0.7, label=f'Knee: {knee_point}')
        ax1.scatter([knee_point], [knee_value], color='red', s=150, zorder=5, marker='o')
    
    ax1.set_xlabel('Number of Components', fontsize=12)
    ax1.set_ylabel('Trustworthiness Score', fontsize=12)
    ax1.set_title('UMAP Trustworthiness vs Components\n(Local Neighborhood Preservation)', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylim(0.9, 1.01)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Summary table
    ax2 = axes[1]
    ax2.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['Selected Components', f'{knee_point} / {X_train.shape[1]} ({knee_point/X_train.shape[1]*100:.2f}%)'],
        ['Trustworthiness at Knee', f'{knee_value:.4f} ({knee_value*100:.2f}%)'],
        ['Max Trustworthiness', f'{max(trustworthiness_scores):.4f}'],
        ['n_neighbors', str(n_neighbors)],
        ['min_dist', str(min_dist)],
    ]
    
    table = ax2.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='center',
        colColours=['lightcoral', 'lightcoral']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 1.8)
    
    ax2.set_title('UMAP Configuration Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('UMAP Dimensionality Selection', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return knee_point, trustworthiness_scores


def plot_distance_distribution(
    X_original: np.ndarray,
    X_umap: np.ndarray,
    n_samples: int = 1000,
    save_path: Optional[str] = None,
):
    """
    Plot distance distribution before and after UMAP transformation.
    
    This shows how UMAP reshapes the distance structure of the data.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original high-dimensional data.
    X_umap : np.ndarray
        UMAP-transformed data.
    n_samples : int
        Number of random distance pairs to sample.
    save_path : str, optional
        Path to save the figure.
    """
    from sklearn.metrics import pairwise_distances
    
    # Sample random pairs for efficiency
    n = X_original.shape[0]
    np.random.seed(42)
    idx1 = np.random.choice(n, n_samples, replace=True)
    idx2 = np.random.choice(n, n_samples, replace=True)
    
    # Compute pairwise distances
    dist_original = np.array([np.linalg.norm(X_original[i] - X_original[j]) 
                              for i, j in zip(idx1, idx2)])
    dist_umap = np.array([np.linalg.norm(X_umap[i] - X_umap[j]) 
                          for i, j in zip(idx1, idx2)])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Original distance distribution
    axes[0].hist(dist_original, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Pairwise Distance', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'Original Space\n({X_original.shape[1]} dims)', fontsize=12, fontweight='bold')
    axes[0].axvline(np.mean(dist_original), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(dist_original):.1f}')
    axes[0].legend()
    
    # Plot 2: UMAP distance distribution
    axes[1].hist(dist_umap, bins=50, color='darkorange', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Pairwise Distance', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'UMAP Space\n({X_umap.shape[1]} dims)', fontsize=12, fontweight='bold')
    axes[1].axvline(np.mean(dist_umap), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(dist_umap):.1f}')
    axes[1].legend()
    
    # Plot 3: Correlation between distances
    axes[2].scatter(dist_original, dist_umap, alpha=0.3, s=10, c='green')
    axes[2].set_xlabel('Original Distance', fontsize=11)
    axes[2].set_ylabel('UMAP Distance', fontsize=11)
    axes[2].set_title('Distance Correlation', fontsize=12, fontweight='bold')
    
    # Compute correlation
    corr = np.corrcoef(dist_original, dist_umap)[0, 1]
    axes[2].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[2].transAxes,
                 fontsize=11, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('UMAP Distance Structure Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_cluster_separation(
    X_umap: np.ndarray,
    y: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Plot inter-class vs intra-class distance distributions.
    
    Shows how well UMAP separates different classes.
    
    Parameters
    ----------
    X_umap : np.ndarray
        UMAP-transformed data.
    y : np.ndarray
        Class labels.
    save_path : str, optional
        Path to save the figure.
    """
    intra_distances = []
    inter_distances = []
    
    unique_classes = np.unique(y)
    
    # Sample distances for efficiency
    np.random.seed(42)
    n_samples_per_class = 50
    
    for cls in unique_classes:
        cls_mask = y == cls
        cls_indices = np.where(cls_mask)[0]
        other_indices = np.where(~cls_mask)[0]
        
        # Intra-class distances
        if len(cls_indices) > 1:
            sampled = np.random.choice(cls_indices, min(n_samples_per_class, len(cls_indices)), replace=False)
            for i in range(len(sampled)):
                for j in range(i+1, len(sampled)):
                    intra_distances.append(np.linalg.norm(X_umap[sampled[i]] - X_umap[sampled[j]]))
        
        # Inter-class distances
        if len(cls_indices) > 0 and len(other_indices) > 0:
            sampled_cls = np.random.choice(cls_indices, min(10, len(cls_indices)), replace=False)
            sampled_other = np.random.choice(other_indices, min(50, len(other_indices)), replace=False)
            for i in sampled_cls:
                for j in sampled_other:
                    inter_distances.append(np.linalg.norm(X_umap[i] - X_umap[j]))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Overlapping histograms
    axes[0].hist(intra_distances, bins=50, alpha=0.6, color='green', 
                 label=f'Intra-class (same person)', edgecolor='black')
    axes[0].hist(inter_distances, bins=50, alpha=0.6, color='red', 
                 label=f'Inter-class (different people)', edgecolor='black')
    axes[0].set_xlabel('Distance in UMAP Space', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Class Separation in UMAP Space', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].axvline(np.mean(intra_distances), color='darkgreen', linestyle='--', linewidth=2)
    axes[0].axvline(np.mean(inter_distances), color='darkred', linestyle='--', linewidth=2)
    
    # Plot 2: Summary statistics
    axes[1].axis('off')
    
    separation_ratio = np.mean(inter_distances) / np.mean(intra_distances)
    
    table_data = [
        ['Metric', 'Intra-class', 'Inter-class'],
        ['Mean Distance', f'{np.mean(intra_distances):.3f}', f'{np.mean(inter_distances):.3f}'],
        ['Std Distance', f'{np.std(intra_distances):.3f}', f'{np.std(inter_distances):.3f}'],
        ['Min Distance', f'{np.min(intra_distances):.3f}', f'{np.min(inter_distances):.3f}'],
        ['Max Distance', f'{np.max(intra_distances):.3f}', f'{np.max(inter_distances):.3f}'],
    ]
    
    table = axes[1].table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc='center',
        loc='upper center',
        colColours=['lightyellow', 'lightgreen', 'lightcoral']
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Add separation ratio
    axes[1].text(0.5, 0.25, f'Separation Ratio: {separation_ratio:.2f}x', 
                 transform=axes[1].transAxes, fontsize=14, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1].text(0.5, 0.1, '(Inter / Intra - higher is better)', 
                 transform=axes[1].transAxes, fontsize=10, ha='center', va='center')
    
    plt.suptitle('UMAP Class Separation Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    
    return separation_ratio


def main():
    """Main function to demonstrate UMAP dimensionality reduction."""
    
    # Create outputs directory
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "umist_cropped.mat"
    )
    
    print("=" * 70)
    print("UMAP DIMENSIONALITY REDUCTION")
    print("=" * 70)
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data_with_augmentation(
        dataset_path=path
    )
    print(f"Loaded data: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Fit UMAP for clustering (higher dimensionality)
    print("\n" + "-" * 70)
    print("UMAP for Clustering (50 components)")
    print("-" * 70)
    
    X_train_umap, X_val_umap, X_test_umap, umap_model = fit_and_transform_umap(
        X_train, X_val, X_test,
        n_components=50,
        n_neighbors=15,
        min_dist=0.1,
    )
    
    # Fit UMAP for visualization (2 components)
    print("\n" + "-" * 70)
    print("UMAP for Visualization (2 components)")
    print("-" * 70)
    
    X_train_umap_2d, _, _, _ = fit_and_transform_umap(
        X_train, X_val, X_test,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
    )
    
    # Plot 2D embedding
    plot_umap_embedding(
        X_train_umap_2d, y_train,
        title="UMAP 2D Embedding of UMIST Training Data",
        save_path=os.path.join(output_dir, "umap_2d_embedding.png")
    )
    
    # Plot trustworthiness vs components (analogous to PCA variance curve)
    print("\n" + "-" * 70)
    print("Trustworthiness vs Components Analysis")
    print("-" * 70)
    
    knee_point, trust_scores = plot_trustworthiness_vs_components(
        X_train,
        component_range=[2, 5, 10, 20, 30, 50, 75, 100],
        n_neighbors=15,
        min_dist=0.1,
        save_path=os.path.join(output_dir, "umap_trustworthiness_curve.png")
    )
    
    # Plot distance distribution
    print("\n" + "-" * 70)
    print("Distance Distribution Analysis")
    print("-" * 70)
    
    plot_distance_distribution(
        X_train, X_train_umap,
        n_samples=2000,
        save_path=os.path.join(output_dir, "umap_distance_distribution.png")
    )
    
    # Plot cluster separation
    print("\n" + "-" * 70)
    print("Class Separation Analysis")
    print("-" * 70)
    
    separation_ratio = plot_cluster_separation(
        X_train_umap, y_train,
        save_path=os.path.join(output_dir, "umap_class_separation.png")
    )
    
    # Parameter exploration (optional - takes longer)
    print("\n" + "-" * 70)
    print("Parameter Exploration")
    print("-" * 70)
    
    explore_umap_parameters(
        X_train, y_train,
        n_neighbors_range=[5, 15, 30],
        min_dist_range=[0.0, 0.1, 0.5],
        save_path=os.path.join(output_dir, "umap_parameter_exploration.png")
    )
    
    print("\n✓ UMAP dimensionality reduction complete!")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - umap_2d_embedding.png")
    print(f"  - umap_trustworthiness_curve.png")
    print(f"  - umap_distance_distribution.png")
    print(f"  - umap_class_separation.png")
    print(f"  - umap_parameter_exploration.png")
    
    return X_train_umap, X_val_umap, X_test_umap, umap_model


if __name__ == "__main__":
    main()
