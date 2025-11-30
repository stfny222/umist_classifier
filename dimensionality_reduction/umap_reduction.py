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

try:
    import umap
except ImportError:
    raise ImportError(
        "UMAP not installed. Install with: pip install umap-learn"
    )

# Import pipeline loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_preprocessed_data

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
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data(
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
    
    print("\n✓ UMAP dimensionality reduction complete!")
    
    return X_train_umap, X_val_umap, X_test_umap, umap_model


if __name__ == "__main__":
    main()
