"""
Evaluation metrics for clustering and dimensionality reduction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.manifold import trustworthiness


def cluster_purity(y_true, y_pred):
    """
    Calculate cluster purity score.
    
    Purity measures how homogeneous each cluster is with respect to true labels.
    For each cluster, count the majority class and sum across all clusters.
    
    Parameters
    ----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted cluster assignments
        
    Returns
    -------
    float
        Purity score between 0 and 1 (higher is better)
    """
    contingency = pd.crosstab(y_pred, y_true)
    return contingency.max(axis=1).sum() / len(y_true)


def evaluate_clustering(X, y_true, y_pred):
    """
    Evaluate clustering performance using multiple metrics.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix used for clustering
    y_true : np.ndarray
        True class labels
    y_pred : np.ndarray
        Predicted cluster assignments
        
    Returns
    -------
    dict
        Dictionary containing all evaluation metrics
    """
    n_clusters = len(np.unique(y_pred))
    
    metrics = {
        "silhouette": silhouette_score(X, y_pred) if n_clusters > 1 else 0,
        "purity": cluster_purity(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
    }
    return metrics


def print_metrics(metrics, prefix=""):
    """
    Print evaluation metrics in a formatted way.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric name -> value
    prefix : str, optional
        Prefix to add before printing
    """
    if prefix:
        print(prefix)
    print(f"  Silhouette: {metrics['silhouette']:.4f}")
    print(f"  Purity:     {metrics['purity']:.4f}")
    print(f"  NMI:        {metrics['nmi']:.4f}")
    print(f"  ARI:        {metrics['ari']:.4f}")


# =============================================================================
# Dimensionality Reduction Quality Metrics
# =============================================================================

def compute_trustworthiness(X_original, X_reduced, n_neighbors=10):
    """
    Compute trustworthiness score for dimensionality reduction.
    
    Trustworthiness measures how well the local neighborhood structure
    is preserved after dimensionality reduction. A score of 1.0 means
    perfect preservation of local neighborhoods.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original high-dimensional data (n_samples, n_features_original)
    X_reduced : np.ndarray
        Reduced dimensionality data (n_samples, n_features_reduced)
    n_neighbors : int, optional
        Number of neighbors to consider. Default is 10.
        
    Returns
    -------
    float
        Trustworthiness score between 0 and 1 (higher is better)
    """
    return trustworthiness(X_original, X_reduced, n_neighbors=n_neighbors)


def compute_reconstruction_error(X_original, X_reduced, pca_model):
    """
    Compute reconstruction error for PCA.
    
    Measures how much information is lost during dimensionality reduction
    by reconstructing the original data and computing the mean squared error.
    
    Note: This only works for linear methods like PCA that support inverse_transform.
    UMAP is non-linear and non-invertible, so reconstruction error is not applicable.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original high-dimensional data (n_samples, n_features_original)
    X_reduced : np.ndarray
        PCA-reduced data (n_samples, n_components)
    pca_model : PCA
        Fitted PCA model with inverse_transform method
        
    Returns
    -------
    float
        Mean squared reconstruction error (lower is better)
    """
    X_reconstructed = pca_model.inverse_transform(X_reduced)
    mse = np.mean((X_original - X_reconstructed) ** 2)
    return mse


def compute_relative_reconstruction_error(X_original, X_reduced, pca_model):
    """
    Compute relative reconstruction error (normalized by original variance).
    
    This gives a percentage of variance lost during reconstruction.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original high-dimensional data
    X_reduced : np.ndarray
        PCA-reduced data
    pca_model : PCA
        Fitted PCA model
        
    Returns
    -------
    float
        Relative reconstruction error (0 = perfect, 1 = total loss)
    """
    X_reconstructed = pca_model.inverse_transform(X_reduced)
    mse = np.mean((X_original - X_reconstructed) ** 2)
    variance = np.var(X_original)
    return mse / variance if variance > 0 else 0


def evaluate_dimensionality_reduction(X_original, X_reduced, n_neighbors=10, 
                                       pca_model=None, method_name=""):
    """
    Evaluate dimensionality reduction quality.
    
    Parameters
    ----------
    X_original : np.ndarray
        Original high-dimensional data
    X_reduced : np.ndarray
        Reduced dimensionality data
    n_neighbors : int, optional
        Number of neighbors for trustworthiness. Default is 10.
    pca_model : PCA, optional
        Fitted PCA model (required for reconstruction error)
    method_name : str, optional
        Name of the method for display
        
    Returns
    -------
    dict
        Dictionary containing dimensionality reduction metrics
    """
    metrics = {
        "method": method_name,
        "n_components": X_reduced.shape[1],
        "trustworthiness": compute_trustworthiness(X_original, X_reduced, n_neighbors),
    }
    
    # Reconstruction error only for PCA (linear, invertible)
    if pca_model is not None:
        metrics["reconstruction_error"] = compute_reconstruction_error(
            X_original, X_reduced, pca_model
        )
        metrics["relative_recon_error"] = compute_relative_reconstruction_error(
            X_original, X_reduced, pca_model
        )
    else:
        metrics["reconstruction_error"] = None
        metrics["relative_recon_error"] = None
    
    return metrics


def print_dimred_metrics(metrics):
    """
    Print dimensionality reduction metrics in a formatted way.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of metric name -> value
    """
    print(f"\n  {metrics['method']} ({metrics['n_components']} components):")
    print(f"    Trustworthiness:      {metrics['trustworthiness']:.4f}")
    if metrics['reconstruction_error'] is not None:
        print(f"    Reconstruction Error: {metrics['reconstruction_error']:.4f}")
        print(f"    Relative Recon Error: {metrics['relative_recon_error']:.4f} ({metrics['relative_recon_error']*100:.2f}%)")
    else:
        print(f"    Reconstruction Error: N/A (non-linear method)")

