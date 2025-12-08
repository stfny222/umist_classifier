"""
Shared clustering utilities - evaluation and visualization.
"""

from .metrics import (
    cluster_purity,
    evaluate_clustering,
    print_metrics,
    compute_trustworthiness,
    compute_reconstruction_error,
    compute_relative_reconstruction_error,
    evaluate_dimensionality_reduction,
    print_dimred_metrics,
)

from .plots import (
    plot_metric_comparison,
    plot_clustering_2d,
    plot_summary_table,
    plot_dimred_comparison,
    plot_cluster_images_comparison,
)

__all__ = [
    # Metrics
    "cluster_purity",
    "evaluate_clustering",
    "print_metrics",
    "compute_trustworthiness",
    "compute_reconstruction_error",
    "compute_relative_reconstruction_error",
    "evaluate_dimensionality_reduction",
    "print_dimred_metrics",
    # Plots
    "plot_metric_comparison",
    "plot_clustering_2d",
    "plot_summary_table",
    "plot_dimred_comparison",
    "plot_cluster_images_comparison",
]
