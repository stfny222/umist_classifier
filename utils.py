"""Utility functions for visualization and image processing."""

import numpy as np
import matplotlib.pyplot as plt
import os


def display_images(X, n_samples=5, n_subjects=10, title="Images", save=False, save_path=None):
    """
    Display images in a grid, optionally save to file.

    Parameters
    ----------
    X : np.ndarray
        Images, shape (n_images, n_features) - flattened
    n_samples : int
        Number of images per subject/row to display
    n_subjects : int
        Number of subjects/rows to display
    title : str
        Figure title
    save : bool
        If True, saves the figure
    save_path : str, optional
        Where to save (e.g., 'results/original.png'). Required if save=True.
    """
    img_height, img_width = 112, 92

    # Calculate total images to display
    total_images = n_samples * n_subjects
    total_images = min(total_images, X.shape[0])

    # Random sample selection
    np.random.seed(42)
    sample_indices = np.random.choice(X.shape[0], total_images, replace=False)

    # Create grid (n_subjects rows, n_samples columns)
    fig, axes = plt.subplots(n_subjects, n_samples, figsize=(n_samples * 2, n_subjects * 1.8))

    # Handle edge cases for axes indexing
    if n_subjects == 1 and n_samples == 1:
        axes = np.array([[axes]])
    elif n_subjects == 1:
        axes = axes.reshape(1, -1)
    elif n_samples == 1:
        axes = axes.reshape(-1, 1)

    for idx, img_idx in enumerate(sample_indices):
        row = idx // n_samples
        col = idx % n_samples

        if row >= n_subjects:
            break

        ax = axes[row, col]
        img = X[img_idx].reshape(img_height, img_width)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        # Label first column with row/subject number
        if col == 0:
            ax.set_ylabel(f'Group {row}', fontsize=9, fontweight='bold')

    # Hide unused subplots
    for row in range(n_subjects):
        for col in range(n_samples):
            idx = row * n_samples + col
            if idx >= len(sample_indices):
                axes[row, col].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        if save_path is None:
            raise ValueError("save_path required when save=True")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to: {save_path}")
    else:
        plt.show()


__all__ = ['display_images']
