"""
UMIST Dataset Data Augmentation Module
=======================================

This module provides data augmentation to artificially expand the training set size.
Use this to improve model performance by generating more training samples!

What it does:
- Applies random transformations to images (rotation, shifts, zoom, flip)
- Multiplies your training data (e.g., 345 samples → 1725 samples with 5x augmentation)
- Preserves class labels automatically
- Includes visualization tools to check augmentation quality

Quick Start:
    from data_preprocessing.data_augmentation import augment_training_data, visualize_augmentations

    # First, check quality with visualization
    visualize_augmentations(X_train, num_samples=10)

    # Then generate augmented data (345 → 1725 samples)
    X_train_aug, y_train_aug = augment_training_data(X_train, y_train, augmentation_factor=5)

    # Use augmented data for training
    model.fit(X_train_aug, y_train_aug)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_HEIGHT = 112
IMG_WIDTH = 92
IMG_SIZE = IMG_HEIGHT * IMG_WIDTH

sns.set_style("whitegrid")


def create_augmentation_generator():
    """
    Create a TensorFlow model that applies random augmentations to face images.

    Augmentations applied:
    - Random rotation: ±10 degrees
    - Random shifts: ±10% horizontal/vertical
    - Random zoom: ±10%
    - Random horizontal flip (mimics different viewing angles)

    Returns
    -------
    tf.keras.Sequential
        Model that applies random augmentations when training=True
    """
    data_gen = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.03, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, fill_mode="nearest"),
        tf.keras.layers.RandomFlip(mode="horizontal")
    ])

    return data_gen


def augment_training_data(
    X_train,
    y_train,
    augmentation_factor=5,
    batch_size=32,
    random_state=42
):
    """
    Generate augmented training data by applying random transformations.

    Parameters
    ----------
    X_train : np.ndarray
        Training images, shape (n_samples, n_features) - flattened
    y_train : np.ndarray
        Training labels, shape (n_samples,)
    augmentation_factor : int
        How many augmented versions to generate per original image (default: 5)
    batch_size : int
        Batch size for generation (default: 32)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    X_train_augmented : np.ndarray
        Augmented training set including originals, shape (n_samples * (factor+1), n_features)
    y_train_augmented : np.ndarray
        Corresponding labels, shape (n_samples * (factor+1),)

    Notes
    -----
    - Original images are included in the output
    - Images are reshaped to 2D for augmentation, then flattened back
    - Labels are properly replicated for augmented samples
    """
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    # Reshape flattened images back to 2D for augmentation
    X_train_2d = X_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH)

    # Add channel dimension (expects 4D: samples, height, width, channels)
    X_train_4d = X_train_2d[:, :, :, np.newaxis]

    # Create augmentation model
    augmentation_model = create_augmentation_generator()

    # Initialize lists to collect augmented data
    X_augmented_list = [X_train]  # Start with original data
    y_augmented_list = [y_train]

    print(f"Generating {augmentation_factor}x augmented data...")
    print(f"Original training samples: {n_samples}")

    # Generate augmented samples
    for aug_idx in range(augmentation_factor):
        print(f"  Generating augmentation batch {aug_idx + 1}/{augmentation_factor}...")

        X_aug_batch = []

        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch = X_train_4d[i:batch_end]

            # Apply augmentation using TensorFlow layers
            aug_batch = augmentation_model(batch, training=True).numpy()

            # Flatten back to 1D and add to list
            aug_batch_flat = aug_batch.reshape(-1, n_features)
            X_aug_batch.append(aug_batch_flat)

        X_aug_batch = np.vstack(X_aug_batch)
        X_augmented_list.append(X_aug_batch)
        y_augmented_list.append(y_train)  # Replicate labels

    # Combine all data
    X_train_augmented = np.vstack(X_augmented_list)
    y_train_augmented = np.hstack(y_augmented_list)

    print(f"✓ Augmentation complete!")
    print(f"  Final training samples: {X_train_augmented.shape[0]}")
    print(f"  Multiplication factor: {X_train_augmented.shape[0] / n_samples:.1f}x")

    return X_train_augmented, y_train_augmented


def visualize_augmentations(
    X_train,
    num_samples=10,
    num_augmentations=5,
    subjects_to_show=None,
    figsize=(15, 20)
):
    """
    Visualize original images alongside their augmented versions.

    Parameters
    ----------
    X_train : np.ndarray
        Training images, shape (n_samples, n_features) - flattened
    num_samples : int
        Number of original images to show (default: 10)
    num_augmentations : int
        Number of augmented versions to generate per image (default: 5)
    subjects_to_show : list, optional
        Specific subject IDs to visualize. If None, random samples are shown.
    figsize : tuple
        Figure size (default: (15, 20))

    Notes
    -----
    Creates a grid showing:
    - Column 0: Original image
    - Columns 1-N: Augmented versions
    """
    print("\n" + "=" * 70)
    print("DATA AUGMENTATION QUALITY VISUALIZATION")
    print("=" * 70)

    # Reshape to 2D images
    X_train_2d = X_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH)

    # Select random samples
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_train_2d), num_samples, replace=False)

    # Add channel dimension
    X_samples = X_train_2d[sample_indices, :, :, np.newaxis]

    # Create augmentation model
    augmentation_model = create_augmentation_generator()

    # Create figure
    fig, axes = plt.subplots(
        num_samples,
        num_augmentations + 1,
        figsize=figsize
    )

    print(f"\nDisplaying {num_samples} samples with {num_augmentations} augmentations each...")

    for row_idx in range(num_samples):
        # Show original image
        ax = axes[row_idx, 0]
        ax.imshow(X_samples[row_idx, :, :, 0], cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        if row_idx == 0:
            ax.set_title("Original", fontweight='bold', fontsize=12)

        ax.set_ylabel(f"Sample {sample_indices[row_idx]}", fontweight='bold')

        # Add green border to original
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2)

        # Generate and show augmented versions
        single_image = X_samples[row_idx:row_idx+1]

        for col_idx in range(1, num_augmentations + 1):
            # Apply augmentation using TensorFlow layers
            aug_img = augmentation_model(single_image, training=True).numpy()[0, :, :, 0]

            ax = axes[row_idx, col_idx]
            ax.imshow(aug_img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(f"Aug {col_idx}", fontweight='bold', fontsize=12)

    plt.suptitle(
        "Original vs Augmented Face Images - Quality Check",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    plt.show()

    print("✓ Visualization complete!")


def compute_augmentation_statistics(X_original, X_augmented):
    """
    Compute statistics to assess augmentation quality.

    Parameters
    ----------
    X_original : np.ndarray
        Original images, shape (n_samples, n_features)
    X_augmented : np.ndarray
        Augmented images (including originals), shape (n_samples_aug, n_features)

    Returns
    -------
    dict
        Statistics including:
        - Dataset sizes (original vs augmented)
        - Pixel value ranges (check for clipping issues)
        - Mean/std statistics (verify distribution preserved)

    Notes
    -----
    Good augmentation should:
    - Maintain similar pixel ranges to original
    - Keep mean/std close to original values
    - Not introduce artifacts or unrealistic images
    """
    print("\n" + "=" * 70)
    print("AUGMENTATION QUALITY STATISTICS")
    print("=" * 70)

    n_original = X_original.shape[0]
    n_augmented = X_augmented.shape[0]

    # Basic statistics
    stats = {
        'n_original': n_original,
        'n_augmented': n_augmented,
        'augmentation_factor': n_augmented / n_original,
        'pixel_range_original': (X_original.min(), X_original.max()),
        'pixel_range_augmented': (X_augmented.min(), X_augmented.max()),
        'mean_original': X_original.mean(),
        'std_original': X_original.std(),
        'mean_augmented': X_augmented.mean(),
        'std_augmented': X_augmented.std(),
    }

    print(f"\nDataset Size:")
    print(f"  Original samples: {stats['n_original']}")
    print(f"  Augmented samples: {stats['n_augmented']}")
    print(f"  Multiplication factor: {stats['augmentation_factor']:.1f}x")

    print(f"\nPixel Value Ranges:")
    print(f"  Original: [{stats['pixel_range_original'][0]:.4f}, {stats['pixel_range_original'][1]:.4f}]")
    print(f"  Augmented: [{stats['pixel_range_augmented'][0]:.4f}, {stats['pixel_range_augmented'][1]:.4f}]")

    print(f"\nPixel Statistics:")
    print(f"  Original - Mean: {stats['mean_original']:.4f}, Std: {stats['std_original']:.4f}")
    print(f"  Augmented - Mean: {stats['mean_augmented']:.4f}, Std: {stats['std_augmented']:.4f}")

    # Check if pixel ranges look reasonable
    if stats['pixel_range_augmented'][0] < stats['pixel_range_original'][0] - 0.1 or \
       stats['pixel_range_augmented'][1] > stats['pixel_range_original'][1] + 0.1:
        print("\n  ⚠ Warning: Augmented pixel range differs significantly from original")
    else:
        print("\n  ✓ Pixel ranges look good!")

    print(f"\n✓ Statistics computed!")

    return stats


def test_augmentation_pipeline():
    """
    Test the complete augmentation pipeline with sample data.

    This function:
    1. Loads preprocessed training data
    2. Visualizes augmentation quality
    3. Generates augmented dataset
    4. Computes quality statistics
    5. Shows class distribution before/after
    """
    from data_preprocessing import load_preprocessed_data
    import pandas as pd

    print("\n" + "=" * 70)
    print("TESTING DATA AUGMENTATION PIPELINE")
    print("=" * 70)

    # Load data
    print("\n1. Loading preprocessed data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "umist_cropped.mat"
        )
    )

    print(f"   Training set: {X_train.shape}")

    # Visualize augmentations
    print("\n2. Visualizing augmentation quality...")
    visualize_augmentations(X_train, num_samples=10, num_augmentations=5)

    # Generate augmented data (smaller factor for testing)
    print("\n3. Generating augmented dataset...")
    X_train_aug, y_train_aug = augment_training_data(
        X_train, y_train, augmentation_factor=3
    )

    # Compute statistics
    print("\n4. Computing quality statistics...")
    stats = compute_augmentation_statistics(X_train, X_train_aug)

    # Show class distribution
    print("\n5. Class distribution analysis...")
    print("\nOriginal distribution:")
    original_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"   Samples per class: {original_counts.values}")
    print(f"   Mean: {original_counts.mean():.1f}, Std: {original_counts.std():.1f}")

    print("\nAugmented distribution:")
    aug_counts = pd.Series(y_train_aug).value_counts().sort_index()
    print(f"   Samples per class: {aug_counts.values}")
    print(f"   Mean: {aug_counts.mean():.1f}, Std: {aug_counts.std():.1f}")

    # Verify class balance is maintained
    ratio = aug_counts / original_counts
    print(f"\nAugmentation ratio per class:")
    print(f"   Min: {ratio.min():.2f}x, Max: {ratio.max():.2f}x, Mean: {ratio.mean():.2f}x")

    if ratio.std() < 0.01:
        print("   ✓ Class balance perfectly maintained!")
    else:
        print(f"   ⚠ Minor class imbalance detected (std: {ratio.std():.3f})")

    print("\n" + "=" * 70)
    print("AUGMENTATION PIPELINE TEST COMPLETE!")
    print("=" * 70)

    return X_train_aug, y_train_aug, stats


if __name__ == "__main__":
    # Run test pipeline
    X_train_aug, y_train_aug, stats = test_augmentation_pipeline()
