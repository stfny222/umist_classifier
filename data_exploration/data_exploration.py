"""
UMIST Dataset Exploration
==========================

This script explores the UMIST facial recognition dataset by:
1. Loading the full dataset as a DataFrame
2. Displaying dataset statistics, missing values, and class distribution
3. Visualizing class distribution in the full dataset
4. Loading the split datasets and visualizing class distribution across splits
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import data_preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocessing import load_umist_data, load_preprocessed_data, load_preprocessed_data_with_augmentation

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def explore_full_dataset():
    """
    Load and explore the full UMIST dataset.
    """

    # Load full dataset

    faceimg, label, df = load_umist_data(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "umist_cropped.mat",
        )
    )

    # Dataset statistics
    print("\n" + "-" * 70)
    print("DATASET STATISTICS")
    print("-" * 70)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"  Rows (images): {df.shape[0]}")
    print(f"  Columns (features + label): {df.shape[1]}")
    print(f"  Features: {faceimg.shape[1]}")
    print(f"  Subjects: {len(np.unique(label))}")

    print(f"\nPixel value statistics (first 5 pixels):")
    print(df.iloc[:, :5].describe())

    # Missing values
    print("\n" + "-" * 70)
    print("MISSING VALUES")
    print("-" * 70)
    missing_count = df.isnull().sum().sum()
    print(f"\nTotal missing values: {missing_count}")
    if missing_count == 0:
        print("No missing values detected - dataset is complete!")
    else:
        print("\nMissing values by column:")
        missing_by_col = df.isnull().sum()
        print(missing_by_col[missing_by_col > 0])

    # Class distribution
    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION")
    print("-" * 70)
    class_counts = df["subject_id"].value_counts().sort_index()
    print("\nImages per subject:")
    print(class_counts)

    print(f"\nDistribution statistics:")
    print(f"  Mean images per subject: {class_counts.mean():.2f}")
    print(f"  Std images per subject: {class_counts.std():.2f}")
    print(f"  Min images per subject: {class_counts.min()}")
    print(f"  Max images per subject: {class_counts.max()}")

    # Plot class distribution
    print("\n" + "-" * 70)
    print("VISUALIZATION: Class Distribution")
    print("-" * 70)

    plt.figure(figsize=(14, 6))

    # Bar chart
    bars = plt.bar(
        class_counts.index,
        class_counts.values,
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
    )

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add mean line
    mean_val = class_counts.mean()
    plt.axhline(
        y=mean_val,
        color="red",
        linestyle="--",
        label=f"Mean: {mean_val:.2f}",
        linewidth=2,
    )

    plt.xlabel("Subject ID", fontsize=12, fontweight="bold")
    plt.ylabel("Number of Images", fontsize=12, fontweight="bold")
    plt.title(
        "UMIST Dataset - Class Distribution (Full Dataset)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(class_counts.index)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return df, class_counts


def explore_split_datasets(augmented=False):
    """
    Load split datasets and visualize class distribution across splits.
    """
    print("\n\n" + "=" * 70)
    print("SPLIT DATASETS EXPLORATION")
    print("=" * 70)

    # Load preprocessed splits
    if augmented:
        print("\nLoading AUGMENTED preprocessed data splits...")
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data_with_augmentation(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "umist_cropped.mat",
            )
        )
    else:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "umist_cropped.mat",
            )
        )

    print("\nNumber of samples per split:")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    print(f"  Test samples: {len(y_test)}")
    print(f"  Total: {len(y_train) + len(y_val) + len(y_test)}")

    # Class distribution for each split
    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION PER SPLIT")
    print("-" * 70)

    train_counts = pd.Series(y_train).value_counts().sort_index()
    val_counts = pd.Series(y_val).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()

    print("\nTraining set:")
    print(train_counts)
    print(f"  Mean: {train_counts.mean():.2f}, Std: {train_counts.std():.2f}")

    print("\nValidation set:")
    print(val_counts)
    print(f"  Mean: {val_counts.mean():.2f}, Std: {val_counts.std():.2f}")

    print("\nTest set:")
    print(test_counts)
    print(f"  Mean: {test_counts.mean():.2f}, Std: {test_counts.std():.2f}")

    # Plot split distributions
    print("\n" + "-" * 70)
    print("VISUALIZATION: Class Distribution Across Splits")
    print("-" * 70)

    # Prepare data for grouped bar chart
    subjects = np.arange(20)

    # Ensure all subjects are represented (fill with 0 if missing)
    train_vals = [train_counts.get(i, 0) for i in subjects]
    val_vals = [val_counts.get(i, 0) for i in subjects]
    test_vals = [test_counts.get(i, 0) for i in subjects]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(subjects))
    width = 0.25

    bars1 = ax.bar(
        x - width,
        train_vals,
        width,
        label="Train (60%)",
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
    )
    bars2 = ax.bar(
        x,
        val_vals,
        width,
        label="Validation (20%)",
        color="orange",
        alpha=0.8,
        edgecolor="black",
    )
    bars3 = ax.bar(
        x + width,
        test_vals,
        width,
        label="Test (20%)",
        color="green",
        alpha=0.8,
        edgecolor="black",
    )

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel("Subject ID", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Images", fontsize=12, fontweight="bold")
    ax.set_title(
        "UMIST Dataset - Class Distribution Across Train/Val/Test Splits",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary statistics
    print("\n" + "-" * 70)
    print("SPLIT RATIO VERIFICATION")
    print("-" * 70)
    total = len(y_train) + len(y_val) + len(y_test)
    print(f"\nActual split ratios:")
    print(f"  Train: {len(y_train)/total*100:.2f}% (target: 60%)")
    print(f"  Validation: {len(y_val)/total*100:.2f}% (target: 20%)")
    print(f"  Test: {len(y_test)/total*100:.2f}% (target: 20%)")

    print(f"\nStratification verification:")
    print(f"  All 20 subjects represented in train: {len(train_counts) == 20}")
    print(f"  All 20 subjects represented in val: {len(val_counts) == 20}")
    print(f"  All 20 subjects represented in test: {len(test_counts) == 20}")


def display_sample_images(n_samples=5, n_subjects=10):
    """
    Display sample images from the UMIST dataset.

    Parameters
    ----------
    n_samples : int, optional
        Number of sample images per subject to display. Default is 5.
    n_subjects : int, optional
        Number of subjects to display. Default is 10.
    """
    print("\n\n" + "=" * 70)
    print("SAMPLE IMAGES DISPLAY")
    print("=" * 70)

    # Load full dataset
    faceimg, label, df = load_umist_data(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "umist_cropped.mat",
        )
    )

    # Reshape images back to original dimensions (112, 92)
    img_height, img_width = 112, 92
    total_subjects = len(np.unique(label))
    n_subjects = min(n_subjects, total_subjects)

    # Create figure with subplots - one row per subject, n_samples columns
    fig, axes = plt.subplots(
        n_subjects, n_samples, figsize=(n_samples * 2, n_subjects * 2.5)
    )

    # Ensure axes is 2D even if only one subject
    if n_subjects == 1:
        axes = axes.reshape(1, -1)

    print(f"\nDisplaying {n_samples} sample images per subject ({n_subjects} subjects)")

    # Iterate through each subject
    for subject_id in range(n_subjects):
        # Get all indices for this subject
        subject_indices = np.where(label == subject_id)[0]
        # Sample up to n_samples images from this subject
        sample_indices = np.random.choice(
            subject_indices, min(n_samples, len(subject_indices)), replace=False
        )

        # Display each sample
        for col, idx in enumerate(sample_indices):
            ax = axes[subject_id, col]

            # Reshape flattened image back to 2D
            img = faceimg[idx].reshape(img_height, img_width)

            # Display image in grayscale
            ax.imshow(img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])

            # Add title for first column
            if col == 0:
                ax.set_ylabel(f"Subject {subject_id}", fontweight="bold")

            # Add image index annotation
            ax.text(
                0.98,
                0.02,
                f"#{idx}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

    plt.suptitle(
        f"UMIST Dataset - Sample Face Images",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.show()

    print("✓ Sample images displayed successfully!")


if __name__ == "__main__":
    # Explore full dataset
    df, class_counts = explore_full_dataset()

    # Explore split datasets
    explore_split_datasets(augmented=False)

    # Display sample images
    display_sample_images(n_samples=5)
