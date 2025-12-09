"""
UMIST Dataset Complete Pipeline
================================

This module provides high-level convenience functions that orchestrate
both data loading and splitting with intelligent caching at all levels.

This is the recommended entry point for most use cases.

Usage:
    from data_preprocessing import load_preprocessed_data

    # One simple call - handles everything with caching
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
        load_preprocessed_data()

OPTIONAL - Data Augmentation:
    If you want more training samples, use data augmentation:

    from data_preprocessing.data_augmentation import augment_training_data

    # Multiply training data by 5x (345 → 1725 samples)
    X_train_aug, y_train_aug = augment_training_data(X_train, y_train, augmentation_factor=5)

    # Then train your model with augmented data
    model.fit(X_train_aug, y_train_aug)
"""

import os
from .data_loader import load_umist_data
from .data_splitter import split_and_normalize_data, load_splits


def load_preprocessed_data(dataset_path='umist_cropped.mat',
                          cache_dir='processed_data',
                          train_ratio=0.30, val_ratio=0.35, test_ratio=0.35,
                          random_state=42):
    """
    Master function: Load preprocessed data with intelligent multi-level caching.

    This is the recommended entry point for most use cases. It handles:
    - Checking for cached preprocessed splits (fastest path)
    - Checking for cached raw data (if splits missing)
    - Loading from .mat file only if no cache exists (slowest path)
    - Saving everything to cache at each level

    Cache Strategy:
    - Level 1: Check for preprocessed splits cache (processed_data/splits/)
    - Level 2: Check for raw data cache (processed_data/raw/)
    - Level 3: Load from .mat file and cache everything

    Parameters
    ----------
    dataset_path : str, optional
        Path to umist_cropped.mat file. Only used if cache missing.
        Default: 'umist_cropped.mat'
    cache_dir : str, optional
        Base directory for all caching. Set to None to disable caching.
        Default: 'processed_data'
    train_ratio, val_ratio, test_ratio : float, optional
        Split ratios. Default: 0.60, 0.20, 0.20
    random_state : int, optional
        Random seed for reproducibility. Default: 42

    Returns
    -------
    tuple
        - X_train_norm: Normalized training features
        - X_val_norm: Normalized validation features
        - X_test_norm: Normalized test features
        - y_train: Training labels
        - y_val: Validation labels
        - y_test: Test labels
        - scaler: Fitted StandardScaler object

    Notes
    -----
    Cache behavior:
    - First run: Loads .mat (slow), processes, saves both raw and splits
    - Subsequent runs: Loads splits from cache (very fast! ~100x faster)
    - If splits deleted but raw cached: Loads raw, reprocesses splits
    - To regenerate everything: Delete cache_dir or set cache_dir=None

    Examples
    --------
    >>> # Typical usage - simple and fast
    >>> X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data()

    >>> # Custom cache location
    >>> splits = load_preprocessed_data(cache_dir='my_cache')

    >>> # Disable caching (always load fresh)
    >>> splits = load_preprocessed_data(cache_dir=None)
    """
    # Level 1: Check if preprocessed splits exist in cache
    splits_dir = os.path.join(cache_dir, 'splits') if cache_dir else None

    if splits_dir and os.path.exists(splits_dir):
        try:
            print(f"✓ Loading preprocessed splits from cache: {splits_dir}")
            cached = load_splits(splits_dir)
            return (cached['X_train_norm'], cached['X_val_norm'], cached['X_test_norm'],
                    cached['y_train'], cached['y_val'], cached['y_test'],
                    cached['scaler'])
        except Exception as e:
            print(f"⚠️  Failed to load splits cache ({e}), will regenerate...")

    # Level 2 & 3: Load raw data (from cache or .mat) and process
    print(f"Processing data pipeline...")

    # load_umist_data() handles its own caching internally
    raw_cache_dir = os.path.join(cache_dir, 'raw') if cache_dir else None
    faceimg, label, df = load_umist_data(dataset_path, cache_dir=raw_cache_dir)

    # Split and normalize (also handles its own caching)
    print("Generating splits and normalizing...")
    X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, scaler = \
        split_and_normalize_data(faceimg, label, train_ratio, val_ratio,
                                test_ratio, random_state, cache_dir=splits_dir)

    print("✓ Pipeline complete! Data ready for use.")

    return X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test, scaler


def load_preprocessed_data_with_augmentation(
    dataset_path='umist_cropped.mat',
    cache_dir='processed_data',
    augmentation_factor=5,
    train_ratio=0.30,
    val_ratio=0.35,
    test_ratio=0.35,
    random_state=42
):
    """
    Convenience function: Load preprocessed data AND apply augmentation in one call.

    This is useful if you know you want augmented data. It combines:
    1. load_preprocessed_data() - loads and caches splits
    2. augment_training_data() - generates augmented training set

    Parameters
    ----------
    dataset_path : str, optional
        Path to umist_cropped.mat file
    cache_dir : str, optional
        Base directory for caching
    augmentation_factor : int, optional
        How many augmented versions per image (default: 5)
        - 0 or 1: No augmentation (returns original data)
        - 5: 5x augmentation (345 → 1725 training samples)
    train_ratio, val_ratio, test_ratio : float, optional
        Split ratios (default: 0.30, 0.35, 0.35)
    random_state : int, optional
        Random seed (default: 42)

    Returns
    -------
    tuple
        - X_train_aug: Augmented training features (includes originals)
        - X_val_norm: Validation features (not augmented)
        - X_test_norm: Test features (not augmented)
        - y_train_aug: Training labels (replicated for augmented samples)
        - y_val: Validation labels
        - y_test: Test labels
        - scaler: Fitted StandardScaler object

    Examples
    --------
    >>> # Load with 5x augmentation (345 → 1725 training samples)
    >>> X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
    ...     load_preprocessed_data_with_augmentation(augmentation_factor=5)

    >>> # No augmentation (same as load_preprocessed_data)
    >>> X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
    ...     load_preprocessed_data_with_augmentation(augmentation_factor=1)

    Notes
    -----
    - Only training data is augmented (val/test stay original for fair evaluation)
    - Augmentation is NOT cached (regenerated each time for randomness)
    - For more control, use load_preprocessed_data() + augment_training_data() separately
    """
    # Load base preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data(
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )

    # Apply augmentation if requested
    if augmentation_factor > 1:
        print(f"\nApplying {augmentation_factor}x data augmentation to training set...")
        try:
            from .data_augmentation import augment_training_data
            X_train_aug, y_train_aug = augment_training_data(
                X_train, y_train,
                augmentation_factor=augmentation_factor - 1,  # -1 because originals are included
                random_state=random_state
            )
            return X_train_aug, X_val, X_test, y_train_aug, y_val, y_test, scaler
        except ImportError:
            print("⚠️  Warning: data_augmentation module not available (requires TensorFlow)")
            print("   Returning original data without augmentation...")
            return X_train, X_val, X_test, y_train, y_val, y_test, scaler
    else:
        print("\nNo augmentation applied (augmentation_factor <= 1)")
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler
