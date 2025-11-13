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
"""

import os
from .data_loader import load_umist_data
from .data_splitter import split_and_normalize_data, load_splits

def load_preprocessed_data(dataset_path='umist_cropped.mat', 
                          cache_dir='processed_data',
                          train_ratio=0.60, val_ratio=0.20, test_ratio=0.20,
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
