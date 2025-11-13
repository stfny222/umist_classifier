"""
UMIST Dataset Data Splitter Module
===================================

This module provides utilities for splitting the UMIST dataset into training,
validation, and test sets using stratified sampling, and for normalizing features.

Key Features:
- Stratified sampling ensures class balance across splits
- Proper train-test separation prevents data leakage
- Normalization fitted on training data only
- Automatic caching of preprocessed splits

Usage:
    from data_preprocessing import split_and_normalize_data
    from data_preprocessing import load_umist_data
    
    # Load data
    faceimg, label, _ = load_umist_data()
    
    # Split and normalize (with automatic caching)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
        split_and_normalize_data(faceimg, label)
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

def split_data_stratified(X, y, train_ratio=0.60, val_ratio=0.20, 
                          test_ratio=0.20, random_state=42):
    """
    Split dataset into training, validation, and test sets using stratified sampling.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Labels of shape (n_samples,)
    train_ratio : float, optional
        Proportion of data for training. Default: 0.60
    val_ratio : float, optional
        Proportion of data for validation. Default: 0.20
    test_ratio : float, optional
        Proportion of data for testing. Default: 0.20
    random_state : int, optional
        Random seed for reproducibility. Default: 42
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'X_train', 'X_val', 'X_test': Feature matrices
        - 'y_train', 'y_val', 'y_test': Label arrays
    
    Raises
    ------
    ValueError
        If train_ratio + val_ratio + test_ratio != 1.0
    
    Notes
    -----
    - Training (60%): Provides sufficient data to learn robust facial features
      across all 20 subjects with limited total samples (575 images)
    - Validation (20%): Adequate for hyperparameter tuning and early stopping
      without overfitting to specific validation examples
    - Test (20%): Provides reliable, independent evaluation on unseen data
      with enough samples for statistical significance
    """
    # Validate split ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )
    
    # First split: train (60%) and temp (40% - will be split 50-50 for val/test)
    # stratify=y ensures each class is represented proportionally in both subsets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=y
    )
    
    # Second split: split the temp set into validation and test (50-50)
    # This gives us val_ratio and test_ratio from the original dataset
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_test_ratio),
        random_state=random_state,
        stratify=y_temp
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
    }

def normalize_features(X_train, X_val, X_test, scaler=None):
    """
    Normalize features using StandardScaler.
    
    CRITICAL: Scaler is ALWAYS fitted on training data only.
    
    This prevents data leakage where information from validation/test data
    would influence the scaler, creating unrealistic performance estimates.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features of shape (n_train, n_features)
    X_val : np.ndarray
        Validation features of shape (n_val, n_features)
    X_test : np.ndarray
        Test features of shape (n_test, n_features)
    scaler : StandardScaler, optional
        Pre-fitted scaler. If None, a new scaler is created and fitted.
        Default: None
    
    Returns
    -------
    tuple
        - X_train_norm: Normalized training features
        - X_val_norm: Normalized validation features
        - X_test_norm: Normalized test features
        - scaler: The StandardScaler object (fitted on training data)
    """
    if scaler is None:
        scaler = StandardScaler()
    
    # Fit scaler ONLY on training data
    # This calculates: mean = (sum of values) / n, std = sqrt(var)
    X_train_norm = scaler.fit_transform(X_train)
    
    # Transform validation and test using the TRAINED scaler
    # Transformation: (X - training_mean) / training_std
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_val_norm, X_test_norm, scaler

def split_and_normalize_data(X, y, train_ratio=0.60, val_ratio=0.20,
                             test_ratio=0.20, random_state=42, 
                             cache_dir='processed_data/splits'):
    """
    Complete pipeline: split dataset and normalize features.
    
    This is the main entry point for preparing data for modeling.
    It handles both stratified splitting and proper normalization.
    
    CACHING: Automatically saves splits to disk on first run and loads from cache
    on subsequent runs (much faster). Delete cache_dir to regenerate.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Labels of shape (n_samples,)
    train_ratio : float, optional
        Proportion for training. Default: 0.60
    val_ratio : float, optional
        Proportion for validation. Default: 0.20
    test_ratio : float, optional
        Proportion for testing. Default: 0.20
    random_state : int, optional
        Random seed for reproducibility. Default: 42
    cache_dir : str, optional
        Directory to cache preprocessed splits. Set to None to disable caching.
        Default: 'processed_data/splits'
    
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
    - First run: Splits data, normalizes, saves to cache_dir
    - Subsequent runs: Loads from cache (much faster!)
    - To regenerate: Delete cache_dir or set cache_dir=None
    """
    # Check if cached splits exist
    if cache_dir and os.path.exists(cache_dir):
        try:
            print(f"Loading preprocessed data from cache: {cache_dir}")
            cached = load_splits(cache_dir)
            return (cached['X_train_norm'], cached['X_val_norm'], cached['X_test_norm'],
                    cached['y_train'], cached['y_val'], cached['y_test'],
                    cached['scaler'])
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), regenerating splits...")
    
    # Generate splits
    print(f"Generating new splits (this may take a moment)...")
    splits = split_data_stratified(X, y, train_ratio, val_ratio, test_ratio, random_state)
    
    # Normalize features (scaler fitted on training data only)
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
        splits['X_train'], splits['X_val'], splits['X_test']
    )
    
    # Save to cache if enabled
    if cache_dir:
        print(f"Saving splits to cache: {cache_dir}")
        save_splits(cache_dir, 
                   splits['X_train'], splits['X_val'], splits['X_test'],
                   X_train_norm, X_val_norm, X_test_norm,
                   splits['y_train'], splits['y_val'], splits['y_test'],
                   scaler, train_ratio, val_ratio, test_ratio)
        print(f"✓ Cache saved! Subsequent runs will be much faster.")
    
    return (X_train_norm, X_val_norm, X_test_norm,
            splits['y_train'], splits['y_val'], splits['y_test'],
            scaler)

def save_splits(split_dir, X_train, X_val, X_test, X_train_norm, X_val_norm, 
                X_test_norm, y_train, y_val, y_test, scaler, 
                train_ratio=0.60, val_ratio=0.20, test_ratio=0.20):
    """
    Save split data and scaler to disk for reproducibility.
    
    Parameters
    ----------
    split_dir : str
        Directory to save split files
    X_train, X_val, X_test : np.ndarray
        Original (non-normalized) feature matrices
    X_train_norm, X_val_norm, X_test_norm : np.ndarray
        Normalized feature matrices
    y_train, y_val, y_test : np.ndarray
        Label arrays
    scaler : StandardScaler
        Fitted scaler object
    train_ratio, val_ratio, test_ratio : float
        Split ratios for metadata
    
    Notes
    -----
    Saves:
    - X_train_norm.npy, X_val_norm.npy, X_test_norm.npy: Normalized features
    - X_train.npy, X_val.npy, X_test.npy: Original features
    - y_train.npy, y_val.npy, y_test.npy: Labels
    - scaler.pkl: Fitted StandardScaler for deployment
    - split_summary.json: Metadata about splits
    """
    os.makedirs(split_dir, exist_ok=True)
    
    # Save normalized data
    np.save(os.path.join(split_dir, 'X_train_norm.npy'), X_train_norm)
    np.save(os.path.join(split_dir, 'X_val_norm.npy'), X_val_norm)
    np.save(os.path.join(split_dir, 'X_test_norm.npy'), X_test_norm)
    
    # Save original data
    np.save(os.path.join(split_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(split_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(split_dir, 'X_test.npy'), X_test)
    
    # Save labels
    np.save(os.path.join(split_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(split_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(split_dir, 'y_test.npy'), y_test)
    
    # Save scaler
    scaler_path = os.path.join(split_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Save metadata
    split_summary = {
        'train_size': int(len(X_train)),
        'val_size': int(len(X_val)),
        'test_size': int(len(X_test)),
        'train_ratio': float(train_ratio),
        'val_ratio': float(val_ratio),
        'test_ratio': float(test_ratio),
        'n_features': int(X_train.shape[1]),
        'n_subjects': 20,
        'random_state': 42,
        'stratification': 'stratified',
        'normalization': 'StandardScaler',
        'normalization_fit_on': 'training_set_only',
        'total_samples': int(len(y_train) + len(y_val) + len(y_test))
    }
    
    summary_path = os.path.join(split_dir, 'split_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(split_summary, f, indent=2)

def load_splits(split_dir):
    """
    Load previously saved split data from disk.
    
    Parameters
    ----------
    split_dir : str
        Directory containing saved split files
    
    Returns
    -------
    dict
        Dictionary containing:
        - X_train_norm, X_val_norm, X_test_norm
        - X_train, X_val, X_test
        - y_train, y_val, y_test
        - scaler
        - split_summary (metadata)
    """
    splits = {}
    
    # Load normalized data
    splits['X_train_norm'] = np.load(os.path.join(split_dir, 'X_train_norm.npy'))
    splits['X_val_norm'] = np.load(os.path.join(split_dir, 'X_val_norm.npy'))
    splits['X_test_norm'] = np.load(os.path.join(split_dir, 'X_test_norm.npy'))
    
    # Load original data
    splits['X_train'] = np.load(os.path.join(split_dir, 'X_train.npy'))
    splits['X_val'] = np.load(os.path.join(split_dir, 'X_val.npy'))
    splits['X_test'] = np.load(os.path.join(split_dir, 'X_test.npy'))
    
    # Load labels
    splits['y_train'] = np.load(os.path.join(split_dir, 'y_train.npy'))
    splits['y_val'] = np.load(os.path.join(split_dir, 'y_val.npy'))
    splits['y_test'] = np.load(os.path.join(split_dir, 'y_test.npy'))
    
    # Load scaler
    scaler_path = os.path.join(split_dir, 'scaler.pkl')
    splits['scaler'] = joblib.load(scaler_path)
    
    # Load metadata
    summary_path = os.path.join(split_dir, 'split_summary.json')
    with open(summary_path, 'r') as f:
        splits['split_summary'] = json.load(f)
    
    return splits
