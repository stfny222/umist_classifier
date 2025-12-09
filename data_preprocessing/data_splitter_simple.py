"""Split dataset and normalize features."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

def split_data_stratified(X, y, train_ratio=0.60, val_ratio=0.20,
                          test_ratio=0.20, random_state=42):
    """Split dataset into train/val/test with stratification."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Train/temp split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=y
    )

    # Val/test split
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_test_ratio), random_state=random_state, stratify=y_temp
    )

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
    }

def normalize_features(X_train, X_val, X_test, scaler=None):
    """Normalize features using StandardScaler (fitted on training only)."""
    if scaler is None:
        scaler = StandardScaler()

    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)

    return X_train_norm, X_val_norm, X_test_norm, scaler

def split_and_normalize_data(X, y, train_ratio=0.60, val_ratio=0.20,
                             test_ratio=0.20, random_state=42,
                             cache_dir='processed_data/splits'):
    """Complete pipeline: split and normalize with automatic caching."""
    # Try cache first
    if cache_dir and os.path.exists(cache_dir):
        try:
            cached = load_splits(cache_dir)
            return (cached['X_train_norm'], cached['X_val_norm'], cached['X_test_norm'],
                    cached['y_train'], cached['y_val'], cached['y_test'], cached['scaler'])
        except:
            pass

    # Generate splits
    splits = split_data_stratified(X, y, train_ratio, val_ratio, test_ratio, random_state)

    # Normalize
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(
        splits['X_train'], splits['X_val'], splits['X_test']
    )

    # Save to cache
    if cache_dir:
        save_splits(cache_dir,
                   splits['X_train'], splits['X_val'], splits['X_test'],
                   X_train_norm, X_val_norm, X_test_norm,
                   splits['y_train'], splits['y_val'], splits['y_test'],
                   scaler, train_ratio, val_ratio, test_ratio)

    return (X_train_norm, X_val_norm, X_test_norm,
            splits['y_train'], splits['y_val'], splits['y_test'], scaler)

def save_splits(split_dir, X_train, X_val, X_test, X_train_norm, X_val_norm,
                X_test_norm, y_train, y_val, y_test, scaler,
                train_ratio=0.60, val_ratio=0.20, test_ratio=0.20):
    """Save splits to disk."""
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
    joblib.dump(scaler, os.path.join(split_dir, 'scaler.pkl'))

    # Save metadata
    split_summary = {
        'train_size': int(len(X_train)),
        'val_size': int(len(X_val)),
        'test_size': int(len(X_test)),
        'train_ratio': float(train_ratio),
        'val_ratio': float(val_ratio),
        'test_ratio': float(test_ratio),
        'n_features': int(X_train.shape[1]),
    }

    with open(os.path.join(split_dir, 'split_summary.json'), 'w') as f:
        json.dump(split_summary, f, indent=2)

def load_splits(split_dir):
    """Load saved splits."""
    splits = {}
    splits['X_train_norm'] = np.load(os.path.join(split_dir, 'X_train_norm.npy'))
    splits['X_val_norm'] = np.load(os.path.join(split_dir, 'X_val_norm.npy'))
    splits['X_test_norm'] = np.load(os.path.join(split_dir, 'X_test_norm.npy'))
    splits['X_train'] = np.load(os.path.join(split_dir, 'X_train.npy'))
    splits['X_val'] = np.load(os.path.join(split_dir, 'X_val.npy'))
    splits['X_test'] = np.load(os.path.join(split_dir, 'X_test.npy'))
    splits['y_train'] = np.load(os.path.join(split_dir, 'y_train.npy'))
    splits['y_val'] = np.load(os.path.join(split_dir, 'y_val.npy'))
    splits['y_test'] = np.load(os.path.join(split_dir, 'y_test.npy'))
    splits['scaler'] = joblib.load(os.path.join(split_dir, 'scaler.pkl'))

    with open(os.path.join(split_dir, 'split_summary.json'), 'r') as f:
        splits['split_summary'] = json.load(f)

    return splits
