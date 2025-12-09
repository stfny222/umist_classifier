"""Load UMIST dataset from .mat file."""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
import json

def load_umist_mat(dataset_path='umist_cropped.mat'):
    """Load UMIST .mat file."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return loadmat(dataset_path)

def extract_and_flatten_images(mat_data):
    """Extract and flatten images from MATLAB structure."""
    facedat = mat_data['facedat']
    n_subjects = facedat.shape[1]

    faceimg_list = []
    label_list = []

    for subject_idx in range(n_subjects):
        subject_images = facedat[0, subject_idx]
        n_images = subject_images.shape[2]

        for img_idx in range(n_images):
            img = subject_images[:, :, img_idx]
            faceimg_list.append(img.flatten())  # Flatten 112x92 -> 10304
            label_list.append(subject_idx)

    faceimg = np.column_stack(faceimg_list).T
    label = np.array(label_list, dtype=np.int64)

    return faceimg, label

def create_dataframe(faceimg, label):
    """Convert to DataFrame with pixel columns and subject_id."""
    n_features = faceimg.shape[1]
    data_dict = {}

    for pixel_idx in range(n_features):
        data_dict[f'pixel_{pixel_idx}'] = faceimg[:, pixel_idx]

    data_dict['subject_id'] = label
    return pd.DataFrame(data_dict)

def save_raw_data(cache_dir, faceimg, label, df):
    """Save raw data to cache."""
    os.makedirs(cache_dir, exist_ok=True)

    np.save(os.path.join(cache_dir, 'faceimg.npy'), faceimg)
    np.save(os.path.join(cache_dir, 'label.npy'), label)
    df.to_pickle(os.path.join(cache_dir, 'dataframe.pkl'))

    raw_summary = {
        'n_images': int(len(label)),
        'n_subjects': int(len(set(label))),
        'n_features': int(faceimg.shape[1]),
        'pixel_range': [float(faceimg.min()), float(faceimg.max())],
    }

    with open(os.path.join(cache_dir, 'raw_data_summary.json'), 'w') as f:
        json.dump(raw_summary, f, indent=2)

def load_raw_data_from_cache(cache_dir):
    """Load cached raw data."""
    faceimg = np.load(os.path.join(cache_dir, 'faceimg.npy'))
    label = np.load(os.path.join(cache_dir, 'label.npy'))
    df = pd.read_pickle(os.path.join(cache_dir, 'dataframe.pkl'))
    return faceimg, label, df

def load_umist_data(dataset_path='umist_cropped.mat', cache_dir='processed_data/raw'):
    """
    Load UMIST dataset with automatic caching.

    Returns: (faceimg, label, df)
        - faceimg: flattened images (n_images, 10304)
        - label: subject IDs (n_images,)
        - df: DataFrame with pixels + subject_id
    """
    # Try cache first
    if cache_dir and os.path.exists(cache_dir):
        try:
            faceimg, label, df = load_raw_data_from_cache(cache_dir)
            return faceimg, label, df
        except:
            pass

    # Load from .mat
    mat_data = load_umist_mat(dataset_path)
    faceimg, label = extract_and_flatten_images(mat_data)
    df = create_dataframe(faceimg, label)

    # Save to cache
    if cache_dir:
        save_raw_data(cache_dir, faceimg, label, df)

    return faceimg, label, df
