"""
UMIST Dataset Data Loader Module
=================================

This module provides utilities for loading and preprocessing the UMIST facial 
recognition dataset. It handles loading the .mat file, extracting images and 
labels, and converting them to standard formats for use across the project.

Features:
- Load MATLAB .mat files
- Extract and flatten facial images
- Convert to standard Python formats (numpy arrays, pandas DataFrames)
- Intelligent caching for faster subsequent loads

Usage (with caching - recommended):
    from data_loader import load_umist_data
    
    # First run: loads from .mat, saves to cache
    # Subsequent runs: loads from cache (much faster!)
    faceimg, label, df = load_umist_data()

Usage (without caching):
    from data_loader import load_umist_data
    
    faceimg, label, df = load_umist_data(cache_dir=None)
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
import json

def load_umist_mat(dataset_path='umist_cropped.mat'):
    """
    Load the UMIST dataset from MATLAB .mat file.
    
    Parameters
    ----------
    dataset_path : str, optional
        Path to the umist_cropped.mat file. Default is relative path.
    
    Returns
    -------
    dict
        Raw MATLAB data containing 'facedat' and 'dirnames' keys
    
    Raises
    ------
    FileNotFoundError
        If the .mat file is not found at the specified path
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    mat_data = loadmat(dataset_path)
    return mat_data

def extract_and_flatten_images(mat_data):
    """
    Extract images and labels from MATLAB data structure and flatten them.
    
    The UMIST dataset structure: 'facedat' contains (1, 20) object array where
    each element is a (112, 92, ~28-48) array of face images for one subject.
    
    Parameters
    ----------
    mat_data : dict
        Raw MATLAB data loaded from .mat file
    
    Returns
    -------
    tuple
        - faceimg (np.ndarray): Shape (n_images, height*width), flattened images
        - label (np.ndarray): Shape (n_images,), subject IDs (0-19)
    
    Notes
    -----
    - Images are flattened from 2D (height, width) to 1D (height*width,)
    - Labels are integer indices from 0 to 19 (20 subjects total)
    - Subjects are not guaranteed to have equal number of images
    """
    facedat = mat_data['facedat']
    n_subjects = facedat.shape[1]
    
    faceimg_list = []
    label_list = []
    
    # Iterate through each subject and extract images
    for subject_idx in range(n_subjects):
        subject_images = facedat[0, subject_idx]  # Shape: (height, width, n_images)
        n_images_for_subject = subject_images.shape[2]
        
        # Iterate through each image for this subject
        for img_idx in range(n_images_for_subject):
            img = subject_images[:, :, img_idx]
            # Flatten: (112, 92) -> (10304,)
            faceimg_list.append(img.flatten())
            # Store label (subject index)
            label_list.append(subject_idx)
    
    # Convert to numpy arrays
    # faceimg: (n_images, 10304) - each row is a flattened image
    # label: (n_images,) - subject ID for each image
    faceimg = np.column_stack(faceimg_list).T
    label = np.array(label_list, dtype=np.int64)
    
    return faceimg, label

def create_dataframe(faceimg, label):
    """
    Convert flattened images and labels to Pandas DataFrame.
    
    Parameters
    ----------
    faceimg : np.ndarray
        Flattened images of shape (n_images, n_features)
    label : np.ndarray
        Subject labels of shape (n_images,)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with pixel_0 to pixel_N columns and subject_id column
    
    Notes
    -----
    - Pixel columns are named pixel_0, pixel_1, ..., pixel_N
    - subject_id column contains integer labels (0-19)
    """
    n_features = faceimg.shape[1]
    data_dict = {}
    
    # Add pixel features
    for pixel_idx in range(n_features):
        data_dict[f'pixel_{pixel_idx}'] = faceimg[:, pixel_idx]
    
    # Add labels
    data_dict['subject_id'] = label
    
    df = pd.DataFrame(data_dict)
    return df

def save_raw_data(cache_dir, faceimg, label, df):
    """
    Save raw loaded data to cache for faster subsequent loads.
    
    Parameters
    ----------
    cache_dir : str
        Directory to save cached raw data
    faceimg : np.ndarray
        Flattened images of shape (n_images, n_features)
    label : np.ndarray
        Subject labels of shape (n_images,)
    df : pd.DataFrame
        Complete DataFrame with pixel features and subject_id
    
    Notes
    -----
    Saves:
    - faceimg.npy: Flattened images
    - label.npy: Subject labels
    - dataframe.pkl: Complete DataFrame (includes all pixel columns)
    - raw_data_summary.json: Metadata about the raw data
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save arrays
    np.save(os.path.join(cache_dir, 'faceimg.npy'), faceimg)
    np.save(os.path.join(cache_dir, 'label.npy'), label)
    
    # Save DataFrame
    df.to_pickle(os.path.join(cache_dir, 'dataframe.pkl'))
    
    # Save metadata
    raw_summary = {
        'n_images': int(len(label)),
        'n_subjects': int(len(set(label))),
        'n_features': int(faceimg.shape[1]),
        'pixel_value_range': [float(faceimg.min()), float(faceimg.max())],
        'dataframe_shape': list(df.shape),
        'dataframe_columns': list(df.columns[:5]) + ['...'] + list(df.columns[-1:])
    }
    
    with open(os.path.join(cache_dir, 'raw_data_summary.json'), 'w') as f:
        json.dump(raw_summary, f, indent=2)

def load_raw_data_from_cache(cache_dir):
    """
    Load previously cached raw data.
    
    Parameters
    ----------
    cache_dir : str
        Directory containing cached raw data
    
    Returns
    -------
    tuple
        - faceimg (np.ndarray): Flattened images
        - label (np.ndarray): Subject labels
        - df (pd.DataFrame): Complete DataFrame
    
    Raises
    ------
    FileNotFoundError
        If cache files are missing
    """
    faceimg = np.load(os.path.join(cache_dir, 'faceimg.npy'))
    label = np.load(os.path.join(cache_dir, 'label.npy'))
    df = pd.read_pickle(os.path.join(cache_dir, 'dataframe.pkl'))
    
    return faceimg, label, df

def load_umist_data(dataset_path='umist_cropped.mat', cache_dir='processed_data/raw'):
    """
    Complete pipeline to load UMIST dataset with intelligent caching.
    
    This is the main entry point for loading the UMIST dataset. It handles:
    - Checking for cached data
    - Loading from cache if available (very fast!)
    - Loading from .mat file if cache missing
    - Saving to cache for next time
    
    Parameters
    ----------
    dataset_path : str, optional
        Path to umist_cropped.mat file. Only used if cache missing.
        Default: 'umist_cropped.mat'
    cache_dir : str, optional
        Directory to cache raw data. Set to None to disable caching.
        Default: 'processed_data/raw'
    
    Returns
    -------
    tuple
        - faceimg (np.ndarray): Flattened images, shape (n_images, height*width)
        - label (np.ndarray): Subject IDs, shape (n_images,)
        - df (pd.DataFrame): Complete dataset with pixel features and labels
    
    Notes
    -----
    Cache behavior:
    - First run: Loads .mat (slow), saves to cache
    - Subsequent runs: Loads from cache (fast! ~50x faster)
    - To regenerate: Delete cache_dir or set cache_dir=None
    
    Examples
    --------
    >>> # With caching (recommended)
    >>> faceimg, label, df = load_umist_data()
    
    >>> # Without caching
    >>> faceimg, label, df = load_umist_data(cache_dir=None)
    
    >>> # Custom cache location
    >>> faceimg, label, df = load_umist_data(cache_dir='my_cache')
    """
    # Check if cached data exists
    if cache_dir and os.path.exists(cache_dir):
        try:
            print(f"Loading raw data from cache: {cache_dir}")
            faceimg, label, df = load_raw_data_from_cache(cache_dir)
            return faceimg, label, df
        except Exception as e:
            print(f"Warning: Failed to load cache ({e}), loading from .mat file...")
    
    # Cache miss - load from .mat file
    print(f"Loading data from {dataset_path}...")
    
    # Load raw MATLAB data
    mat_data = load_umist_mat(dataset_path)
    
    # Extract and flatten images
    faceimg, label = extract_and_flatten_images(mat_data)
    
    # Create DataFrame
    df = create_dataframe(faceimg, label)
    
    # Save to cache if enabled
    if cache_dir:
        print(f"Saving raw data to cache: {cache_dir}")
        save_raw_data(cache_dir, faceimg, label, df)
        print(f"✓ Cache saved! Subsequent loads will be much faster.")
    
    return faceimg, label, df
