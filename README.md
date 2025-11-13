# UMIST Facial Recognition - Team Project

## Overview

This project provides shared preprocessing modules for the UMIST facial recognition dataset. All data loading and splitting is cached automatically for fast subsequent runs.

---

## Quick Start (Recommended)

```python
from data_preprocessing import load_preprocessed_data

# One simple call - handles everything with automatic caching
X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data()

# Train your model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print(f"Test Accuracy: {model.score(X_test, y_test):.3f}")
```

**First run:** Loads .mat file, processes, caches everything
**Subsequent runs:** Loads from cache

---

## Data Flow

```
┌─────────────────────────────────┐
│  umist_cropped.mat              │  Raw MATLAB file
└──────────────┬──────────────────┘
               │
               │ load_umist_data() - cached in processed_data/raw/
               ↓
    ┌────────────────────────────┐
    │ faceimg (575, 10304)       │  Flattened images
    │ label (575,)               │  Subject IDs (0-19)
    │ df (575, 10305)            │  DataFrame with pixels + labels
    └────────────┬───────────────┘
                 │
                 │ split_and_normalize_data() - cached in processed_data/splits/
                 ↓
    ┌─────────────────────────────────────────┐
    │ X_train_norm (345, 10304) - 60%         │
    │ X_val_norm (115, 10304) - 20%           │
    │ X_test_norm (115, 10304) - 20%          │
    │ y_train, y_val, y_test                  │
    │ scaler (fitted on training data only)   │
    └─────────────────────────────────────────┘
                 │
                 │ Your algorithm here
                 ↓
         [Dimensionality Reduction / Clustering / Classifier]
```

---

## Advanced Usage

### For Data Exploration on Full Dataset

```python
from data_preprocessing import load_umist_data

# Load raw data with caching - useful for visualization/exploration
faceimg, label, df = load_umist_data()

# df contains all pixel columns + subject_id
# Perfect for pandas operations, plotting, etc.
```

### Manual Control

```python
from data_preprocessing import load_umist_data, split_and_normalize_data

# Step 1: Load data (cached automatically)
faceimg, label, df = load_umist_data()

# Step 2: Split and normalize (cached automatically)
X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
    split_and_normalize_data(faceimg, label)
```

---

## What the Modules Do

### `data_preprocessing/data_loader.py`
- Loads MATLAB .mat file
- Extracts and flattens images (112×92 → 10304 features)
- Creates labels (subject IDs 0-19)
- **Caches raw data** for faster reloading

### `data_preprocessing/data_splitter.py`
- Splits data with stratification (maintains class balance)
- Normalizes using StandardScaler (fitted on training data only)
- Uses 60-20-20 train/val/test split
- Fixed random seed (42) for reproducibility
- **Caches splits** for instant reuse

### `data_preprocessing/pipeline.py`
- High-level orchestration of loading + splitting
- Multi-level caching for maximum performance
- **Use `load_preprocessed_data()` for most cases**

---

## Why This Approach?

**Single Source of Truth:** Everyone uses the same preprocessing → fair algorithm comparison

**No Data Leakage:** Scaler fitted only on training data → realistic performance estimates

**Stratified Splits:** Each subject appears in all splits proportionally → balanced evaluation

**Reproducible:** Fixed random seed → same results every time

**Fast:** Automatic caching at all levels → work efficiently

---

## Cache Management

Cache is stored in `processed_data/` (gitignored - won't be committed):
- `raw/` - Original loaded data
- `splits/` - Preprocessed train/val/test splits

**To regenerate cache:** Delete the `processed_data/` directory

---

## Notes

- Dataset: 575 images, 20 subjects, 112×92 pixels
- Split ratio: 60% train (345), 20% val (115), 20% test (115)
- Normalization: StandardScaler (mean=0, std=1 on training data)
- All preprocessing code includes detailed docstrings and comments
