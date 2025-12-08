# UMIST Facial Recognition - Team Project

## Overview

This project provides shared preprocessing modules for the UMIST facial recognition dataset. All data loading and splitting is cached automatically for fast subsequent runs.

---

## Setup

### 1. Clone Repository

```bash
git clone git@github.com:stfny222/umist_classifier.git
cd umist_classifier
```

### 2. Download Dataset

The UMIST dataset (`umist_cropped.mat`) is not included in the repository (gitignored to save space).

**Place in repository root:**

```bash
umist_classifier/
├── umist_cropped.mat    # ← Download and add this file here
├── data_preprocessing/
├── README.md
├── .gitignore
└── ...
```

### 3. Install Dependencies (Optional)

If you don't already have them, install the required packages:

```bash
pip install numpy pandas scikit-learn scipy joblib kneed matplotlib seaborn tensorflow
```

### 4. Test Setup

```bash
python -c "from data_preprocessing import load_preprocessed_data; X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data(); print('✓ Setup complete!')"
```

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

## Project Architecture

```
umist_classifier/
├── umist_cropped.mat                 # Raw data (download separately)
├── data_preprocessing/               # Data loading & preprocessing
│   ├── __init__.py
│   ├── data_loader.py               # Load & cache raw .mat file
│   ├── data_splitter.py             # Split & normalize data
│   ├── data_augmentation.py         # Optional: augment training data
│   └── pipeline.py                  # High-level orchestration
│
├── dimensionality_reduction/         # 4 reduction methods
│   ├── pca.py                       # PCA (linear, reconstructible)
│   ├── umap_reduction.py            # UMAP (non-linear, visualizable)
│   ├── autoencoding.py              # Standard autoencoder
│   ├── autoencoding_improved.py     # Improved autoencoder with SSIM loss
│   └── compare_all_methods.py       # Comprehensive comparison
│
├── clustering/                       # 2 clustering pipelines
│   ├── shared/                      # Shared metrics & plots
│   │   ├── __init__.py
│   │   ├── metrics.py               # Silhouette, purity, NMI, ARI
│   │   └── plots.py                 # All visualization functions
│   ├── kmeans/                      # K-Means clustering
│   │   ├── main.py                  # K-Means pipeline (PCA + UMAP)
│   │   ├── clustering_utils.py      # Utility functions
│   │   └── outputs/                 # Results & plots
│   └── agglomerative/               # Hierarchical clustering
│       ├── main.py                  # Agglomerative pipeline (PCA + UMAP + Autoencoder)
│       ├── visualization/           # Dendrogram plotting
│       └── outputs/                 # Results & plots
│
├── classification/                   # 3 classification pipelines
│   ├── mlp.py                       # Multi-layer perceptron with K-Means features
│   ├── cnn.py                       # Convolutional neural network
│   ├── svm_lda.py                   # SVM + LDA classifiers
│   ├── predictions.py               # Ensemble predictions
│   └── outputs/                     # Results & plots
│
└── processed_data/                   # Cache (auto-generated, gitignored)
    ├── raw/                         # Cached raw data
    └── splits/                      # Cached train/val/test splits
```

---

## Data Flow

### 1. Core Data Pipeline (Shared by All Algorithms)

```
┌─────────────────────────────────┐
│  umist_cropped.mat              │  Raw MATLAB file (575 images, 20 subjects)
└──────────────┬──────────────────┘
               │ load_preprocessed_data()
               │ (auto-cached in processed_data/)
               ↓
    ┌────────────────────────────────────┐
    │ NORMALIZED & SPLIT DATA            │
    ├────────────────────────────────────┤
    │ X_train_norm (345, 10304) - 60%    │  StandardScaler fitted on train only
    │ X_val_norm (115, 10304) - 20%      │  (prevents data leakage)
    │ X_test_norm (115, 10304) - 20%     │
    │ y_train, y_val, y_test             │  Subject IDs (0-19, stratified)
    │ scaler (fitted on X_train only)    │
    └────────────┬────────────────────────┘
                 │
                 └─→ Feed to: Clustering, Classification, or Dimensionality Reduction
```

### 2. Dimensionality Reduction Methods (Unsupervised)

```
                ┌─────────────────────────────────────────┐
                │ Preprocessed Data (345-460, 10304)      │
                └──────────┬────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┬──────────────────┐
         │                 │                 │                  │
         ↓                 ↓                 ↓                  ↓
    ┌─────────┐      ┌──────────┐      ┌───────────┐     ┌──────────────┐
    │   PCA   │      │  UMAP    │      │Autoenc.  │     │Autoenc.Impr. │
    │(Linear) │      │(Non-lin) │      │(Non-lin) │     │ (SSIM loss)  │
    └────┬────┘      └────┬─────┘      └────┬─────┘     └────┬─────────┘
         │                │                 │                │
         │ Variance=0.95  │ n_comp=50      │ latent_dim   │ latent_dim
         │ →~90 features  │ (max)          │ ~90 features │ ~90 features
         │                │                │              │
         ↓                ↓                ↓              ↓
    ┌────────────────────────────────────────────────────────┐
    │ REDUCED FEATURES (345-460, 50-90)                      │
    │ Used for: Clustering & Classification                  │
    ├────────────────────────────────────────────────────────┤
    │ Metrics: Trustworthiness, Continuity, Reconstruction   │
    └────────────────────────────────────────────────────────┘
```

### 3. Clustering Pipelines

#### 3A: K-Means Clustering (PCA + UMAP)

```
         ┌─────────────────────────────────────┐
         │ Reduced Features (460, ~50-90)      │
         │ PCA: (460, 90)  | UMAP: (460, 50)  │
         └────────┬────────────────┬───────────┘
                  │                │
            ┌─────↓─────┐    ┌─────↓─────┐
            │ K-Means   │    │ K-Means   │
            │ (PCA)     │    │ (UMAP)    │
            └─────┬─────┘    └─────┬─────┘
                  │                │
         Test k=5,10,15,...,25 (ground truth + optimal)
         Metrics: Silhouette, Purity, NMI, ARI
                  │                │
            ┌─────↓─────────────────↓─────┐
            │ RESULTS: Best k & visualizations
            │ - Metric comparison plots
            │ - 2D clustering visualizations
            │ - Summary table & CSV exports
            └──────────────────────────────┘
```

#### 3B: Agglomerative Clustering (PCA + UMAP + Autoencoder)

```
         ┌──────────────────────────────────────────┐
         │ Reduced Features (460, ~50-90)           │
         │ PCA: (460,90) | UMAP: (460,50)          │
         │ Autoencoder: (460,90)                   │
         └────────┬─────────────┬────────────┬─────┘
                  │             │            │
            ┌─────↓─┐  ┌────────↓──┐  ┌─────↓──────┐
            │Agg.   │  │Agg.       │  │Agg.        │
            │(PCA)  │  │(UMAP)     │  │(Autoenc.)  │
            └─────┬─┘  └────────┬──┘  └─────┬──────┘
                  │             │            │
         Test k=5,10,15,...,25 (ground truth + optimal)
         Linkage: Ward (agglomerative)
         Metrics: Silhouette, Purity, NMI, ARI
                  │             │            │
            ┌─────↓─────────────↓────────────↓────┐
            │ RESULTS: Best k & visualizations
            │ - Dendrograms for each method
            │ - Metric comparison plots
            │ - 2D clustering visualizations
            │ - Cluster images comparison (actual faces)
            │ - Per-cluster purity analysis
            │ - Summary table & CSV exports
            └────────────────────────────────────┘
```

### 4. Classification Pipelines

#### 4A: MLP Classifier with K-Means Features

```
Preprocessed Data
      ↓
    PCA (95% variance)
      ↓
  K-Means Clustering
      ↓
Extract Distance Features to Cluster Centers
      ↓
Feed to MLP → Predictions → Accuracy/F1/Confusion Matrix
```

#### 4B: CNN Classifier

```
Preprocessed Data (reshaped to 112×92 images)
      ↓
Convolutional Layers + Pooling
      ↓
Fully Connected Layers
      ↓
Softmax Classification → Predictions → Accuracy/F1
```

#### 4C: SVM + LDA Classifiers

```
Preprocessed Data
      ↓
PCA Dimensionality Reduction (95% variance)
      ↓
      ├─→ SVM Classification
      └─→ LDA Classification
      ↓
Predictions → Accuracy/F1/Confusion Matrix
```

---

## Data Processing Steps

### Raw Data Loading (`data_preprocessing/data_loader.py`)

- **Input:** `umist_cropped.mat` (MATLAB format)
- **Output:** 
  - `faceimg` (575, 10304): Flattened images
  - `label` (575,): Subject IDs 0-19
  - `df` (575, 10305): DataFrame with all pixel columns
- **Caching:** Stored in `processed_data/raw/`

### Data Splitting & Normalization (`data_preprocessing/data_splitter.py`)

- **Stratified Split:** Maintains class balance across splits
  - Train: 60% (345 samples)
  - Val: 20% (115 samples)
  - Test: 20% (115 samples)
- **Normalization:** StandardScaler (zero mean, unit variance)
  - **Fitted ONLY on training data** → prevents data leakage
  - Applied to val & test using training statistics
- **Caching:** Stored in `processed_data/splits/`

### Data Augmentation (Optional, `data_preprocessing/data_augmentation.py`)

- **Transforms:** Random rotations (±10°), shifts (±10%), zoom (±10%), flips
- **Multiplier:** Configurable (default 5x: 345 → 1,725 training samples)
- **Usage:** For models needing more training data (CNN, MLP with small datasets)
- **Preserves:** Class labels automatically

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

## Data Augmentation (Optional - For Better Performance)

If your model needs more training data, you have two options:

### Option 1: One-Line Augmentation (Easiest)

```python
from data_preprocessing import load_preprocessed_data_with_augmentation

# Load data with 5x augmentation in one call (345 → 1725 training samples)
X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
    load_preprocessed_data_with_augmentation(augmentation_factor=5)

# Train directly
model.fit(X_train, y_train)
```

### Option 2: Manual Control (For Experimentation)

```python
from data_preprocessing import load_preprocessed_data
from data_preprocessing.data_augmentation import augment_training_data, visualize_augmentations

# Load data as usual
X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data()

# First, visualize to check quality
visualize_augmentations(X_train, num_samples=10)

# Generate 5x more training data (345 → 1725 samples)
X_train_aug, y_train_aug = augment_training_data(X_train, y_train, augmentation_factor=5)

# Train with augmented data
model.fit(X_train_aug, y_train_aug)
```

**What it does:**

- Applies random rotations (±10°), shifts (±10%), zoom (±10%), and flips
- Automatically preserves class labels
- Helps models generalize better with more diverse training samples

**Note:** Requires TensorFlow. Install with: `pip install tensorflow`

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
