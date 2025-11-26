from .data_loader import (
    load_umist_data,
    load_umist_mat,
    extract_and_flatten_images,
    create_dataframe,
    save_raw_data,
    load_raw_data_from_cache
)
from .data_splitter import (
    split_data_stratified,
    normalize_features,
    split_and_normalize_data,
    save_splits,
    load_splits
)
from .pipeline import load_preprocessed_data, load_preprocessed_data_with_augmentation

# Import augmentation functions (optional - requires TensorFlow)
try:
    from .data_augmentation import (
        create_augmentation_generator,
        augment_training_data,
        visualize_augmentations,
        compute_augmentation_statistics,
        test_augmentation_pipeline
    )
    _has_augmentation = True
except ImportError:
    _has_augmentation = False

__all__ = [
    # High-level pipeline functions (RECOMMENDED - use these!)
    'load_preprocessed_data',
    'load_preprocessed_data_with_augmentation',

    # Data loading (independent use for exploration)
    'load_umist_data',
    'load_umist_mat',
    'extract_and_flatten_images',
    'create_dataframe',
    'save_raw_data',
    'load_raw_data_from_cache',

    # Data splitting and normalization (independent use)
    'split_data_stratified',
    'normalize_features',
    'split_and_normalize_data',
    'save_splits',
    'load_splits',
]

# Add augmentation functions to __all__ if available
if _has_augmentation:
    __all__.extend([
        'create_augmentation_generator',
        'augment_training_data',
        'visualize_augmentations',
        'compute_augmentation_statistics',
        'test_augmentation_pipeline',
    ])

__version__ = '1.0.0'
