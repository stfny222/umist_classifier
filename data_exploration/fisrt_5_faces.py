"""
Display first 5 faces of each class in the augmented dataset.
Grid: 20 rows (subjects) x 5 columns (faces per subject)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_preprocessed_data_with_augmentation

# Constants
IMG_HEIGHT = 112
IMG_WIDTH = 92

def display_first_faces_per_class():
    """Display first 5 faces of each class in augmented dataset."""
    
    # Load augmented data
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "umist_cropped.mat"
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = (
        load_preprocessed_data_with_augmentation(dataset_path=path)
    )
    
    # Combine all data
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])
    
    print(f"Total samples: {len(y_all)}")
    print(f"Unique classes: {len(np.unique(y_all))}")
    
    # Get unique classes (should be 0-19)
    unique_classes = np.sort(np.unique(y_all))
    n_classes = len(unique_classes)
    n_faces = 5
    
    # Create figure: 20 rows x 5 columns
    fig, axes = plt.subplots(n_classes, n_faces, figsize=(10, 40))
    
    for row, class_id in enumerate(unique_classes):
        # Get indices for this class
        class_indices = np.where(y_all == class_id)[0]
        
        # Get first 5 faces
        for col in range(n_faces):
            if col < len(class_indices):
                idx = class_indices[col]
                img = X_all[idx].reshape(IMG_HEIGHT, IMG_WIDTH)
                axes[row, col].imshow(img, cmap='gray')
            else:
                axes[row, col].axis('off')
            
            axes[row, col].axis('off')
            
            # Add row label (subject ID) on first column
            if col == 0:
                axes[row, col].set_ylabel(f'Subject {class_id}', fontsize=10, rotation=0, 
                                          labelpad=50, va='center')
                axes[row, col].yaxis.set_visible(True)
    
    # Add column titles
    for col in range(n_faces):
        axes[0, col].set_title(f'Face {col + 1}', fontsize=10)
    
    plt.suptitle('First 5 Faces of Each Subject (Augmented Dataset)', 
                 fontsize=14, fontweight='bold', y=1.001)
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'first_5_faces_per_class.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: {os.path.join(output_dir, 'first_5_faces_per_class.png')}")


if __name__ == "__main__":
    display_first_faces_per_class()
