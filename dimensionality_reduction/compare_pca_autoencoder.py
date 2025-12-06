"""
Compare PCA vs Autoencoder Reconstruction Quality
==================================================

This script trains both PCA and Autoencoder on the same data and compares
their reconstruction quality side-by-side.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from data_preprocessing import load_preprocessed_data_with_augmentation
from dimensionality_reduction.pca import determine_pca_components, fit_and_transform_pca, reconstruct_from_pca
from dimensionality_reduction.autoencoding import build_autoencoder, train_autoencoder, reconstruct_images
from utils import display_images


def compare_reconstructions(train_split=0.30, val_split=0.35, test_split=0.35, augmentation_factor=5):
    """
    Compare PCA vs Autoencoder reconstruction quality.

    Parameters
    ----------
    train_split, val_split, test_split : float
        Data split ratios
    latent_dim : int
        Dimensionality of compressed representation (same for both methods)
    """
    print("=" * 70)
    print("PCA vs AUTOENCODER RECONSTRUCTION COMPARISON")
    print("=" * 70)

    # Load data
    cache_dir = f'processed_data_{int(train_split*100)}_{int(val_split*100)}_{int(test_split*100)}'

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data_with_augmentation(
        dataset_path='umist_cropped.mat',
        cache_dir=cache_dir,
        augmentation_factor=augmentation_factor,
        train_ratio=train_split,
        val_ratio=val_split,
        test_ratio=test_split,
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")

    # ========================================================================
    # PCA RECONSTRUCTION
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: PCA RECONSTRUCTION")
    print("=" * 70)

    # Determine components (or use fixed latent_dim)
    n_components, pca_full, cum_var, var_ratio = determine_pca_components(
        X_train,
        variance_threshold=0.95,
        plot=False
    )

    print(f"\nAuto-selected {n_components} components")
    print(f"Explained variance: {cum_var[n_components-1]:.2%}")

    # Fit and transform
    X_train_pca, X_val_pca, X_test_pca, pca = fit_and_transform_pca(
        X_train, X_val, X_test, n_components
    )

    # Reconstruct
    print("\nReconstructing from PCA...")
    # Don't use scaler for reconstruction - keep in normalized space to match X_test
    X_test_pca_recon = reconstruct_from_pca(X_test_pca, pca, scaler=None)

    # Calculate error
    pca_mse = np.mean((X_test - X_test_pca_recon) ** 2)
    pca_mae = np.mean(np.abs(X_test - X_test_pca_recon))

    print(f"\nPCA Reconstruction Error:")
    print(f"  MSE: {pca_mse:.6f}")
    print(f"  MAE: {pca_mae:.6f}")

    # ========================================================================
    # AUTOENCODER RECONSTRUCTION
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: AUTOENCODER RECONSTRUCTION")
    print("=" * 70)

    # Train autoencoder
    autoencoder, encoder, history = train_autoencoder(
        X_train, X_val,
        latent_dim=n_components,
        epochs=50,
        batch_size=32
    )

    # Reconstruct
    print("\nReconstructing from Autoencoder...")
    X_test_auto_recon = reconstruct_images(autoencoder, X_test)

    # Calculate error
    auto_mse = np.mean((X_test - X_test_auto_recon) ** 2)
    auto_mae = np.mean(np.abs(X_test - X_test_auto_recon))

    print(f"\nAutoencoder Reconstruction Error:")
    print(f"  MSE: {auto_mse:.6f}")
    print(f"  MAE: {auto_mae:.6f}")

    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\nCompressed Dimensions: {n_components}/{X_train.shape[1]} ({n_components/X_train.shape[1]*100:.1f}%)")
    print(f"\nReconstruction Error (MSE):")
    print(f"  PCA:         {pca_mse:.6f}")
    print(f"  Autoencoder: {auto_mse:.6f}")
    print(f"  Difference:  {pca_mse - auto_mse:.6f} ({'PCA better' if pca_mse < auto_mse else 'Autoencoder better'})")

    print(f"\nReconstruction Error (MAE):")
    print(f"  PCA:         {pca_mae:.6f}")
    print(f"  Autoencoder: {auto_mae:.6f}")
    print(f"  Difference:  {pca_mae - auto_mae:.6f} ({'PCA better' if pca_mae < auto_mae else 'Autoencoder better'})")

    improvement_mse = ((pca_mse - auto_mse) / pca_mse * 100) if pca_mse > auto_mse else ((auto_mse - pca_mse) / auto_mse * 100)
    improvement_mae = ((pca_mae - auto_mae) / pca_mae * 100) if pca_mae > auto_mae else ((auto_mae - pca_mae) / auto_mae * 100)

    print(f"\nRelative Improvement:")
    if pca_mse < auto_mse:
        print(f"  PCA is {improvement_mse:.1f}% better in MSE")
    else:
        print(f"  Autoencoder is {improvement_mse:.1f}% better in MSE")

    if pca_mae < auto_mae:
        print(f"  PCA is {improvement_mae:.1f}% better in MAE")
    else:
        print(f"  Autoencoder is {improvement_mae:.1f}% better in MAE")

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING VISUALIZATIONS")
    print("=" * 70)

    # Original images
    print("\n1. Saving original test images...")
    display_images(
        X_test,
        n_samples=5,
        n_subjects=10,
        title="Original Test Images (Normalized)",
        save=True,
        save_path=f"results/comparison_original_{int(train_split*100)}_{int(val_split*100)}_{int(test_split*100)}.png"
    )

    # PCA reconstruction
    print("2. Saving PCA reconstructed images...")
    display_images(
        X_test_pca_recon,
        n_samples=5,
        n_subjects=10,
        title=f"PCA Reconstructed ({n_components} components, MSE: {pca_mse:.4f})",
        save=True,
        save_path=f"results/comparison_pca_{n_components}comp_{int(train_split*100)}_{int(val_split*100)}_{int(test_split*100)}.png"
    )

    # Autoencoder reconstruction
    print("3. Saving Autoencoder reconstructed images...")
    display_images(
        X_test_auto_recon,
        n_samples=5,
        n_subjects=10,
        title=f"Autoencoder Reconstructed ({n_components} dims, MSE: {auto_mse:.4f})",
        save=True,
        save_path=f"results/comparison_autoencoder_{n_components}dims_{int(train_split*100)}_{int(val_split*100)}_{int(test_split*100)}.png"
    )

    # Save models
    print("\n4. Saving models...")
    autoencoder.save(f'models/autoencoder_{n_components}dims.keras')
    encoder.save(f'models/encoder_{n_components}dims.keras')

    print("\n" + "=" * 70)
    print("✓ COMPARISON COMPLETE!")
    print("=" * 70)

    print(f"\n📊 Results saved to results/ directory:")
    print(f"   - comparison_original_{int(train_split*100)}_{int(val_split*100)}_{int(test_split*100)}.png")
    print(f"   - comparison_pca_{n_components}comp_{int(train_split*100)}_{int(val_split*100)}_{int(test_split*100)}.png")
    print(f"   - comparison_autoencoder_{n_components}dims_{int(train_split*100)}_{int(val_split*100)}_{int(test_split*100)}.png")

    print(f"\n💾 Models saved to models/ directory:")
    print(f"   - autoencoder_{n_components}dims.keras")
    print(f"   - encoder_{n_components}dims.keras")

    print(f"\n🎯 Open the three images side-by-side to visually compare quality!")

    return {
        'pca': {'mse': pca_mse, 'mae': pca_mae, 'reconstructed': X_test_pca_recon},
        'autoencoder': {'mse': auto_mse, 'mae': auto_mae, 'reconstructed': X_test_auto_recon},
        'original': X_test,
        'n_components': n_components
    }


if __name__ == "__main__":
    results = compare_reconstructions(
        train_split=0.50,
        val_split=0.25,
        test_split=0.25,
        augmentation_factor=5
    )

    print("\n" + "=" * 70)
    print("ANALYSIS TIPS")
    print("=" * 70)
    print("\n1. Look at the MSE/MAE numbers - lower is better")
    print("2. Visually inspect the reconstructions:")
    print("   - Are facial features preserved?")
    print("   - Is there blurring or artifacts?")
    print("   - Which method handles details better?")
    print("\n3. Consider trade-offs:")
    print("   - PCA: Fast, deterministic, linear")
    print("   - Autoencoder: Slower training, non-linear, potentially better")
