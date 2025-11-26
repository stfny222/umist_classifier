"""
Convolutional Autoencoder for Face Image Dimensionality Reduction
==================================================================

This autoencoder uses convolutional layers to learn a compressed representation
of face images, then reconstructs them. Compare reconstruction quality with PCA.
"""

import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_autoencoder(input_shape=(112, 92, 1), latent_dim=123):
    """
    Build a convolutional autoencoder.

    Parameters
    ----------
    input_shape : tuple
        Shape of input images (height, width, channels)
    latent_dim : int
        Size of bottleneck layer (compressed representation)

    Returns
    -------
    autoencoder : tf.keras.Model
        Full autoencoder model
    encoder : tf.keras.Model
        Encoder only (for extracting compressed features)
    """
    # Encoder
    input_img = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)  # 56x46

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)  # 28x23

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)  # 14x12

    shape_before_flattening = x.shape[1:]
    x = tf.keras.layers.Flatten()(x)

    encoded = tf.keras.layers.Dense(latent_dim, activation='relu', name='bottleneck')(x)

    # Decoder
    x = tf.keras.layers.Dense(np.prod(shape_before_flattening), activation='relu')(encoded)
    x = tf.keras.layers.Reshape(shape_before_flattening)(x)

    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)

    # Crop to exact output size (112x92) - handles the slight size mismatch
    x = tf.keras.layers.Cropping2D(cropping=((0, 0), (2, 2)))(x)  # 112x92

    # Output layer - linear activation for normalized data
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)

    # Models
    autoencoder = tf.keras.models.Model(inputs=input_img, outputs=decoded, name='autoencoder')
    encoder = tf.keras.models.Model(inputs=input_img, outputs=encoded, name='encoder')

    return autoencoder, encoder


def train_autoencoder(X_train, X_val, latent_dim=123, epochs=50, batch_size=32):
    """
    Train the autoencoder.

    Parameters
    ----------
    X_train : np.ndarray
        Training images, shape (n_samples, n_features) - flattened
    X_val : np.ndarray
        Validation images, shape (n_samples, n_features) - flattened
    latent_dim : int
        Bottleneck dimension (default: 123 to match PCA)
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size

    Returns
    -------
    autoencoder : trained model
    encoder : trained encoder
    history : training history
    """
    # Reshape from flattened to 2D images
    X_train_reshaped = X_train.reshape(-1, 112, 92, 1)
    X_val_reshaped = X_val.reshape(-1, 112, 92, 1)

    print(f"Training data shape: {X_train_reshaped.shape}")
    print(f"Validation data shape: {X_val_reshaped.shape}")

    # Build model
    autoencoder, encoder = build_autoencoder(input_shape=(112, 92, 1), latent_dim=latent_dim)

    # Compile
    autoencoder.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    print("\nAutoencoder Architecture:")
    autoencoder.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train
    print(f"\nTraining autoencoder with {latent_dim} latent dimensions...")
    history = autoencoder.fit(
        X_train_reshaped, X_train_reshaped,  # Input = output for autoencoder
        validation_data=(X_val_reshaped, X_val_reshaped),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return autoencoder, encoder, history


def reconstruct_images(autoencoder, X):
    """
    Reconstruct images using trained autoencoder.

    Parameters
    ----------
    autoencoder : tf.keras.Model
        Trained autoencoder
    X : np.ndarray
        Images to reconstruct, shape (n_samples, n_features) - flattened

    Returns
    -------
    X_reconstructed : np.ndarray
        Reconstructed images, shape (n_samples, n_features) - flattened
    """
    # Reshape to 2D images
    X_reshaped = X.reshape(-1, 112, 92, 1)

    # Reconstruct
    X_recon_reshaped = autoencoder.predict(X_reshaped, verbose=0)

    # Flatten back
    X_reconstructed = X_recon_reshaped.reshape(X.shape[0], -1)

    return X_reconstructed


if __name__ == "__main__":
    from data_preprocessing import load_preprocessed_data_with_augmentation
    from utils import display_images

    print("=" * 70)
    print("AUTOENCODER TRAINING AND RECONSTRUCTION")
    print("=" * 70)

    # Load data with 30/35/35 split
    cache_dir = 'processed_data_30_35_35'

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data_with_augmentation(
        dataset_path='umist_cropped.mat',
        cache_dir=cache_dir,
        augmentation_factor=7,
        train_ratio=0.30,
        val_ratio=0.35,
        test_ratio=0.35,
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"  Pixel value range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  Note: Negative values are from StandardScaler normalization (mean=0, std=1)")

    # Train autoencoder
    autoencoder, encoder, history = train_autoencoder(
        X_train, X_val,
        latent_dim=123,  # Match PCA component count
        epochs=50,
        batch_size=32
    )

    # Reconstruct test images
    print("\nReconstructing test images...")
    X_test_reconstructed = reconstruct_images(autoencoder, X_test)

    # Calculate reconstruction error
    mse = np.mean((X_test - X_test_reconstructed) ** 2)
    mae = np.mean(np.abs(X_test - X_test_reconstructed))

    print(f"\nReconstruction Error:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")

    # Visualize results
    print("\nSaving visualization...")
    display_images(
        X_test_reconstructed,
        n_samples=5,
        n_subjects=10,
        title="Autoencoder Reconstructed (123 latent dims)",
        save=True,
        save_path="results/autoencoder_reconstructed_123dims_30_35_35.png"
    )

    # Save model
    autoencoder.save('models/autoencoder_123dims.keras')
    encoder.save('models/encoder_123dims.keras')

    print("\n✓ Complete! Models saved to models/ directory")
    print("✓ Reconstruction image saved to results/ directory")
    print("\n💡 Compare with PCA reconstruction:")
    print("   - results/original_test_30_35_35.png")
    print("   - results/pca_reconstructed_123comp_30_35_35.png")
    print("   - results/autoencoder_reconstructed_123dims_30_35_35.png")


