"""
Improved Convolutional Autoencoder with SSIM Loss
==================================================

This version adds:
- SSIM (Structural Similarity Index) for perceptual quality
- LeakyReLU activations throughout
- L2 regularization to prevent overfitting
- Dropout for better generalization
- Optimized training callbacks
"""

import tensorflow as tf
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# SSIM Metric for perceptual quality
def ssim_metric(y_true, y_pred):
    """
    Compute SSIM between true and predicted images.
    Higher is better (range 0-1).
    """
    # Get min/max from normalized data range
    data_range = 9.5 - (-4.9)  # Adjust based on your StandardScaler range
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=data_range))


def ssim_loss(y_true, y_pred):
    """
    SSIM loss function (1 - SSIM) so lower is better.
    """
    return 1.0 - ssim_metric(y_true, y_pred)


def combined_loss(y_true, y_pred, alpha=0.84):
    """
    Combined MSE + SSIM loss for balanced optimization.
    alpha controls the weight: alpha*SSIM + (1-alpha)*MSE
    """
    ssim_component = ssim_loss(y_true, y_pred)
    mse_component = tf.reduce_mean(tf.square(y_true - y_pred))
    return alpha * ssim_component + (1 - alpha) * mse_component


def build_autoencoder_improved(input_shape=(112, 92, 1), latent_dim=137, dropout_rate=0.2, l2_reg=1e-4):
    """
    Build an improved convolutional autoencoder with L2 regularization and LeakyReLU.

    Parameters
    ----------
    input_shape : tuple
        Shape of input images (height, width, channels)
    latent_dim : int
        Size of bottleneck layer
    dropout_rate : float
        Dropout rate for regularization
    l2_reg : float
        L2 regularization strength

    Returns
    -------
    autoencoder : tf.keras.Model
        Full autoencoder model
    encoder : tf.keras.Model
        Encoder only
    """
    from tensorflow.keras import regularizers

    input_img = tf.keras.Input(shape=input_shape)

    # --- Encoder ---
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(l2_reg))(input_img)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)  # 56x46
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)  # 28x23
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same',
                                kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)  # 14x12

    # Bottleneck
    shape_before_flattening = x.shape[1:]
    x = tf.keras.layers.Flatten()(x)
    encoded = tf.keras.layers.Dense(latent_dim, name='bottleneck')(x)  # Linear activation

    # --- Decoder ---
    x = tf.keras.layers.Dense(int(np.prod(shape_before_flattening)))(encoded)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Reshape(shape_before_flattening)(x)

    x = tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same',
                                        kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 28x24
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same',
                                        kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 56x48
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same',
                                        kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)  # 112x96

    # Crop to exact output size
    x = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 4)))(x)  # 112x92

    # Output layer
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='linear', padding='same')(x)

    # Models
    autoencoder = tf.keras.models.Model(inputs=input_img, outputs=decoded, name='autoencoder_ssim')
    encoder = tf.keras.models.Model(inputs=input_img, outputs=encoded, name='encoder_ssim')

    return autoencoder, encoder


def train_autoencoder_improved(X_train, X_val, latent_dim=137, epochs=150,
                               batch_size=32, learning_rate=0.001, dropout_rate=0.2,
                               l2_reg=1e-4, loss_type='combined'):
    """
    Train the improved autoencoder with SSIM-based loss.

    Parameters
    ----------
    X_train : np.ndarray
        Training images, shape (n_samples, n_features) - flattened
    X_val : np.ndarray
        Validation images, shape (n_samples, n_features) - flattened
    latent_dim : int
        Bottleneck dimension
    epochs : int
        Maximum number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Initial learning rate
    dropout_rate : float
        Dropout rate
    l2_reg : float
        L2 regularization strength
    loss_type : str
        'mse', 'ssim', or 'combined' (default)

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
    autoencoder, encoder = build_autoencoder_improved(
        input_shape=(112, 92, 1),
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg
    )

    # Select loss function
    if loss_type == 'mse':
        loss_fn = 'mse'
    elif loss_type == 'ssim':
        loss_fn = ssim_loss
    else:  # combined
        loss_fn = combined_loss

    # Compile with SSIM metric
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['mae', ssim_metric]
    )

    print(f"\nAutoencoder Architecture (SSIM-optimized, loss={loss_type}):")
    autoencoder.summary()

    # Create models directory
    os.makedirs('models', exist_ok=True)

    # Enhanced callbacks with longer patience for SSIM optimization
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/autoencoder_ssim_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train
    print(f"\nTraining SSIM-optimized autoencoder with {latent_dim} latent dimensions...")
    print(f"Configuration: lr={learning_rate}, dropout={dropout_rate}, l2_reg={l2_reg}, loss={loss_type}")

    history = autoencoder.fit(
        X_train_reshaped, X_train_reshaped,
        validation_data=(X_val_reshaped, X_val_reshaped),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return autoencoder, encoder, history


def reconstruct_images(autoencoder, X):
    """Reconstruct images using trained autoencoder."""
    X_reshaped = X.reshape(-1, 112, 92, 1)
    X_recon_reshaped = autoencoder.predict(X_reshaped, verbose=0)
    X_reconstructed = X_recon_reshaped.reshape(X.shape[0], -1)
    return X_reconstructed


if __name__ == "__main__":
    from data_preprocessing import load_preprocessed_data_with_augmentation
    from utils import display_images
    import matplotlib.pyplot as plt

    print("=" * 70)
    print("SSIM-OPTIMIZED AUTOENCODER TRAINING")
    print("=" * 70)

    # Load data with 50/25/25 split
    cache_dir = 'processed_data_50_25_25'

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data_with_augmentation(
        dataset_path='umist_cropped.mat',
        cache_dir=cache_dir,
        augmentation_factor=5,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
    )

    print(f"\nDataset loaded:")
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Train with SSIM-optimized hyperparameters
    autoencoder, encoder, history = train_autoencoder_improved(
        X_train, X_val,
        latent_dim=137,  # Match PCA components
        epochs=150,
        batch_size=32,
        learning_rate=0.001,
        dropout_rate=0.2,
        l2_reg=1e-4,
        loss_type='combined'  # Use combined MSE + SSIM loss
    )

    # Reconstruct test images
    print("\nReconstructing test images...")
    X_test_reconstructed = reconstruct_images(autoencoder, X_test)

    # Calculate errors
    mse = np.mean((X_test - X_test_reconstructed) ** 2)
    mae = np.mean(np.abs(X_test - X_test_reconstructed))

    # Calculate SSIM on test set
    X_test_reshaped = X_test.reshape(-1, 112, 92, 1).astype(np.float32)
    X_test_recon_reshaped = X_test_reconstructed.reshape(-1, 112, 92, 1).astype(np.float32)
    ssim_score = ssim_metric(
        tf.constant(X_test_reshaped, dtype=tf.float32),
        tf.constant(X_test_recon_reshaped, dtype=tf.float32)
    ).numpy()

    print(f"\nSSIM-Optimized Autoencoder Reconstruction:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  SSIM: {ssim_score:.6f} (higher is better, max=1.0)")

    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Plot training history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History - Combined Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training History - MAE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(history.history['ssim_metric'], label='Train SSIM')
    plt.plot(history.history['val_ssim_metric'], label='Val SSIM')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('Training History - SSIM (higher is better)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/autoencoder_ssim_training_history.png', dpi=150)
    print("✓ Saved training history plot")
    plt.show()

    # Visualize results
    print("\nSaving reconstructed images...")
    display_images(
        X_test_reconstructed,
        n_samples=5,
        n_subjects=10,
        title=f"SSIM-Optimized Autoencoder (137 dims, SSIM: {ssim_score:.4f})",
        save=True,
        save_path="results/autoencoder_ssim_reconstructed_137dims.png"
    )

    # Save model
    autoencoder.save('models/autoencoder_ssim_137dims.keras')
    encoder.save('models/encoder_ssim_137dims.keras')
    print("✓ Saved models")

    print("\n" + "=" * 70)
    print("COMPARISON WITH PCA")
    print("=" * 70)
    print(f"\n🎯 Target to beat:")
    print(f"   PCA MSE: 0.104712")
    print(f"\n📊 Your SSIM-Optimized Autoencoder:")
    print(f"   MSE:  {mse:.6f}")
    print(f"   SSIM: {ssim_score:.6f}")

    if mse < 0.104712:
        print(f"\n   🎉 SUCCESS! Autoencoder is better by {(0.104712-mse)/0.104712*100:.1f}%")
    else:
        improvement = (mse - 0.104712) / 0.104712 * 100
        print(f"\n   MSE is {improvement:.1f}% behind PCA")
        print(f"   BUT check visual quality - SSIM captures perceptual similarity better!")

    print("\n💡 Note: SSIM measures perceptual quality, which may be more important")
    print("   than pixel-level MSE for facial recognition tasks.")

