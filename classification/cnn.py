"""
CNN Classifier with Keras Tuner Hyperparameter Optimization
============================================================

Simple CNN classifier on unreduced, augmented UMIST face data.
Optimized for lower-end hardware with minimal hyperparameters.

Usage:
------
    python cnn.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import keras_tuner as kt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_preprocessed_data_with_augmentation

sns.set_style("whitegrid")

IMG_HEIGHT = 112
IMG_WIDTH = 92
IMG_CHANNELS = 1


def load_data():
    """Load and prepare data for CNN."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "umist_cropped.mat"
    )
    
    print("=" * 60)
    print("LOADING DATA (UNREDUCED + AUGMENTED)")
    print("=" * 60)
    
    X_train, X_val, X_test, y_train, y_val, y_test, _ = (
        load_preprocessed_data_with_augmentation(
            dataset_path=path
        )
    )
    
    num_classes = len(np.unique(y_train))
    print(f"\nTrain: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}, Classes: {num_classes}")
    
    # Reshape for CNN
    X_train = X_train.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    X_val = X_val.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    X_test = X_test.reshape(-1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    
    # One-hot encode
    y_train_oh = to_categorical(y_train, num_classes)
    y_val_oh = to_categorical(y_val, num_classes)
    y_test_oh = to_categorical(y_test, num_classes)
    
    return (X_train, X_val, X_test, 
            y_train, y_val, y_test,
            y_train_oh, y_val_oh, y_test_oh, num_classes)


def build_model(hp, num_classes):
    """Build CNN with tunable hyperparameters."""
    model = keras.Sequential()
    model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
    
    # HP: number of filters
    filters = hp.Choice('filters', values=[32, 64], default=32)
    
    # HP: dropout rate
    dropout = hp.Float('dropout', min_value=0.25, max_value=0.5, step=0.25, default=0.25)
    
    # HP: L2 regularization
    l2 = hp.Choice('l2_reg', values=[1e-4, 1e-3], default=1e-4)
    
    # Block 1
    model.add(layers.Conv2D(filters, 3, padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(dropout))
    
    # Block 2
    model.add(layers.Conv2D(filters * 2, 3, padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(dropout))
    
    # Block 3
    model.add(layers.Conv2D(filters * 4, 3, padding='same', activation='relu',
                            kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(dropout))
    
    # Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2)))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # HP: learning rate
    lr = hp.Choice('learning_rate', values=[1e-3, 1e-4], default=1e-3)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def run_tuning(X_train, X_val, y_train_oh, y_val_oh, num_classes):
    """Run hyperparameter search."""
    print("\n" + "=" * 60)
    print("KERAS TUNER - HYPERPARAMETER SEARCH")
    print("=" * 60)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'cnn', 'tuner')
    os.makedirs(output_dir, exist_ok=True)
    
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, num_classes),
        objective='val_accuracy',
        max_epochs=30,
        factor=3,  # Reduction factor for early stopping
        directory=output_dir,
        project_name='cnn',
        overwrite=True
    )
    
    print(f"\nRunning Hyperband search...")
    
    tuner.search(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        batch_size=32,
        verbose=0
    )
    
    best_hps = tuner.get_best_hyperparameters()[0]
    
    print("\n" + "-" * 40)
    print("BEST HYPERPARAMETERS")
    print("-" * 40)
    print(f"  Val Accuracy: {tuner.get_best_trials(num_trials=1)[0].score:.4f}")
    print(f"  Filters: {best_hps.get('filters')}")
    print(f"  Dropout: {best_hps.get('dropout')}")
    print(f"  L2 reg: {best_hps.get('l2_reg')}")
    print(f"  Learning rate: {best_hps.get('learning_rate')}")
    
    return tuner.get_best_models()[0], best_hps


def train_final(best_hps, X_train, X_val, y_train_oh, y_val_oh, num_classes):
    """Train final model with best HPs."""
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)
    
    model = build_model(best_hps, num_classes)
    model.summary()
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'cnn')
    os.makedirs(output_dir, exist_ok=True)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_val, y_val_oh),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    model.save(os.path.join(output_dir, 'best_cnn.keras'))
    
    return model, history


def evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate model and print metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    results = {}
    
    for name, X, y in [('Train', X_train, y_train), 
                        ('Val', X_val, y_val), 
                        ('Test', X_test, y_test)]:
        y_pred = np.argmax(model.predict(X, verbose=0), axis=1)
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        print(f"\n{name}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
        
        results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    
    # Classification report for test
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n" + "-" * 40)
    print("TEST SET CLASSIFICATION REPORT")
    print("-" * 40)
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    return results, y_test_pred


def plot_history(history):
    """Plot training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Val')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Val')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'outputs', 'cnn', 'training_cnn.png'), dpi=100)
    plt.show()


def plot_metrics(results):
    """Plot metrics comparison."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for i, (name, color) in enumerate([('Train', '#2ecc71'), ('Val', '#3498db'), ('Test', '#e74c3c')]):
        vals = [results[name][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.8)
        for bar, v in zip(bars, vals):
            ax.annotate(f'{v:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    ax.set_ylabel('Score')
    ax.set_title('CNN Performance (Unreduced + Augmented)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'outputs', 'cnn', 'metrics_cnn.png'), dpi=100)
    plt.show()


def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'cnn')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    (X_train, X_val, X_test, 
     y_train, y_val, y_test,
     y_train_oh, y_val_oh, y_test_oh, num_classes) = load_data()
    
    # Check if trained model already exists
    model_path = os.path.join(output_dir, 'best_cnn.keras')
    
    if os.path.exists(model_path):
        print("\n" + "=" * 60)
        print("LOADING EXISTING MODEL")
        print("=" * 60)
        print(f"Loading from: {model_path}")
        model = keras.models.load_model(model_path)
        history = None
    else:
        # Tune
        best_model, best_hps = run_tuning(X_train, X_val, y_train_oh, y_val_oh, num_classes)
        
        # Train final
        model, history = train_final(best_hps, X_train, X_val, y_train_oh, y_val_oh, num_classes)
    
    # Evaluate
    results, _ = evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Plot
    print("\n" + "=" * 60)
    print("PLOTS")
    print("=" * 60)
    if history:
        plot_history(history)
    else:
        print("Skipping training history plot (loaded existing model)")
    plot_metrics(results)
    
    # Summary
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Test Accuracy: {results['Test']['accuracy']:.4f}")
    print(f"Test F1: {results['Test']['f1']:.4f}")


if __name__ == "__main__":
    main()
