"""
Dimensionality Reduction via PCA and PCA->LDA
==============================================

This script loads the preprocessed UMIST facial recognition dataset
(train/val/test splits already normalized) and applies dimensionality
reduction using two methods:

Method 1: PCA Only (Automatic Component Selection)
---------------------------------------------------
1. Fit an initial PCA (randomized solver) on the training data with the
   maximum feasible number of components (min(n_samples, n_features)).
2. Compute the cumulative explained variance curve.
3. Use `kneed.KneeLocator` to detect the "knee" (point of diminishing returns)
   on the cumulative variance curve (concave, increasing).
4. If a knee is found, select that number of components.
5. If not found, fall back to a variance threshold (default 0.95).

Method 2: PCA -> LDA (Two-Stage Reduction)
------------------------------------------
1. First reduce dimensionality with PCA to n_samples - n_classes - 5
   (slightly less than the theoretical maximum for LDA stability).
2. Display an elbow curve with the calculated n_components marked.
3. Apply LDA to the PCA-reduced data for supervised dimensionality reduction.
4. LDA reduces to at most n_classes - 1 components.

Outputs:
--------
- Prints chosen number of principal components and cumulative variance.
- Displays explained variance plots with knee/threshold markers.
- Transforms train/val/test sets into reduced space.
- (Optional) Saves models and transformed splits.

Usage:
------
	python "dimensionality reduction.py"

Dependencies (ensure installed):
	pip install scikit-learn kneed matplotlib seaborn numpy pandas

Notes:
------
- PCA and LDA are fit ONLY on training data to avoid leakage.
- Validation and test sets are transformed using the fitted models.
- The randomized solver is used for speed on high-dimensional data.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional

from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Import pipeline loader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_preprocessed_data, load_preprocessed_data_with_augmentation

sns.set_style("whitegrid")


def determine_pca_components(
	X_train: np.ndarray,
	variance_threshold: float = 0.95,
	max_components: Optional[int] = None,
	plot: bool = True,
	save_plot_path: Optional[str] = None,
) -> Tuple[int, PCA, np.ndarray, np.ndarray]:
	"""Determine number of PCA components automatically using KneeLocator.

	Parameters
	----------
	X_train : np.ndarray
		Normalized training feature matrix (n_samples, n_features).
	variance_threshold : float, optional
		Fallback cumulative explained variance threshold if knee not found.
	max_components : int, optional
		Explicit cap on components; defaults to min(n_samples, n_features).
	plot : bool, optional
		Whether to display the explained variance curve plot.
	save_plot_path : str, optional
		If provided, saves the plot to this path instead of (or in addition to) showing.

	Returns
	-------
	n_components : int
		Selected number of principal components.
	pca_full : PCA
		PCA model fitted with full component count (used only for variance curve).
	cum_var : np.ndarray
		Cumulative explained variance ratios.
	var_ratio : np.ndarray
		Explained variance ratio per component.
	"""
	n_samples, n_features = X_train.shape
	if max_components is None:
		max_components = min(n_samples, n_features)

	print(f"Fitting initial PCA with up to {max_components} components for curve...")
	pca_full = PCA(n_components=max_components, svd_solver="randomized", random_state=42)
	pca_full.fit(X_train)

	var_ratio = pca_full.explained_variance_ratio_
	cum_var = np.cumsum(var_ratio)
	x = np.arange(1, len(cum_var) + 1)

	# Knee detection on cumulative variance curve
	print("Running KneeLocator on cumulative explained variance curve...")
	try:
		knee_locator = KneeLocator(
			x,
			cum_var,
			curve="concave",
			direction="increasing",
			online=False,
		)
		knee = knee_locator.knee  # may be None
	except Exception as e:  # pragma: no cover - defensive
		print(f"KneeLocator failed: {e}. Falling back to variance threshold.")
		knee = None

	if knee is not None:
		n_components = int(knee)
		method = "knee"
	else:
		# Fallback: first index reaching variance_threshold
		above_threshold = np.where(cum_var >= variance_threshold)[0]
		if above_threshold.size > 0:
			n_components = int(above_threshold[0] + 1)  # +1 due to 0-index
		else:
			n_components = len(cum_var)  # use all components as last resort
		method = f"threshold ({variance_threshold:.2f})"

	final_cum_var = cum_var[n_components - 1]
	print(
		f"Selected {n_components} components using {method}; "
		f"cumulative explained variance = {final_cum_var:.4f}"
	)

	if plot:
		plt.figure(figsize=(10, 6))
		plt.plot(x, cum_var, label="Cumulative Explained Variance", color="steelblue")
		plt.scatter(x, cum_var, s=12, alpha=0.4, color="steelblue")
		plt.axvline(x=n_components, color="green", linestyle="--", label=f"Selected Components: {n_components}")
		if knee is not None:
			plt.axhline(y=final_cum_var, color="orange", linestyle="--", label=f"Explained Variance at Knee: {final_cum_var:.2f}")
			plt.scatter([n_components], [final_cum_var], color="red", s=60, label="Knee")
		else:
			plt.axhline(y=variance_threshold, color="orange", linestyle="--", label=f"Threshold {variance_threshold:.2f}")
			plt.scatter([n_components], [final_cum_var], color="red", s=60, label="Threshold")
		plt.xlabel("Number of Components")
		plt.ylabel("Cumulative Explained Variance Ratio")
		plt.title("PCA Explained Variance Curve with Knee/Threshold Selection")
		plt.legend()
		plt.tight_layout()
		plt.show()

	return n_components, pca_full, cum_var, var_ratio


def fit_and_transform_pca(
	X_train: np.ndarray,
	X_val: np.ndarray,
	X_test: np.ndarray,
	n_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
	"""Fit PCA on training set and transform all splits.

	Parameters
	----------
	X_train, X_val, X_test : np.ndarray
		Normalized feature matrices.
	n_components : int
		Number of components to retain.

	Returns
	-------
	X_train_pca, X_val_pca, X_test_pca : np.ndarray
		Transformed feature matrices.
	pca : PCA
		Fitted PCA object.
	"""
	print(f"Fitting final PCA with {n_components} components on training set...")
	pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
	X_train_pca = pca.fit_transform(X_train)
	X_val_pca = pca.transform(X_val)
	X_test_pca = pca.transform(X_test)
	print(
		"Shapes after PCA -> Train: {} | Val: {} | Test: {}".format(
			X_train_pca.shape, X_val_pca.shape, X_test_pca.shape
		)
	)
	return X_train_pca, X_val_pca, X_test_pca, pca


def reconstruct_from_pca(X_pca, pca, scaler=None):
	"""
	Reconstruct original images from PCA-transformed data.

	Parameters
	----------
	X_pca : np.ndarray
		PCA-transformed data (n_samples, n_components)
	pca : PCA
		Fitted PCA object
	scaler : StandardScaler, optional
		If provided, inverse transform to original scale

	Returns
	-------
	X_reconstructed : np.ndarray
		Reconstructed images in original feature space (n_samples, n_features)
	"""
	# Inverse transform from PCA space to original feature space
	X_reconstructed = pca.inverse_transform(X_pca)

	# If scaler provided, inverse transform to original pixel scale
	if scaler is not None:
		X_reconstructed = scaler.inverse_transform(X_reconstructed)

	return X_reconstructed


def determine_pca_components_for_lda(
	X_train: np.ndarray,
	y_train: np.ndarray,
	plot: bool = True,
	save_plot_path: Optional[str] = None,
) -> Tuple[int, PCA, np.ndarray, np.ndarray]:
	"""Determine number of PCA components for PCA->LDA pipeline.

	For LDA to work properly, we need to reduce dimensionality with PCA first
	to a value less than n_samples - n_classes. This function calculates the
	appropriate number of components and displays the explained variance curve.

	Parameters
	----------
	X_train : np.ndarray
		Normalized training feature matrix (n_samples, n_features).
	y_train : np.ndarray
		Training labels.
	plot : bool, optional
		Whether to display the explained variance curve plot.
	save_plot_path : str, optional
		If provided, saves the plot to this path instead of (or in addition to) showing.

	Returns
	-------
	n_components : int
		Selected number of principal components for PCA step.
	pca_full : PCA
		PCA model fitted with full component count (used only for variance curve).
	cum_var : np.ndarray
		Cumulative explained variance ratios.
	var_ratio : np.ndarray
		Explained variance ratio per component.
	"""
	n_samples, n_features = X_train.shape
	n_classes = len(np.unique(y_train))

	# For LDA, we need n_components < n_samples - n_classes
	# We use a slightly smaller value to ensure numerical stability
	max_pca_components = min(n_samples, n_features)
	n_components = n_samples - n_classes - 5  # Small buffer for stability
	n_components = max(1, min(n_components, max_pca_components))

	print(f"PCA->LDA Pipeline:")
	print(f"  Training samples: {n_samples}")
	print(f"  Number of classes: {n_classes}")
	print(f"  Max PCA components for LDA: {n_samples - n_classes}")
	print(f"  Selected PCA components: {n_components}")

	print(f"\nFitting initial PCA with up to {max_pca_components} components for curve...")
	pca_full = PCA(n_components=max_pca_components, svd_solver="randomized", random_state=42)
	pca_full.fit(X_train)

	var_ratio = pca_full.explained_variance_ratio_
	cum_var = np.cumsum(var_ratio)
	x = np.arange(1, len(cum_var) + 1)

	final_cum_var = cum_var[n_components - 1]
	print(
		f"Selected {n_components} PCA components; "
		f"cumulative explained variance = {final_cum_var:.4f}"
	)

	if plot:
		plt.figure(figsize=(10, 6))
		plt.plot(x, cum_var, label="Cumulative Explained Variance", color="steelblue")
		plt.scatter(x, cum_var, s=12, alpha=0.4, color="steelblue")

		# Vertical line at selected components
		plt.axvline(x=n_components, color="green", linestyle="--",
					label=f"PCA Components for LDA: {n_components}")

		# Horizontal line at the explained variance for selected components
		plt.axhline(y=final_cum_var, color="orange", linestyle="--",
					label=f"Explained Variance: {final_cum_var:.2f}")

		# Mark the intersection point
		plt.scatter([n_components], [final_cum_var], color="red", s=100, zorder=5,
					label=f"n_samples - n_classes - 5 = {n_components}")

		# Add annotation
		plt.annotate(f'({n_components}, {final_cum_var:.3f})',
					xy=(n_components, final_cum_var),
					xytext=(n_components + 20, final_cum_var - 0.05),
					fontsize=10,
					arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

		plt.xlabel("Number of Components")
		plt.ylabel("Cumulative Explained Variance Ratio")
		plt.title("PCA Explained Variance Curve for PCA→LDA Pipeline\n"
				  f"(n_samples={n_samples}, n_classes={n_classes})")
		plt.legend(loc='lower right')
		plt.tight_layout()

		if save_plot_path:
			plt.savefig(save_plot_path, dpi=150, bbox_inches='tight')
			print(f"Plot saved to {save_plot_path}")

		plt.show()

	return n_components, pca_full, cum_var, var_ratio


def fit_and_transform_pca_lda(
	X_train: np.ndarray,
	X_val: np.ndarray,
	X_test: np.ndarray,
	y_train: np.ndarray,
	n_pca_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, LDA]:
	"""Fit PCA->LDA pipeline on training set and transform all splits.

	This two-step dimensionality reduction:
	1. First applies PCA to reduce to n_pca_components
	2. Then applies LDA for further supervised dimensionality reduction

	Parameters
	----------
	X_train, X_val, X_test : np.ndarray
		Normalized feature matrices.
	y_train : np.ndarray
		Training labels (required for LDA).
	n_pca_components : int
		Number of PCA components to retain before LDA.

	Returns
	-------
	X_train_lda, X_val_lda, X_test_lda : np.ndarray
		Transformed feature matrices after PCA->LDA.
	pca : PCA
		Fitted PCA object.
	lda : LDA
		Fitted LDA object.
	"""
	n_classes = len(np.unique(y_train))

	# Step 1: PCA
	print(f"\n[Step 1] Fitting PCA with {n_pca_components} components on training set...")
	pca = PCA(n_components=n_pca_components, svd_solver="randomized", random_state=42)
	X_train_pca = pca.fit_transform(X_train)
	X_val_pca = pca.transform(X_val)
	X_test_pca = pca.transform(X_test)
	print(
		"Shapes after PCA -> Train: {} | Val: {} | Test: {}".format(
			X_train_pca.shape, X_val_pca.shape, X_test_pca.shape
		)
	)

	# Step 2: LDA
	# LDA can have at most min(n_classes - 1, n_features) components
	max_lda_components = min(n_classes - 1, n_pca_components)
	print(f"\n[Step 2] Fitting LDA with up to {max_lda_components} components (n_classes - 1)...")
	lda = LDA(n_components=max_lda_components)
	X_train_lda = lda.fit_transform(X_train_pca, y_train)
	X_val_lda = lda.transform(X_val_pca)
	X_test_lda = lda.transform(X_test_pca)
	print(
		"Shapes after LDA -> Train: {} | Val: {} | Test: {}".format(
			X_train_lda.shape, X_val_lda.shape, X_test_lda.shape
		)
	)

	# Display LDA explained variance ratio
	lda_var_ratio = lda.explained_variance_ratio_
	lda_cum_var = np.cumsum(lda_var_ratio)
	print(f"\nLDA explained variance ratios: {lda_var_ratio}")
	print(f"LDA cumulative explained variance: {lda_cum_var[-1]:.4f}")

	return X_train_lda, X_val_lda, X_test_lda, pca, lda


def main():
	path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "umist_cropped.mat")

	X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data_with_augmentation(dataset_path=path)
	print(
		f"Loaded splits -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}. "
		f"Feature dimensionality: {X_train.shape[1]}"
	)
	
	print("\n" + "="*70)
	print("PCA DIMENSIONALITY REDUCTION")
	print("="*70)

	n_components, pca_full, cum_var, var_ratio = determine_pca_components(
		X_train,
		variance_threshold=0.95,
		max_components=None,
		plot=True,
	)

	# Fit and transform
	X_train_pca, X_val_pca, X_test_pca, pca = fit_and_transform_pca(
		X_train, X_val, X_test, n_components
	)

	final_var = np.cumsum(pca.explained_variance_ratio_)[-1]
	print(f"\nFinal PCA summary:")
	print(f"  Components: {n_components}/{X_train.shape[1]} ({n_components/X_train.shape[1]*100:.1f}%)")
	print(f"  Explained variance: {final_var:.4f} ({final_var*100:.2f}%)")

	print("\n" + "="*70)
	print("PCA -> LDA DIMENSIONALITY REDUCTION")
	print("="*70)
	# Determine PCA components for LDA
	n_pca_components, pca_full_lda, cum_var_lda, var_ratio_lda = determine_pca_components_for_lda(
		X_train,
		y_train,
		plot=True,
	)
	# Fit and transform
	X_train_lda, X_val_lda, X_test_lda, pca_lda, lda = fit_and_transform_pca_lda(
		X_train, X_val, X_test, y_train, n_pca_components
	)
	
	print(f"\nFinal PCA->LDA summary:")
	print(f"  Reduction: {X_train.shape[1]} -> {n_pca_components} (PCA) -> {X_train_lda.shape[1]} (LDA)")
	print(f"  Explained variance (PCA): {np.cumsum(pca_lda.explained_variance_ratio_)[-1]:.4f}")
	print(f"  Explained variance (LDA): {np.cumsum(lda.explained_variance_ratio_)[-1]:.4f}")


if __name__ == "__main__":
	main()

