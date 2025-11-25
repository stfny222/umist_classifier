"""
Dimensionality Reduction via PCA with Automatic Component Selection
===================================================================

This script loads the preprocessed UMIST facial recognition dataset
(train/val/test splits already normalized) and applies Principal Component
Analysis (PCA) to reduce dimensionality.

Automatic Component Selection Strategy:
--------------------------------------
1. Fit an initial PCA (randomized solver) on the training data with the
   maximum feasible number of components (min(n_samples, n_features)).
2. Compute the cumulative explained variance curve.
3. Use `kneed.KneeLocator` to detect the "knee" (point of diminishing returns)
   on the cumulative variance curve (concave, increasing).
4. If a knee is found, select that number of components.
5. If not found, fall back to a variance threshold (default 0.95).

Outputs:
--------
- Prints chosen number of principal components and cumulative variance.
- Displays an explained variance plot with knee/threshold marker.
- Transforms train/val/test sets into PCA space.
- (Optional) Saves PCA model and transformed splits under `processed_data/pca/`.

Usage:
------
	python "dimensionality reduction.py"

	# Optional arguments (run via PowerShell):
	python "dimensionality reduction.py" --variance-threshold 0.98 --no-save

Dependencies (ensure installed):
	pip install scikit-learn kneed matplotlib seaborn numpy pandas

Notes:
------
- PCA is fit ONLY on training data to avoid leakage.
- Validation and test sets are transformed using the fitted PCA.
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

# Import pipeline loader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import load_preprocessed_data

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


def main():

	X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_preprocessed_data(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "umist_cropped.mat"))
	print(
		f"Loaded splits -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}. "
		f"Feature dimensionality: {X_train.shape[1]}"
	)

	n_components, pca_full, cum_var, var_ratio = determine_pca_components(
		X_train,
		variance_threshold=0.95,
		max_components=None,
		plot=True,
	)

	X_train_pca, X_val_pca, X_test_pca, pca = fit_and_transform_pca(
		X_train, X_val, X_test, n_components
	)

	print(
		f"Final PCA cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)[-1]:.4f}"
	)


if __name__ == "__main__":
	main()

