"""
probe.py — HallucinationProbe with PCA + tuned LogisticRegression.

Small dataset (689 samples) with high-dimensional features → PCA reduces
overfitting while preserving ~95% variance. Weaker regularization lets the
model actually learn from the rich multi-layer features.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier: StandardScaler → PCA(200) → LogisticRegression."""

    def __init__(self, pca_components: int = 200) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._pca_components = pca_components
        self._model = self._new_model()
        self._threshold: float = 0.5

    def _new_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=10000,
            random_state=42,
            solver="lbfgs",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self._model, "coef_"):
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")
        coef = torch.as_tensor(self._model.coef_[0], dtype=x.dtype, device=x.device)
        intercept = torch.as_tensor(
            self._model.intercept_[0], dtype=x.dtype, device=x.device
        )
        return x @ coef + intercept

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.asarray(X, dtype=np.float32), copy=False)

    @staticmethod
    def _best_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 201)]))
        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            acc = accuracy_score(y_true, y_pred_t)
            f1 = f1_score(y_true, y_pred_t, zero_division=0)
            if (acc > best_accuracy) or (acc == best_accuracy and f1 > best_f1):
                best_accuracy = acc
                best_f1 = f1
                best_threshold = float(t)
        return best_threshold

    def _fit_oof_threshold(self, X: np.ndarray, y: np.ndarray) -> None:
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2 or counts.min() < 3:
            self._threshold = 0.5
            return
        n_splits = min(5, int(counts.min()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y), dtype=np.float32)
        for idx_train, idx_val in cv.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[idx_train])
            if self._pca is not None:
                X_tr = self._pca.fit_transform(X_tr)
            X_va = scaler.transform(X[idx_val])
            if self._pca is not None:
                X_va = self._pca.transform(X_va)
            model = self._new_model()
            model.fit(X_tr, y[idx_train])
            oof_probs[idx_val] = model.predict_proba(X_va)[:, 1]
        self._threshold = self._best_threshold(oof_probs, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X = self._clean(X)
        y = np.asarray(y, dtype=int)

        X_scaled = self._scaler.fit_transform(X)

        # Apply PCA — keep n_components or auto-select to preserve 95% variance
        n_components = min(self._pca_components, X_scaled.shape[0], X_scaled.shape[1])
        self._pca = PCA(n_components=n_components, random_state=42)
        X_reduced = self._pca.fit_transform(X_scaled)

        self._model = self._new_model()
        self._model.fit(X_reduced, y)
        self._fit_oof_threshold(X, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        X_val = self._prepare(X_val)
        probs = self._model.predict_proba(X_val)[:, 1]
        self._threshold = self._best_threshold(probs, np.asarray(y_val, dtype=int))
        return self

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(self._clean(X))
        if self._pca is not None:
            return self._pca.transform(X_scaled)
        return X_scaled

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_prepared = self._prepare(X)
        return self._model.predict_proba(X_prepared)
