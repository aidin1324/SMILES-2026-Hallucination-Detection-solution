"""
probe.py — HallucinationProbe with scaled LogisticRegression.

The labelled set is small, so the probe uses strong L2 regularization instead
of a high-capacity neural net or PCA projection.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier: StandardScaler → LogisticRegression."""

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._model = self._new_model()
        self._threshold: float = 0.5

    def _new_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=0.0003,
            class_weight=None,
            max_iter=5000,
            random_state=42,
            solver="liblinear",
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
            X_va = scaler.transform(X[idx_val])
            model = self._new_model()
            model.fit(X_tr, y[idx_train])
            oof_probs[idx_val] = model.predict_proba(X_va)[:, 1]
        self._threshold = self._best_threshold(oof_probs, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        X = self._clean(X)
        y = np.asarray(y, dtype=int)

        X_scaled = self._scaler.fit_transform(X)
        self._model = self._new_model()
        self._model.fit(X_scaled, y)
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
        return self._scaler.transform(self._clean(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_prepared = self._prepare(X)
        return self._model.predict_proba(X_prepared)
