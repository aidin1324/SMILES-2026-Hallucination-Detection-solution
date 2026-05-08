"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module`` for compatibility with the starter code, but
    uses a regularised logistic regression probe over scaled hidden-state
    features.  This is deliberately small for the 689-row dataset.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._model = self._new_model()
        self._threshold: float = 0.5

    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the network definition below.
    # ------------------------------------------------------------------
    def _new_model(self) -> LogisticRegression:
        return LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
            solver="liblinear",
        )

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
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
    def _best_threshold(
        probs: np.ndarray,
        y_true: np.ndarray,
    ) -> float:
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
        """Set a default threshold for final training without a validation set."""
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2 or counts.min() < 3:
            self._threshold = 0.5
            return

        n_splits = min(3, int(counts.min()))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y), dtype=np.float32)

        for idx_train, idx_val in cv.split(X, y):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[idx_train])
            X_val = scaler.transform(X[idx_val])
            model = clone(self._model)
            model.fit(X_train, y[idx_train])
            oof_probs[idx_val] = model.predict_proba(X_val)[:, 1]

        self._threshold = self._best_threshold(oof_probs, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
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
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = self._best_threshold(probs, np.asarray(y_val, dtype=int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(self._clean(X))
        return self._model.predict_proba(X_scaled)
