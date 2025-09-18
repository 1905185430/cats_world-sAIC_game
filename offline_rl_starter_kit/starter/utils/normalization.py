
from __future__ import annotations
import numpy as np

class Scaler:
    def __init__(self, eps: float = 1e-6):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + self.eps

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None: return X
        return (X - self.mean) / self.std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None: return X
        return X * self.std + self.mean
