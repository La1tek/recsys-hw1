from __future__ import annotations

import time

import numpy as np
from scipy import linalg, sparse


class EASE:
    def __init__(self, regularization: float = 300.0):
        if not np.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        self.regularization = float(regularization)

    def fit(self, X: sparse.csr_matrix, **kwargs) -> "EASE":
        started = time.perf_counter()
        matrix = sparse.csr_matrix(X, dtype=np.float32, copy=True)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        if matrix.data.size and (
            not np.isfinite(matrix.data).all() or (matrix.data != 1).any()
        ):
            raise ValueError("X must contain binary 0/1 interactions")
        matrix.sort_indices()
        self.X_train_ = matrix
        self.n_users_, self.n_items_ = matrix.shape
        work_matrix = matrix.astype(np.float64)
        gram = (work_matrix.T @ work_matrix).toarray()
        gram.flat[:: self.n_items_ + 1] += self.regularization
        cholesky = linalg.cho_factor(gram, lower=True, check_finite=False)
        precision = linalg.cho_solve(
            cholesky, np.eye(self.n_items_, dtype=np.float64), check_finite=False
        )
        coefficients = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(coefficients, 0.0)
        self.coefficients_ = coefficients.astype(np.float32)
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coefficients_"):
            raise RuntimeError("Call fit before score")
        user_ids = np.asarray(user_ids)
        if user_ids.ndim != 1 or user_ids.dtype.kind not in "iu":
            raise ValueError("user_ids must be a one-dimensional integer array")
        if ((user_ids < 0) | (user_ids >= self.n_users_)).any():
            raise IndexError("user ID is outside the fitted matrix")
        return np.asarray(
            self.X_train_[user_ids] @ self.coefficients_, dtype=np.float32
        )
