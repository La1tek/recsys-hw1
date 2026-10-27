from __future__ import annotations

import time

import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors: int = 100, shrinkage: float = 10.0):
        if not isinstance(neighbors, (int, np.integer)) or neighbors < 1:
            raise ValueError("neighbors must be a positive integer")
        if not np.isfinite(shrinkage) or shrinkage < 0:
            raise ValueError("shrinkage must be finite and non-negative")
        self.neighbors = int(neighbors)
        self.shrinkage = float(shrinkage)

    def fit(self, X: sparse.csr_matrix, **kwargs) -> "ItemKNN":
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

        counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
        cooccurrence = (matrix.T @ matrix).tocsc()
        indptr = [0]
        all_indices = []
        all_values = []
        for target in range(self.n_items_):
            left, right = cooccurrence.indptr[target : target + 2]
            source = cooccurrence.indices[left:right]
            overlap = cooccurrence.data[left:right].astype(np.float64)
            keep = source != target
            source, overlap = source[keep], overlap[keep]
            denominator = np.sqrt(counts[source] * counts[target])
            similarity = np.divide(
                overlap,
                denominator,
                out=np.zeros_like(overlap),
                where=denominator > 0,
            )
            if self.shrinkage:
                similarity *= overlap / (overlap + self.shrinkage)
            selected = np.lexsort((source, -similarity))[: self.neighbors]
            source, similarity = source[selected], similarity[selected]
            order = np.argsort(source)
            all_indices.extend(source[order])
            all_values.extend(similarity[order])
            indptr.append(len(all_indices))
        self.similarity_ = sparse.csc_matrix(
            (
                np.asarray(all_values, dtype=np.float32),
                np.asarray(all_indices, dtype=np.int32),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(self.n_items_, self.n_items_),
        ).tocsr()
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        if not hasattr(self, "similarity_"):
            raise RuntimeError("Call fit before score")
        user_ids = np.asarray(user_ids)
        if user_ids.ndim != 1 or user_ids.dtype.kind not in "iu":
            raise ValueError("user_ids must be a one-dimensional integer array")
        if ((user_ids < 0) | (user_ids >= self.n_users_)).any():
            raise IndexError("user ID is outside the fitted matrix")
        return np.asarray(
            (self.X_train_[user_ids] @ self.similarity_).toarray(),
            dtype=np.float32,
        )
