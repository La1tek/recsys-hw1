from contextlib import nullcontext
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.linalg import solve

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


def _prepare_interactions(X):
    if not sparse.issparse(X):
        raise TypeError("X must be a scipy sparse interaction matrix")
    X = sparse.csr_matrix(X, dtype=np.float64, copy=True)
    if len(X.shape) != 2 or min(X.shape) < 1:
        raise ValueError("X must have at least one user and one item")
    if not np.isfinite(X.data).all() or np.any(X.data < 0):
        raise ValueError("X must contain finite nonnegative interactions")
    X.sum_duplicates()
    X.eliminate_zeros()
    X.sort_indices()
    X.data[:] = 1.0
    return X


def _least_squares_step(observations, fixed_factors, alpha, regularization):
    fixed_factors = np.asarray(fixed_factors, dtype=np.float64)
    factors = fixed_factors.shape[1]
    base = fixed_factors.T @ fixed_factors
    base.flat[:: factors + 1] += regularization
    result = np.zeros((observations.shape[0], factors), dtype=np.float64)
    for row in range(observations.shape[0]):
        start, end = observations.indptr[row : row + 2]
        if start == end:
            continue
        neighbors = fixed_factors[observations.indices[start:end]]
        A = base + alpha * (neighbors.T @ neighbors)
        b = (1.0 + alpha) * (neighbors.T @ observations.data[start:end])
        result[row] = solve(A, b, assume_a="pos", check_finite=False)
    return result


def _objective(observations, users, items, alpha, regularization):
    # Ошибка по всем парам при p=0 и c=1.
    loss = float(np.sum((users.T @ users) * (items.T @ items)))
    for user in range(observations.shape[0]):
        start, end = observations.indptr[user : user + 2]
        if start == end:
            continue
        prediction = items[observations.indices[start:end]] @ users[user]
        target = observations.data[start:end]
        loss += float(np.sum((1.0 + alpha) * (target - prediction) ** 2
                             - prediction ** 2))
    loss += regularization * float(np.sum(users ** 2) + np.sum(items ** 2))
    return loss


class ImplicitALS:
    def __init__(self, factors=32, regularization=1.0, alpha=20.0,
                 iterations=12, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def _validate_parameters(self):
        if not isinstance(self.factors, (int, np.integer)) or self.factors < 1:
            raise ValueError("factors must be a positive integer")
        if not isinstance(self.iterations, (int, np.integer)) or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not np.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be finite and positive")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")

    def fit(self, X, **kwargs):
        observations = _prepare_interactions(X)
        return self._fit_targets(observations)

    def _fit_targets(self, observations):
        self._validate_parameters()
        started = perf_counter()
        self.n_users_, self.n_items_ = observations.shape
        transposed = observations.T.tocsr()
        transposed.sort_indices()
        rng = np.random.default_rng(self.seed)
        items = rng.normal(0.0, 0.01, (self.n_items_, self.factors))
        self.history_ = []
        self.loss_history_ = []
        limits = threadpool_limits(limits=1, user_api="blas") if threadpool_limits else nullcontext()
        with limits:
            for iteration in range(1, self.iterations + 1):
                users = _least_squares_step(observations, items, self.alpha,
                                            self.regularization)
                items = _least_squares_step(transposed, users, self.alpha,
                                            self.regularization)
                objective = _objective(observations, users, items, self.alpha,
                                       self.regularization)
                if not np.isfinite(objective):
                    raise FloatingPointError("ALS objective became non-finite")
                self.loss_history_.append(objective)
                self.history_.append({"iteration": iteration,
                                      "objective": objective,
                                      "elapsed_seconds": perf_counter() - started})
        self.user_factors_ = users.astype(np.float32)
        self.item_factors_ = items.astype(np.float32)
        self.fit_seconds_ = perf_counter() - started
        return self

    def score(self, user_ids):
        if not hasattr(self, "user_factors_"):
            raise RuntimeError("fit must be called before score")
        user_ids = np.asarray(user_ids)
        if user_ids.ndim != 1 or user_ids.dtype.kind not in "iu":
            raise ValueError("user_ids must be a one-dimensional integer array")
        if np.any(user_ids < 0) or np.any(user_ids >= self.n_users_):
            raise ValueError("user ID is outside the fitted interaction matrix")
        return np.asarray(self.user_factors_[user_ids] @ self.item_factors_.T,
                          dtype=np.float32)
