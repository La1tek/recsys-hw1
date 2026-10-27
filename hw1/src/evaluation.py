import numpy as np
from scipy import sparse


def _binary_csr(matrix, name):
    if not sparse.issparse(matrix):
        raise TypeError(f"{name} must be a scipy sparse matrix")
    matrix = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    if min(matrix.shape) < 1:
        raise ValueError(f"{name} must contain at least one user and one item")
    if not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    matrix.data[:] = 1.0
    return matrix


def evaluate(model, history_csr, truth_csr, k=10, batch_size=128,
             return_details=False):
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    history = _binary_csr(history_csr, "history_csr")
    truth = _binary_csr(truth_csr, "truth_csr")
    if history.shape != truth.shape:
        raise ValueError("history and truth must have the same user/item shape")
    if history.multiply(truth).nnz:
        raise ValueError("history and truth must have disjoint nonzero coordinates")

    relevant_counts = np.diff(truth.indptr)
    users = np.flatnonzero(relevant_counts > 0)
    if users.size == 0:
        raise ValueError("truth contains no users with relevant held-out items")
    n_items = history.shape[1]
    n_ranked = min(k, n_items)
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    ideal_dcg = np.cumsum(discounts)
    recalls = np.empty(users.size, dtype=np.float64)
    ndcgs = np.empty(users.size, dtype=np.float64)
    precisions = np.empty(users.size, dtype=np.float64)
    recommendations = np.full((users.size, k), -1, dtype=np.int64) if return_details else None
    recommended_catalog = np.zeros(n_items, dtype=bool)

    for start in range(0, users.size, batch_size):
        end = min(start + batch_size, users.size)
        batch_users = users[start:end]
        raw_scores = np.asarray(model.score(batch_users))
        if raw_scores.shape != (batch_users.size, n_items):
            raise ValueError("model.score must return [batch_size, n_items]")
        if raw_scores.dtype.kind not in "fiu" or not np.isfinite(raw_scores).all():
            raise ValueError("model scores must be real and finite before seen-item masking")
        scores = raw_scores.astype(np.float64, copy=True)
        for row, user in enumerate(batch_users):
            lo, hi = history.indptr[user : user + 2]
            scores[row, history.indices[lo:hi]] = -np.inf
        ranked = np.argsort(-scores, axis=1, kind="stable")[:, :n_ranked]
        available = np.isfinite(np.take_along_axis(scores, ranked, axis=1))
        recs = np.where(available, ranked, -1)
        safe_indices = np.maximum(recs, 0)
        selected_truth = truth[batch_users[:, None], safe_indices]
        selected_truth = selected_truth.toarray() if sparse.issparse(selected_truth) else np.asarray(selected_truth)
        hits = selected_truth > 0
        hits &= available
        hit_counts = hits.sum(axis=1)
        recalls[start:end] = hit_counts / relevant_counts[batch_users]
        precisions[start:end] = hit_counts / float(k)
        ndcgs[start:end] = (hits @ discounts[:n_ranked]
                            / ideal_dcg[np.minimum(k, relevant_counts[batch_users]) - 1])
        recommended_catalog[recs[available]] = True
        if recommendations is not None:
            recommendations[start:end, :n_ranked] = recs

    metrics = {
        f"recall@{k}": float(recalls.mean()),
        f"ndcg@{k}": float(ndcgs.mean()),
        f"precision@{k}": float(precisions.mean()),
        f"coverage@{k}": float(recommended_catalog.mean()),
        "n_users": int(users.size),
    }
    if not return_details:
        return metrics
    return metrics, {"users": users, "recall": recalls, "ndcg": ndcgs,
                     "precision": precisions, "recommendations": recommendations}


class MostPopular:

    def fit(self, X, **kwargs):
        X = _binary_csr(X, "X")
        self.n_users_, self.n_items_ = X.shape
        self.item_scores_ = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        return self

    def score(self, user_ids):
        if not hasattr(self, "item_scores_"):
            raise RuntimeError("fit must be called before score")
        user_ids = np.asarray(user_ids)
        if user_ids.ndim != 1 or user_ids.dtype.kind not in "iu":
            raise ValueError("user_ids must be a one-dimensional integer array")
        if np.any(user_ids < 0) or np.any(user_ids >= self.n_users_):
            raise ValueError("user ID is outside the fitted interaction matrix")
        return np.broadcast_to(self.item_scores_, (user_ids.size, self.n_items_))


def paired_bootstrap_difference(values_a, values_b, n_bootstrap=2000,
                                confidence=0.95, seed=42):
    a, b = np.asarray(values_a, dtype=np.float64), np.asarray(values_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("paired metrics must be equally sized nonempty one-dimensional arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired metrics must be finite")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    differences = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    chunk_size = max(1, min(128, 2_000_000 // a.size))
    for start in range(0, n_bootstrap, chunk_size):
        end = min(start + chunk_size, n_bootstrap)
        indices = rng.integers(0, a.size, size=(end - start, a.size))
        means[start:end] = differences[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return {"mean_difference": float(differences.mean()),
            "ci_low": float(low), "ci_high": float(high),
            "confidence": float(confidence), "n_bootstrap": int(n_bootstrap),
            "n_users": int(a.size), "seed": int(seed)}
