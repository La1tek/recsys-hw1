import numpy as np
from scipy import sparse

from .ials import ImplicitALS, _prepare_interactions


class RecencyALS(ImplicitALS):
    def __init__(self, factors=32, regularization=1.0, alpha=20.0,
                 iterations=12, seed=42, half_life_days=180, floor=0.25):
        super().__init__(factors=factors, regularization=regularization,
                         alpha=alpha, iterations=iterations, seed=seed)
        self.half_life_days = half_life_days
        self.floor = floor

    def fit(self, X, timestamps, cutoff, **kwargs):
        if not np.isfinite(self.half_life_days) or self.half_life_days <= 0:
            raise ValueError("half_life_days must be finite and positive")
        if not np.isfinite(self.floor) or not 0 <= self.floor <= 1:
            raise ValueError("floor must lie in [0, 1]")
        if not np.isfinite(cutoff):
            raise ValueError("cutoff must be a finite epoch-second timestamp")
        observations = _prepare_interactions(X)
        if not sparse.issparse(timestamps):
            raise TypeError("timestamps must be a scipy sparse matrix")
        times = sparse.csr_matrix(timestamps, dtype=np.float64, copy=True)
        before = times.nnz
        times.sum_duplicates()
        times.sort_indices()
        if times.nnz != before:
            raise ValueError("timestamps must contain one entry per observed coordinate")
        if (times.shape != observations.shape
                or not np.array_equal(times.indptr, observations.indptr)
                or not np.array_equal(times.indices, observations.indices)):
            raise ValueError("timestamps must have exactly the observed coordinates of X")
        if not np.isfinite(times.data).all():
            raise ValueError("timestamps must be finite")
        ages = float(cutoff) - times.data
        if np.any(ages < 0):
            raise ValueError("timestamps after the training cutoff are not allowed")
        half_life_seconds = float(self.half_life_days) * 86400.0
        observations.data = (self.floor + (1.0 - self.floor)
                             * np.exp2(-ages / half_life_seconds))
        self.target_summary_ = {
            "cutoff_epoch_seconds": float(cutoff),
            "half_life_days": float(self.half_life_days),
            "floor": float(self.floor),
            "mean_observed_target": float(observations.data.mean()) if observations.nnz else 0.0,
            "min_observed_target": float(observations.data.min()) if observations.nnz else 0.0,
            "max_observed_target": float(observations.data.max()) if observations.nnz else 0.0,
        }
        return self._fit_targets(observations)
