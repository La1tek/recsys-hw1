from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F


def _fm_interaction(summed_factors: torch.Tensor, squared_factors: torch.Tensor) -> torch.Tensor:
    return 0.5 * (summed_factors.square() - squared_factors).sum(dim=-1)


class _TorchFM(nn.Module):
    def __init__(self, n_users: int, n_items: int, item_features: sp.csr_matrix, factors: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_factors = nn.Embedding(n_users, factors)
        self.item_factors = nn.Embedding(n_items, factors)
        for layer in (self.user_bias, self.item_bias):
            nn.init.zeros_(layer.weight)
        for layer in (self.user_factors, self.item_factors):
            nn.init.normal_(layer.weight, std=0.02)
        self.metadata_bias = nn.Embedding(item_features.shape[1] + 1, 1, padding_idx=0)
        self.metadata_factors = nn.Embedding(item_features.shape[1] + 1, factors, padding_idx=0)
        row_lengths = np.diff(item_features.indptr)
        width = max(int(row_lengths.max(initial=0)), 1)
        indices = np.zeros((n_items, width), dtype=np.int64)
        values = np.zeros((n_items, width), dtype=np.float32)
        for item in range(n_items):
            start, end = item_features.indptr[item:item + 2]
            length = end - start
            indices[item, :length] = item_features.indices[start:end] + 1
            values[item, :length] = item_features.data[start:end]
        self.register_buffer("metadata_indices", torch.from_numpy(indices))
        self.register_buffer("metadata_values", torch.from_numpy(values))
        nn.init.zeros_(self.metadata_bias.weight)
        nn.init.normal_(self.metadata_factors.weight, std=0.02)
        with torch.no_grad():
            self.metadata_factors.weight[0].zero_()

    def item_components(self, item_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_ids = self.metadata_indices[item_ids]
        feature_values = self.metadata_values[item_ids]
        metadata_vectors = self.metadata_factors(feature_ids) * feature_values.unsqueeze(-1)
        item_vectors = self.item_factors(item_ids)
        summed_vectors = item_vectors + metadata_vectors.sum(dim=1)
        squared_vectors = item_vectors.square() + metadata_vectors.square().sum(dim=1)
        linear = self.item_bias(item_ids).squeeze(-1)
        linear = linear + (self.metadata_bias(feature_ids).squeeze(-1) * feature_values).sum(dim=1)
        return linear + _fm_interaction(summed_vectors, squared_vectors), summed_vectors

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        item_scores, item_vectors = self.item_components(item_ids)
        cross = (self.user_factors(user_ids) * item_vectors).sum(dim=-1)
        return self.bias + self.user_bias(user_ids).squeeze(-1) + item_scores + cross


class FactorizationMachine:
    def __init__(self, factors=16, learning_rate=0.01, regularization=1e-5,
                 epochs=8, negative_ratio=2, batch_size=2048, seed=42):
        if factors < 1 or learning_rate <= 0 or regularization < 0:
            raise ValueError("Invalid FM dimensionality, learning rate or regularization")
        if epochs < 1 or negative_ratio < 1 or int(negative_ratio) != negative_ratio or batch_size < 1:
            raise ValueError("epochs, negative_ratio and batch_size must be positive integers")
        self.factors = int(factors)
        self.learning_rate = float(learning_rate)
        self.regularization = float(regularization)
        self.epochs = int(epochs)
        self.negative_ratio = int(negative_ratio)
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    def _sample_negative_items(self, user_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        user_ids = np.asarray(user_ids, dtype=np.int64)
        if np.any(np.diff(self._train_csr.indptr)[user_ids] >= self.n_items_):
            raise ValueError("Cannot draw an unseen negative for a user who has seen every item")
        candidates = rng.integers(0, self.n_items_, size=len(user_ids), dtype=np.int64)
        for _ in range(32):
            collisions = np.asarray(self._train_csr[user_ids, candidates]).ravel() != 0
            if not collisions.any():
                return candidates
            candidates[collisions] = rng.integers(0, self.n_items_, size=int(collisions.sum()))
        # Для плотных историй выбираем из точного дополнения.
        collisions = np.asarray(self._train_csr[user_ids, candidates]).ravel() != 0
        for user in np.unique(user_ids[collisions]):
            mask = collisions & (user_ids == user)
            known = self._train_csr.indices[self._train_csr.indptr[user]:self._train_csr.indptr[user + 1]]
            available = np.setdiff1d(np.arange(self.n_items_), known, assume_unique=True)
            candidates[mask] = rng.choice(available, size=int(mask.sum()), replace=True)
        return candidates

    def fit(self, X: sp.csr_matrix, item_features: sp.csr_matrix):
        X = sp.csr_matrix(X, dtype=np.float32, copy=True)
        X.sum_duplicates()
        X.eliminate_zeros()
        X.sort_indices()
        if X.shape[0] < 1 or X.shape[1] < 2 or X.nnz == 0:
            raise ValueError("Training matrix must contain interactions and at least two items")
        if not np.isfinite(X.data).all() or np.any(X.data < 0):
            raise ValueError("Training interactions must be finite and nonnegative")
        X.data[:] = 1.0
        item_features = sp.csr_matrix(item_features, dtype=np.float32, copy=True)
        item_features.sum_duplicates()
        item_features.eliminate_zeros()
        if item_features.shape[0] != X.shape[1] or not np.isfinite(item_features.data).all():
            raise ValueError("Metadata must have one finite-valued row per training item")
        self.n_users_, self.n_items_ = X.shape
        self._train_csr = X
        rng = np.random.default_rng(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model_ = _TorchFM(self.n_users_, self.n_items_, item_features, self.factors)
        regularized = [parameter for name, parameter in self.model_.named_parameters() if name != "bias"]
        optimizer = torch.optim.Adam([
            {"params": regularized, "weight_decay": self.regularization},
            {"params": [self.model_.bias], "weight_decay": 0.0},
        ], lr=self.learning_rate)
        positive_users = np.repeat(np.arange(self.n_users_, dtype=np.int64), np.diff(X.indptr))
        positive_items = X.indices.astype(np.int64, copy=True)
        eligible = np.diff(X.indptr)[positive_users] < self.n_items_
        negative_users = np.repeat(positive_users[eligible], self.negative_ratio)
        if len(negative_users) == 0:
            raise ValueError("At least one user must have an unseen item for negative sampling")
        users = np.concatenate([positive_users, negative_users])
        labels = np.concatenate([np.ones(X.nnz, dtype=np.float32), np.zeros(len(negative_users), dtype=np.float32)])
        users_tensor = torch.from_numpy(users)
        labels_tensor = torch.from_numpy(labels)
        self.history_ = []
        self.model_.train()
        for epoch in range(self.epochs):
            negative_items = self._sample_negative_items(negative_users, rng)
            items_tensor = torch.from_numpy(np.concatenate([positive_items, negative_items]))
            order = rng.permutation(len(users))
            loss_sum = 0.0
            for start in range(0, len(order), self.batch_size):
                positions = torch.from_numpy(order[start:start + self.batch_size])
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(users_tensor[positions], items_tensor[positions])
                loss = F.binary_cross_entropy_with_logits(logits, labels_tensor[positions])
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(positions)
            self.history_.append({"epoch": epoch + 1, "avg_loss": loss_sum / len(users)})
        self.model_.eval()
        with torch.no_grad():
            item_scores, item_vectors = [], []
            for start in range(0, self.n_items_, self.batch_size):
                ids = torch.arange(start, min(start + self.batch_size, self.n_items_))
                scores, vectors = self.model_.item_components(ids)
                item_scores.append(scores)
                item_vectors.append(vectors)
            self._item_scores = torch.cat(item_scores)
            self._item_vectors = torch.cat(item_vectors)
        return self

    def score(self, user_ids) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before score")
        user_ids = np.asarray(user_ids, dtype=np.int64).reshape(-1)
        if np.any(user_ids < 0) or np.any(user_ids >= self.n_users_):
            raise ValueError("Unknown user ID")
        if len(user_ids) == 0:
            return np.empty((0, self.n_items_), dtype=np.float32)
        result = np.empty((len(user_ids), self.n_items_), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(user_ids), self.batch_size):
                ids = torch.from_numpy(user_ids[start:start + self.batch_size])
                scores = self.model_.user_factors(ids) @ self._item_vectors.T
                scores += self.model_.user_bias(ids) + self.model_.bias + self._item_scores.unsqueeze(0)
                result[start:start + len(ids)] = scores.numpy()
        return result
