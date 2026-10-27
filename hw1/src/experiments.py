from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
import torch
from threadpoolctl import threadpool_limits

from models.itemknn import ItemKNN
from models.ease import EASE
from models.ials import ImplicitALS
from models.custom_als import RecencyALS
from models.fm import FactorizationMachine
from src.data import build_item_features
from src.evaluation import MostPopular, evaluate, paired_bootstrap_difference

MAIN_MODELS = ("ItemKNN", "iALS", "EASE", "FM", "RecencyALS")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    paths = [root / "config.json", root / "data" / "prepared.npz",
             root / "data" / "prepared_meta.json", root / "src" / "data.py",
             root / "src" / "evaluation.py", root / "src" / "experiments.py"]
    paths += sorted((root / "models").glob("*.py"))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def matrices(data, include_validation=False):
    parts = ("train", "val") if include_validation else ("train",)
    users = np.concatenate([data[f"{part}_users"] for part in parts])
    items = np.concatenate([data[f"{part}_items"] for part in parts])
    times = np.concatenate([data[f"{part}_times"] for part in parts])
    shape = (data["n_users"], data["n_items"])
    X = sparse.csr_matrix((np.ones(len(users), np.float32), (users, items)), shape=shape)
    T = sparse.csr_matrix((times.astype(np.float64), (users, items)), shape=shape)
    X.sort_indices()
    T.sort_indices()
    if X.nnz != len(users):
        raise ValueError("Repeated user-item pairs survived preprocessing")
    return X, T


def truth_matrix(data, split):
    return sparse.csr_matrix(
        (np.ones(len(data[f"{split}_users"]), np.float32),
         (data[f"{split}_users"], data[f"{split}_items"])),
        shape=(data["n_users"], data["n_items"]),
    )


def create_model(name, params, seed):
    if name == "ItemKNN":
        return ItemKNN(**params)
    if name == "EASE":
        return EASE(**params)
    if name == "iALS":
        return ImplicitALS(**params, seed=seed)
    if name == "RecencyALS":
        return RecencyALS(**params, seed=seed)
    if name in ("FM", "FM_ID"):
        return FactorizationMachine(**params, seed=seed)
    if name == "MostPopular":
        return MostPopular()
    raise ValueError(name)


def fit_model(name, params, X, T, features, cutoff, seed):
    model = create_model(name, params, seed)
    started = perf_counter()
    with threadpool_limits(limits=2):
        if name == "RecencyALS":
            model.fit(X, timestamps=T, cutoff=float(cutoff))
        elif name == "FM":
            model.fit(X, item_features=features)
        elif name == "FM_ID":
            model.fit(X, item_features=sparse.csr_matrix((X.shape[1], 0), dtype=np.float32))
        else:
            model.fit(X)
    return model, perf_counter() - started


def _trial(root, signature, name, index, params, X, T, truth, features, cutoff, seed, force):
    key = hashlib.sha256(json.dumps([signature, name, params], sort_keys=True).encode()).hexdigest()[:20]
    cache = root / ".cache" / f"validation_{name}_{key}.json"
    if cache.exists() and not force:
        print(f"validation {name} #{index}: cached", flush=True)
        return read_json(cache)
    print(f"validation {name} #{index}: {params}", flush=True)
    model, duration = fit_model(name, params, X, T, features, cutoff, seed)
    started = perf_counter()
    with threadpool_limits(limits=2):
        metrics = evaluate(model, X, truth, k=10)
    row = {"model": name, "trial": index, "params": params, "fit_seconds": duration,
           "eval_seconds": perf_counter() - started, **metrics,
           "history": getattr(model, "history_", [])}
    write_json(cache, row)
    print(f"  NDCG@10={metrics['ndcg@10']:.6f}; Recall@10={metrics['recall@10']:.6f}; fit={duration:.1f}s", flush=True)
    return row


def run_validation(data, root, force=False):
    root = Path(root)
    torch.set_num_threads(2)
    config = read_json(root / "config.json")
    signature = fingerprint(root)
    X, T = matrices(data)
    truth = truth_matrix(data, "val")
    features = build_item_features(data["train_items"], data["train_times"],
                                   data["n_items"], data["cutoff_train"])
    rows, best = [], {}
    for name in ("ItemKNN", "EASE", "iALS", "FM"):
        family = []
        for index, params in enumerate(config["grids"][name], 1):
            row = _trial(root, signature, name, index, params, X, T, truth, features,
                         data["cutoff_train"], config["seed"], force)
            rows.append(row)
            family.append(row)
        winner = max(family, key=lambda row: row[config["primary_metric"]])
        best[name] = winner["params"]
    custom = []
    for index, half_life in enumerate(config["custom_half_lives_days"], 1):
        params = {**best["iALS"], "half_life_days": half_life,
                  "floor": config["custom_target_floor"]}
        row = _trial(root, signature, "RecencyALS", index, params, X, T, truth, features,
                     data["cutoff_train"], config["seed"], force)
        rows.append(row)
        custom.append(row)
    best["RecencyALS"] = max(custom, key=lambda row: row[config["primary_metric"]])["params"]
    best["MostPopular"] = {}
    best["FM_ID"] = dict(best["FM"])
    for name in ("MostPopular", "FM_ID"):
        rows.append(_trial(root, signature, name, 1, best[name], X, T, truth, features,
                           data["cutoff_train"], config["seed"], force))
    write_json(root / "artifacts" / "validation_results.json", rows)
    write_json(root / "artifacts" / "selected_parameters.json",
               {"fingerprint": signature, "primary_metric": config["primary_metric"],
                "params": best, "test_used_for_selection": False})
    return rows, best


def run_final(data, root, force=False):
    root = Path(root)
    torch.set_num_threads(2)
    config = read_json(root / "config.json")
    signature = fingerprint(root)
    selection = read_json(root / "artifacts" / "selected_parameters.json")
    if selection["fingerprint"] != signature:
        raise ValueError("Source/data/config changed: rerun validation before final evaluation")
    final_path = root / "artifacts" / "final_results.json"
    manifest_path = root / "artifacts" / "final_manifest.json"
    if final_path.exists() and manifest_path.exists() and not force:
        if read_json(manifest_path)["fingerprint"] == signature:
            return read_json(final_path)
    X, T = matrices(data, include_validation=True)
    truth = truth_matrix(data, "test")
    features = build_item_features(
        np.concatenate([data["train_items"], data["val_items"]]),
        np.concatenate([data["train_times"], data["val_times"]]),
        data["n_items"], data["cutoff_val"],
    )
    rows, details_by_model = [], {}
    names = (*MAIN_MODELS, "MostPopular", "FM_ID")
    for name in names:
        params = selection["params"][name]
        print(f"final train+validation: {name} {params}", flush=True)
        model, duration = fit_model(name, params, X, T, features, data["cutoff_val"], config["seed"])
        started = perf_counter()
        with threadpool_limits(limits=2):
            metrics, details = evaluate(model, X, truth, k=10, return_details=True)
        details_by_model[name] = details
        row = {"model": name, "params": params, "fit_seconds": duration,
               "eval_seconds": perf_counter() - started, **metrics,
               "history": getattr(model, "history_", [])}
        if hasattr(model, "target_summary_"):
            row["target_summary"] = model.target_summary_
        rows.append(row)
        print(f"  test NDCG@10={metrics['ndcg@10']:.6f}; Recall@10={metrics['recall@10']:.6f}", flush=True)
        if name in ("EASE", "RecencyALS"):
            _save_examples(data, model, X, truth, details, name, root)
    write_json(final_path, rows)
    comparisons = []
    for first, second in (("RecencyALS", "iALS"), ("FM", "FM_ID"), ("EASE", "MostPopular")):
        a, b = details_by_model[first], details_by_model[second]
        if not np.array_equal(a["users"], b["users"]):
            raise AssertionError("Models were evaluated on different users")
        comparisons.append({"comparison": f"{first} - {second}", "metric": "ndcg@10",
                            **paired_bootstrap_difference(a["ndcg"], b["ndcg"], seed=config["seed"])})
    write_json(root / "artifacts" / "paired_comparisons.json", comparisons)
    write_json(manifest_path, {"fingerprint": signature, "final_refit": "train+validation",
                              "test_used_for_selection": False, "models": list(names)})
    return rows


def _save_examples(data, model, history, truth, details, model_name, root):
    examples = []
    for offset, user in enumerate(details["users"][:3]):
        previous = history.indices[history.indptr[user]:history.indptr[user + 1]]
        relevant = set(truth.indices[truth.indptr[user]:truth.indptr[user + 1]].tolist())
        recommendations = details["recommendations"][offset]
        examples.append({"user_index": int(user),
                         "history": [data["names"][int(item)] for item in previous[-15:]],
                         "held_out_relevant": [data["names"][int(item)] for item in sorted(relevant)],
                         "top10": [{"game": data["names"][int(item)], "hit": int(item) in relevant}
                                   for item in recommendations if item >= 0]})
    write_json(root / "artifacts" / f"examples_{model_name}.json", examples)


def generate_figures(data, root):
    from src.reporting import make_figures
    root = Path(root)
    return make_figures(data, read_json(root / "artifacts" / "validation_results.json"),
                        read_json(root / "artifacts" / "final_results.json"), root)
