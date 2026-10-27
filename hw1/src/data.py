from __future__ import annotations

import ast
import csv
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from scipy import sparse


PIPELINE_VERSION = 1
SOURCES = {
    "steam_reviews.json.gz": 1338063248,
    "steam_games.json.gz": 2740516,
}


def download(root):
    directory = Path(root) / "data"
    directory.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("Install curl, then run again. Ubuntu/Debian: sudo apt install curl ca-certificates")
    for filename, size in SOURCES.items():
        target = directory / filename
        if not target.exists() or target.stat().st_size != size:
            subprocess.run(
                [curl, "--location", "--fail", "--show-error", "--continue-at", "-",
                 "--retry", "3", "--output", str(target),
                 f"https://cseweb.ucsd.edu/~wckang/{filename}"],
                check=True,
            )
        print(f"Ready: {filename}, {target.stat().st_size:,} bytes", flush=True)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _source_signature(path: Path) -> dict:
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _parse_date(value) -> int | None:
    if not isinstance(value, str):
        return None
    try:
        day = date.fromisoformat(value.strip()[:10])
    except (ValueError, OverflowError):
        return None
    return (day - date(1970, 1, 1)).days * 86400


def _iso(timestamp: int) -> str:
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()


def _read_metadata(path: Path) -> tuple[dict, dict]:
    metadata = {}
    stats = {"raw_rows": 0, "parse_errors": 0, "invalid_item_id": 0, "duplicates": 0}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            stats["raw_rows"] += 1
            try:
                row = ast.literal_eval(line)
            except (ValueError, SyntaxError, TypeError, MemoryError):
                stats["parse_errors"] += 1
                continue
            if not isinstance(row, dict):
                stats["parse_errors"] += 1
                continue
            item = str(row.get("id", "")).strip()
            if not item.isdigit():
                stats["invalid_item_id"] += 1
                continue
            if item in metadata:
                stats["duplicates"] += 1
                continue
            metadata[item] = {
                "id": item,
                "name": str(row.get("app_name") or row.get("title") or item),
                **{field: row.get(field) for field in ("genres", "developer", "publisher", "specs")},
            }
    stats["unique_items"] = len(metadata)
    return metadata, stats


def _compact_reviews(root: Path, force: bool = False) -> tuple[Path, dict]:
    raw_path = root / "data" / "steam_reviews.json.gz"
    csv_path = root / "data" / "reviews_compact.csv"
    stat_path = root / "data" / "reviews_compact_stats.json"
    signature = _source_signature(raw_path)
    if not force and csv_path.exists() and stat_path.exists():
        cached = json.loads(stat_path.read_text(encoding="utf-8"))
        if cached.get("source_signature") == signature and cached.get("pipeline_version") == PIPELINE_VERSION:
            return csv_path, cached
    started = time.perf_counter()
    stable_id = re.compile(r"['\"]user_id['\"]\s*:\s*(?:u)?['\"]?(\d{17})(?!\d)")
    stats = {
        "raw_seen": 0,
        "excluded_missing_stable_id": 0,
        "excluded_parse_error": 0,
        "excluded_invalid_stable_id": 0,
        "excluded_invalid_item": 0,
        "excluded_invalid_date": 0,
        "valid_rows": 0,
        "missing_or_invalid_hours": 0,
        "pipeline_version": PIPELINE_VERSION,
        "source_signature": signature,
    }
    temporary = csv_path.with_suffix(".csv.partial")
    with gzip.open(raw_path, "rt", encoding="utf-8") as source, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as destination:
        writer = csv.writer(destination)
        writer.writerow(["user", "item", "timestamp", "hours"])
        for line in source:
            stats["raw_seen"] += 1
            if stats["raw_seen"] % 500_000 == 0:
                print(
                    f"reviews: {stats['raw_seen']:,} raw; {stats['valid_rows']:,} valid stable-ID rows; "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            if not stable_id.search(line):
                stats["excluded_missing_stable_id"] += 1
                continue
            try:
                row = ast.literal_eval(line)
            except (ValueError, SyntaxError, TypeError, MemoryError):
                stats["excluded_parse_error"] += 1
                continue
            if not isinstance(row, dict):
                stats["excluded_parse_error"] += 1
                continue
            user = str(row.get("user_id", "")).strip()
            if len(user) != 17 or not user.isascii() or not user.isdigit():
                stats["excluded_invalid_stable_id"] += 1
                continue
            item = str(row.get("product_id", "")).strip()
            if not item.isascii() or not item.isdigit():
                stats["excluded_invalid_item"] += 1
                continue
            timestamp = _parse_date(row.get("date"))
            if timestamp is None:
                stats["excluded_invalid_date"] += 1
                continue
            try:
                hours = float(row.get("hours", "nan"))
            except (ValueError, TypeError):
                hours = float("nan")
            if not np.isfinite(hours) or hours < 0:
                hours = float("nan")
                stats["missing_or_invalid_hours"] += 1
            pseudonym = hashlib.blake2b(
                user.encode("ascii"), digest_size=12, person=b"steam-hw1"
            ).hexdigest()
            writer.writerow([pseudonym, item, timestamp, hours])
            stats["valid_rows"] += 1
    temporary.replace(csv_path)
    stats["compact_seconds"] = time.perf_counter() - started
    _write_json(stat_path, stats)
    print(f"compact cache: {stats['valid_rows']:,} rows in {stats['compact_seconds']:.1f}s", flush=True)
    return csv_path, stats


def load_prepared(rootpath: str | Path) -> dict:
    root = Path(rootpath)
    metadata = json.loads((root / "data" / "prepared_meta.json").read_text(encoding="utf-8"))
    if metadata.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("Prepared data has an incompatible pipeline version")
    with np.load(root / "data" / "prepared.npz", allow_pickle=False) as archive:
        metadata.update({name: archive[name] for name in archive.files})
    return metadata


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    points = np.quantile(values, [0, .25, .5, .75, .9, .99, 1])
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        **{key: float(value) for key, value in zip(
            ("min", "q25", "median", "q75", "q90", "q99", "max"), points
        )},
    }


def prepare_data(
    rootpath: str | Path,
    max_items: int = 3500,
    min_train_user: int = 3,
    max_users: int | None = 20_000,
    train_fraction: float = 0.70,
    validation_end_fraction: float = 0.85,
    force: bool = False,
    check_requirements: bool = True,
) -> dict:
    started = time.perf_counter()
    root = Path(rootpath)
    (root / "data").mkdir(parents=True, exist_ok=True)
    if max_items < 1 or min_train_user < 1 or (max_users is not None and max_users < 1):
        raise ValueError("max_items, min_train_user and max_users must be positive")
    if not 0 < train_fraction < validation_end_fraction < 1:
        raise ValueError("Expected 0 < train_fraction < validation_end_fraction < 1")
    params = {
        "max_items": int(max_items), "min_train_user": int(min_train_user),
        "max_users": None if max_users is None else int(max_users),
        "train_fraction": float(train_fraction),
        "validation_end_fraction": float(validation_end_fraction),
        "check_requirements": bool(check_requirements),
    }
    signatures = {
        name: _source_signature(root / "data" / name)
        for name in ("steam_reviews.json.gz", "steam_games.json.gz")
    }
    if not force and (root / "data" / "prepared.npz").exists() and (root / "data" / "prepared_meta.json").exists():
        previous = json.loads((root / "data" / "prepared_meta.json").read_text(encoding="utf-8"))
        if previous.get("params") == params and previous.get("source_signatures") == signatures and previous.get("pipeline_version") == PIPELINE_VERSION:
            return load_prepared(root)

    metadata, metadata_stats = _read_metadata(root / "data" / "steam_games.json.gz")
    csv_path, raw_stats = _compact_reviews(root)
    frame = pd.read_csv(
        csv_path, dtype={"user": str, "item": str, "timestamp": np.int64, "hours": np.float32}
    )
    if frame.empty:
        raise ValueError("No valid interactions with explicit stable user IDs")
    frame = frame.sort_values("timestamp", kind="mergesort").drop_duplicates(["user", "item"], keep="first")
    deduplicated_count = len(frame)
    duplicate_count = int(raw_stats["valid_rows"] - deduplicated_count)
    boundaries = frame["timestamp"].quantile(
        [train_fraction, validation_end_fraction], interpolation="nearest"
    ).to_numpy(dtype=np.int64)
    first_cutoff, second_cutoff = map(int, boundaries)
    if first_cutoff >= second_cutoff:
        raise ValueError("The data does not have enough distinct dates for a chronological split")
    train_mask = frame["timestamp"] < first_cutoff
    eligible_train = frame.loc[train_mask & frame["item"].isin(metadata)]
    popularity = eligible_train.groupby("item", sort=False).size().rename("count").reset_index()
    popularity["numeric_id"] = popularity["item"].astype(np.int64)
    popularity = popularity.sort_values(["count", "numeric_id"], ascending=[False, True], kind="mergesort")
    initial_catalog = set(popularity["item"].head(max_items))
    in_catalog = frame["item"].isin(initial_catalog)
    catalog_train = frame.loc[train_mask & in_catalog]
    activity = catalog_train.groupby("user", sort=False).size().rename("count").reset_index()
    activity = activity.loc[activity["count"] >= min_train_user]
    activity = activity.sort_values(["count", "user"], ascending=[False, True], kind="mergesort")
    if max_users is not None:
        activity = activity.head(max_users)
    selected_users = set(activity["user"])
    retained = frame.loc[in_catalog & frame["user"].isin(selected_users)].copy()
    actual_catalog = set(retained.loc[retained["timestamp"] < first_cutoff, "item"])
    cold_catalog_rows = int((~retained["item"].isin(actual_catalog)).sum())
    retained = retained.loc[retained["item"].isin(actual_catalog)].copy()
    item_ids = sorted(actual_catalog, key=int)
    user_ids = sorted(selected_users)
    n_users, n_items = len(user_ids), len(item_ids)
    if not n_users or not n_items:
        raise ValueError("Training catalog/user filters removed all interactions")
    stats = {
        "raw": raw_stats,
        "metadata": metadata_stats,
        "deduplicated_stable_interactions": int(deduplicated_count),
        "excluded_duplicate_user_item": duplicate_count,
        "excluded_without_item_metadata": int((~frame["item"].isin(metadata)).sum()),
        "excluded_metadata_items_outside_training_catalog": int((frame["item"].isin(metadata) & ~in_catalog).sum()),
        "excluded_ineligible_users_within_catalog": int((in_catalog & ~frame["user"].isin(selected_users)).sum()),
        "excluded_items_without_selected_training_users": cold_catalog_rows,
        "selected_interactions": int(len(retained)),
        "selected_users": int(n_users), "selected_items": int(n_items),
        "density": float(len(retained) / (n_users * n_items)),
    }
    if check_requirements and not (100_000 <= len(retained) <= 2_000_000 and n_users >= 1000 and n_items >= 1000):
        _write_json(root / "artifacts" / "data_stats_failed.json", stats)
        raise ValueError(
            f"Assignment limits failed after cleaning: {len(retained):,} interactions, "
            f"{n_users:,} users, {n_items:,} items. Inspect artifacts/data_stats_failed.json "
            "and adjust training-only catalog/user parameters."
        )
    user_map = {value: index for index, value in enumerate(user_ids)}
    item_map = {value: index for index, value in enumerate(item_ids)}
    retained["user_idx"] = retained["user"].map(user_map)
    retained["item_idx"] = retained["item"].map(item_map)
    retained = retained.sort_values(["timestamp", "user_idx", "item_idx"], kind="mergesort")
    splits = {
        "train": retained["timestamp"] < first_cutoff,
        "val": (retained["timestamp"] >= first_cutoff) & (retained["timestamp"] < second_cutoff),
        "test": retained["timestamp"] >= second_cutoff,
    }
    arrays = {}
    split_stats = {}
    for name, mask in splits.items():
        part = retained.loc[mask]
        arrays[f"{name}_users"] = part["user_idx"].to_numpy(dtype=np.int32)
        arrays[f"{name}_items"] = part["item_idx"].to_numpy(dtype=np.int32)
        arrays[f"{name}_times"] = part["timestamp"].to_numpy(dtype=np.int64)
        arrays[f"{name}_hours"] = part["hours"].to_numpy(dtype=np.float32)
        split_stats[name] = {
            "interactions": int(len(part)), "users": int(part["user"].nunique()),
            "items": int(part["item"].nunique()),
            "first_utc": _iso(int(part["timestamp"].min())) if len(part) else None,
            "last_utc": _iso(int(part["timestamp"].max())) if len(part) else None,
        }
    arrays["train_user_counts"] = np.bincount(arrays["train_users"], minlength=n_users).astype(np.int32)
    arrays["train_item_counts"] = np.bincount(arrays["train_items"], minlength=n_items).astype(np.int32)
    assert (arrays["train_user_counts"] >= min_train_user).all()
    assert (arrays["train_item_counts"] > 0).all()
    assert sum(len(arrays[f"{part}_users"]) for part in splits) == len(retained)
    assert not len(arrays["val_times"]) or arrays["train_times"].max() < arrays["val_times"].min()
    assert not len(arrays["test_times"]) or arrays["train_times"].max() < arrays["test_times"].min()
    stats["splits"] = split_stats
    stats["prepare_seconds"] = float(time.perf_counter() - started)
    eda = {
        "train_user_activity": _summary(arrays["train_user_counts"]),
        "train_item_popularity": _summary(arrays["train_item_counts"]),
        "hours_descriptive_only": _summary(retained["hours"].to_numpy()),
        "top_train_items": [
            {"id": item_ids[int(index)], "name": metadata[item_ids[int(index)]]["name"], "count": int(arrays["train_item_counts"][index])}
            for index in np.argsort(-arrays["train_item_counts"], kind="stable")[:15]
        ],
    }
    prepared_metadata = {
        "pipeline_version": PIPELINE_VERSION, "params": params,
        "source_signatures": signatures, "n_users": n_users, "n_items": n_items,
        "item_ids": item_ids, "user_ids": user_ids,
        "cutoff_train": first_cutoff, "cutoff_val": second_cutoff,
        "names": [metadata[item]["name"] for item in item_ids],
        "item_metadata": [metadata[item] for item in item_ids],
        "stats": stats, "eda": eda,
    }
    np.savez_compressed(root / "data" / "prepared.npz", **arrays)
    _write_json(root / "data" / "prepared_meta.json", prepared_metadata)
    print(
        f"prepared: {len(retained):,} interactions, {n_users:,} users, {n_items:,} games; "
        f"splits {[split_stats[name]['interactions'] for name in splits]}", flush=True,
    )
    return {**prepared_metadata, **arrays}


def build_item_features(
    item_indices,
    timestamps,
    n_items: int,
    cutoff,
    return_names: bool = False,
):
    items, times = np.asarray(item_indices), np.asarray(timestamps)
    if items.ndim != 1 or times.ndim != 1 or items.shape != times.shape:
        raise ValueError("item_indices and timestamps must be equally sized one-dimensional arrays")
    if (not isinstance(n_items, (int, np.integer))
            or isinstance(n_items, (bool, np.bool_)) or n_items < 1):
        raise ValueError("n_items must be a positive integer")
    if items.size and (items.dtype.kind not in "iu" or np.any(items < 0) or np.any(items >= n_items)):
        raise ValueError("item_indices must contain integers in [0, n_items)")
    if times.dtype.kind not in "fiu" or not np.isfinite(times).all():
        raise ValueError("timestamps must contain finite epoch-second values")
    cutoff = np.asarray(cutoff)
    if cutoff.ndim != 0 or cutoff.dtype.kind not in "fiu" or not np.isfinite(cutoff):
        raise ValueError("cutoff must be a finite epoch-second scalar")

    historical = times < float(cutoff)
    items = items[historical].astype(np.int64, copy=False)
    times = times[historical].astype(np.float64, copy=False)
    counts = np.bincount(items, minlength=n_items)
    first_times = np.full(n_items, np.inf)
    np.minimum.at(first_times, items, times)
    count_bins = np.full(n_items, "zero", dtype=object)
    years = np.full(n_items, "unknown", dtype=object)
    observed = np.flatnonzero(counts)
    count_bins[observed] = np.floor(np.log2(counts[observed])).astype(np.int64).astype(str)
    for item in observed:
        years[item] = str(datetime.fromtimestamp(first_times[item], tz=timezone.utc).year)
    names = sorted({f"count_log2={value}" for value in count_bins}
                   | {f"first_review_year={value}" for value in years})
    lookup = {name: column for column, name in enumerate(names)}
    columns = np.column_stack([
        [lookup[f"count_log2={value}"] for value in count_bins],
        [lookup[f"first_review_year={value}"] for value in years],
    ]).ravel()
    features = sparse.csr_matrix(
        (np.ones(2 * n_items, dtype=np.float32),
         (np.repeat(np.arange(n_items), 2), columns)),
        shape=(n_items, len(names)), dtype=np.float32,
    )
    return (features, names) if return_names else features
