"""Shared, persisted query splits for fair router comparisons."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable, Tuple


def _qid_fingerprint(qids: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(qids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_manifest(manifest: dict, qids: set[str], seed: int) -> Tuple[set[str], set[str], set[str]]:
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError(
            f"Split manifest seed={manifest.get('seed')} does not match requested seed={seed}."
        )

    train = set(manifest.get("train_qids", []))
    val = set(manifest.get("val_qids", []))
    test = set(manifest.get("test_qids", []))

    if train & val or train & test or val & test:
        raise ValueError("Split manifest contains overlapping train/validation/test qids.")
    if train | val | test != qids:
        missing = sorted(qids - (train | val | test))
        extra = sorted((train | val | test) - qids)
        raise ValueError(
            "Split manifest qids do not match the dataset "
            f"(missing={missing[:5]}, extra={extra[:5]})."
        )
    if manifest.get("qid_fingerprint") != _qid_fingerprint(qids):
        raise ValueError("Split manifest fingerprint does not match the dataset.")
    if not train or not val or not test:
        raise ValueError("Train, validation, and test splits must all be non-empty.")

    return train, val, test


def load_or_create_splits(
    qids: Iterable[str],
    manifest_path: Path,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[set[str], set[str], set[str]]:
    """Return one deterministic split shared by every method for a given seed."""
    qid_set = set(qids)
    if len(qid_set) < 3:
        raise ValueError("At least three unique qids are required.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must be positive and leave a non-empty test fraction.")

    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return _validate_manifest(manifest, qid_set, seed)

    ordered = sorted(qid_set)
    random.Random(seed).shuffle(ordered)
    n_train = int(len(ordered) * train_ratio)
    n_val = int(len(ordered) * val_ratio)

    train = set(ordered[:n_train])
    val = set(ordered[n_train : n_train + n_val])
    test = set(ordered[n_train + n_val :])

    manifest = {
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "qid_fingerprint": _qid_fingerprint(qid_set),
        "train_qids": sorted(train),
        "val_qids": sorted(val),
        "test_qids": sorted(test),
    }
    _validate_manifest(manifest, qid_set, seed)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return train, val, test
