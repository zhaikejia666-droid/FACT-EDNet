import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_manifest(path, data_root):
    root = Path(data_root).resolve(strict=True)
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "group", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires path,label,group,split columns")
        records = list(reader)
    if not records:
        raise ValueError("Manifest is empty")
    seen_paths, groups = set(), {}
    for record in records:
        if any(not record[key].strip() for key in ("path", "label", "group", "split")):
            raise ValueError("Manifest fields cannot be empty")
        relative = Path(record["path"])
        full = (root / relative).resolve(strict=True)
        if relative.is_absolute() or not full.is_relative_to(root) or not full.is_file():
            raise ValueError("Sample paths must identify files within data_root")
        if full in seen_paths:
            raise ValueError(f"Duplicate sample path: {relative}")
        seen_paths.add(full)
        if record["split"] not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        previous = groups.setdefault(record["group"], record["split"])
        if previous != record["split"]:
            raise ValueError(f"Group leakage: {record['group']}")
        record["full_path"] = full
    for split in ("train", "val", "test"):
        if not any(r["split"] == split for r in records):
            raise ValueError(f"Missing {split} split")
    classes = sorted({r["label"] for r in records if r["split"] == "train"})
    if len(classes) < 2 or any(r["label"] not in classes for r in records):
        raise ValueError("Training split must contain all classes and at least two classes")
    return records, classes


def read_signal(path, config):
    array = np.loadtxt(path, delimiter=",", skiprows=config["skip_rows"], ndmin=2, dtype=np.float64)
    length = config["seq_len"]
    column = config["signal_column"]
    if array.shape[0] < length or column < 0 or column >= array.shape[1]:
        raise ValueError(f"Invalid signal column or fewer than {length} rows: {path}")
    signal = array[:length, column].astype(np.float32)[:, None]
    interval = np.full(length, config["sample_interval"], dtype=np.float32)
    if config["time_column"] is not None:
        time_column = config["time_column"]
        if time_column < 0 or time_column >= array.shape[1] or time_column == column:
            raise ValueError("Invalid time_column")
        times = array[:length, time_column]
        steps = np.diff(times)
        if not np.isfinite(times).all() or np.any(steps <= 0):
            raise ValueError(f"Timestamps must be finite and strictly increasing: {path}")
        interval[1:] = steps
        interval[0] = steps[0]
    if not np.isfinite(signal).all() or not np.isfinite(interval).all() or np.any(interval <= 0):
        raise ValueError(f"Non-finite signal or invalid interval: {path}")
    if not np.allclose(interval[1:], interval[1], rtol=1e-4, atol=1e-7):
        raise ValueError(f"Comparison FFT protocol requires uniform timestamps: {path}")
    return signal, interval


class SignalDataset(Dataset):
    def __init__(self, x, y, dt, mean, std):
        self.x = torch.from_numpy(((x - mean) / std).astype(np.float32))
        self.raw = torch.from_numpy(x.astype(np.float32))
        self.y = torch.tensor(y, dtype=torch.long)
        self.dt = torch.from_numpy(dt.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.dt[index], self.raw[index]


def load_datasets(manifest, data_root, config, normalization=None, expected_classes=None):
    records, classes = read_manifest(manifest, data_root)
    if expected_classes is not None and classes != expected_classes:
        raise ValueError("Dataset classes differ from checkpoint classes")
    mapping = {name: index for index, name in enumerate(classes)}
    arrays = {split: {"x": [], "y": [], "dt": []} for split in ("train", "val", "test")}
    fingerprints, identity = {}, hashlib.sha256()
    for record in records:
        signal, interval = read_signal(record["full_path"], config)
        fingerprint = hashlib.sha256(signal.tobytes()).hexdigest()
        previous = fingerprints.setdefault(fingerprint, record["split"])
        if previous != record["split"]:
            raise ValueError(f"Identical cropped signals across splits: {record['path']}")
        identity.update((record["path"] + record["label"] + record["group"] + record["split"] + fingerprint).encode())
        identity.update(interval.tobytes())
        bucket = arrays[record["split"]]
        bucket["x"].append(signal)
        bucket["y"].append(mapping[record["label"]])
        bucket["dt"].append(interval)
    if normalization is None:
        training = np.stack(arrays["train"]["x"]).astype(np.float64)
        normalization = {"mean": float(training.mean()), "std": max(float(training.std()), 1e-8)}
    if not np.isfinite(list(normalization.values())).all() or normalization["std"] <= 0:
        raise ValueError("Invalid normalization")
    datasets = {
        split: SignalDataset(np.stack(bucket["x"]), bucket["y"], np.stack(bucket["dt"]), **normalization)
        for split, bucket in arrays.items()
    }
    return datasets, classes, normalization, identity.hexdigest()
