import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def metrics(targets, predictions, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (targets, predictions), 1)
    true_positive = np.diag(matrix).astype(float)
    precision = np.divide(true_positive, matrix.sum(0), out=np.zeros(num_classes), where=matrix.sum(0) > 0)
    recall = np.divide(true_positive, matrix.sum(1), out=np.zeros(num_classes), where=matrix.sum(1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(num_classes), where=precision + recall > 0)
    return {"accuracy": float(true_positive.sum() / matrix.sum()), "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()), "macro_f1": float(f1.mean()), "confusion_matrix": matrix.tolist(), "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(), "per_class_f1": f1.tolist()}


def select_device(value):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device
