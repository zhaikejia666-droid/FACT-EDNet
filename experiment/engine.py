import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from fact_ednet.losses import HybridObjective
from fact_ednet import FACTEDNet
from .data import load_datasets
from .models import build_model
from .utils import metrics, seed_everything, seed_worker, select_device, write_json


class TrainingBatchSampler(Sampler):
    def __init__(self, size, batch_size, generator):
        if size < 2 or batch_size < 2:
            raise ValueError("Training requires at least two samples and batch_size >= 2")
        self.size, self.batch_size, self.generator = size, batch_size, generator

    def __iter__(self):
        indices = torch.randperm(self.size, generator=self.generator).tolist()
        batches = [indices[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        yield from batches

    def __len__(self):
        count = (self.size + self.batch_size - 1) // self.batch_size
        return count - int(self.size % self.batch_size == 1 and count > 1)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"seq_len", "sample_interval", "signal_column", "time_column", "skip_rows", "batch_size", "epochs", "learning_rate", "min_learning_rate", "weight_decay", "lambda_freq", "lambda_sparse", "num_workers", "seeds", "model_options"}
    if set(config) != required:
        raise ValueError(f"Configuration keys differ: {set(config) ^ required}")
    if config["seq_len"] < 16 or config["batch_size"] < 2 or config["epochs"] < 1 or config["sample_interval"] <= 0 or config["num_workers"] < 0 or config["skip_rows"] < 0:
        raise ValueError("Invalid configuration")
    if not config["seeds"] or config["learning_rate"] <= 0 or not 0 <= config["min_learning_rate"] <= config["learning_rate"] or min(config["lambda_freq"], config["lambda_sparse"], config["weight_decay"]) < 0:
        raise ValueError("Invalid optimizer, loss, or seed configuration")
    return config


def make_loaders(datasets, config, seed):
    generator = torch.Generator().manual_seed(seed)
    training = DataLoader(datasets["train"], batch_sampler=TrainingBatchSampler(len(datasets["train"]), config["batch_size"], generator), num_workers=config["num_workers"], worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed + 1))
    validation = DataLoader(datasets["val"], batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed + 2))
    return training, validation


def run_epoch(model, loader, device, objective=None, optimizer=None):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    targets, predictions = [], []
    components = {"classification": 0.0, "frequency": 0.0, "sparsity": 0.0}
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, dt, pressure in loader:
            x, y, dt, pressure = x.to(device), y.to(device), dt.to(device), pressure.to(device)
            if objective is not None:
                logits, hidden = model(x, dt, return_features=True)
                loss, terms = objective(model, logits, hidden, pressure, y, dt)
            else:
                logits = model(x, dt) if isinstance(model, FACTEDNet) else model(x)
                loss = F.cross_entropy(logits, y)
                terms = {"classification": loss.detach()}
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            count += len(y)
            total += loss.item() * len(y)
            for key, value in terms.items():
                components[key] += value.item() * len(y)
            targets.extend(y.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
    result = metrics(targets, predictions, logits.shape[-1])
    result["loss"] = total / count
    result.update({key: value / count for key, value in components.items()})
    return result


def train_model(name, manifest, data_root, config, output, seed, device="auto"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    seed_everything(seed)
    device = select_device(device)
    datasets, classes, normalization, fingerprint = load_datasets(manifest, data_root, config)
    model = build_model(name, config["seq_len"], len(classes), config["model_options"]).to(device)
    training, validation = make_loaders(datasets, config, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=config["min_learning_rate"])
    objective = HybridObjective(config["lambda_freq"], config["lambda_sparse"]) if name == "fact_ednet" else None
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"model": name, "config": config, "seed": seed, "classes": classes, "normalization": normalization, "dataset_fingerprint": fingerprint, "parameters": sum(p.numel() for p in model.parameters()), "python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device), "cuda": torch.version.cuda, "selection": "validation macro-F1; validation loss breaks ties"}
    write_json(output / "run.json", metadata)
    best = (-1.0, -float("inf"))
    start = time.perf_counter()
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "learning_rate", "train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_macro_f1", "train_classification", "train_frequency", "train_sparsity"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config["epochs"] + 1):
            learning_rate = optimizer.param_groups[0]["lr"]
            train_result = run_epoch(model, training, device, objective, optimizer)
            val_result = run_epoch(model, validation, device, objective)
            scheduler.step(val_result["macro_f1"])
            writer.writerow({"epoch": epoch, "learning_rate": learning_rate, "train_loss": train_result["loss"], "val_loss": val_result["loss"], "train_accuracy": train_result["accuracy"], "val_accuracy": val_result["accuracy"], "val_macro_f1": val_result["macro_f1"], "train_classification": train_result["classification"], "train_frequency": train_result["frequency"], "train_sparsity": train_result["sparsity"]})
            handle.flush()
            ranking = (val_result["macro_f1"], -val_result["loss"])
            if ranking > best:
                best = ranking
                state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save({"state_dict": state, "metadata": metadata, "epoch": epoch, "validation": val_result}, output / "best.pt")
            print(f"{name} seed={seed} epoch={epoch}/{config['epochs']} train_loss={train_result['loss']:.6f} val_macro_f1={val_result['macro_f1']:.6f}", flush=True)
    write_json(output / "training_summary.json", {"elapsed_seconds": time.perf_counter() - start, "best_validation_macro_f1": best[0]})
    return output / "best.pt"


def evaluate_model(checkpoint, manifest, data_root, output=None, device="auto"):
    if output is not None and Path(output).exists():
        raise FileExistsError(output)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    config = metadata["config"]
    seed_everything(metadata["seed"])
    device = select_device(device)
    datasets, classes, _, fingerprint = load_datasets(manifest, data_root, config, metadata["normalization"], metadata["classes"])
    if fingerprint != metadata["dataset_fingerprint"]:
        raise ValueError("Dataset contents or split manifest changed since training")
    model = build_model(metadata["model"], config["seq_len"], len(classes), config["model_options"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    loader = DataLoader(datasets["test"], batch_size=config["batch_size"], shuffle=False)
    result = run_epoch(model, loader, device)
    result.update({"model": metadata["model"], "seed": metadata["seed"], "classes": classes, "selected_epoch": saved["epoch"], "dataset_fingerprint": fingerprint})
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        write_json(output, result)
    return result
