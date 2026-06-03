"""
Shared helpers for BioVerify: seeding, device, class weights,
checkpointing, metrics tracking, timing, and logging.
"""

import csv
import json
import logging
import os
import pickle
import random
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all relevant RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slight perf cost — acceptable for research)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[utils] Device: {device}" + (
        f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    ))
    return device


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights_tensor(
    labels: List[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights and return as a float tensor
    on the given device.

    Uses sklearn compute_class_weight('balanced', ...).
    """
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(labels),
    )
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[utils] Class weights: {weights.tolist()}")
    return tensor


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    path: str,
) -> None:
    """Save model + optimizer state + metadata to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, Any]:
    """Load checkpoint into model (and optionally optimizer). Returns metadata dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[utils] Loaded checkpoint from '{path}'  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


def save_tfidf_checkpoint(pipeline: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)


# ---------------------------------------------------------------------------
# Metric tracking
# ---------------------------------------------------------------------------

class AverageMeter:
    """Running mean of a scalar quantity (e.g. loss per batch)."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Context manager for wall-clock timing."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.time() - self.start
        if self.label:
            print(f"[timer] {self.label}: {self.elapsed:.2f}s")


# ---------------------------------------------------------------------------
# CSV Logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """Append-mode CSV writer for epoch-level training logs."""

    COLUMNS = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_contradiction_f1",
        "learning_rate",
    ]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow({k: row.get(k, "") for k in self.COLUMNS})


def setup_logging(log_dir: str) -> CSVLogger:
    """Create log directory and return a CSVLogger for train_log.csv."""
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "train_log.csv")
    logger = CSVLogger(csv_path)
    print(f"[utils] CSV log → {csv_path}")
    return logger


# ---------------------------------------------------------------------------
# Config snapshot
# ---------------------------------------------------------------------------

def save_config_snapshot(config_dict: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)
