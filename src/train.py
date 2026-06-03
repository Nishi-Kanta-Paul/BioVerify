"""
Training pipeline for BioVerify.

  train_transformer(model_name, ...)  — fine-tune any HF encoder
  train_tfidf(...)                   — fit TF-IDF + LogReg baseline
  train_model(model_name, ...)       — dispatcher
"""

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import CONFIG, LABEL2ID, NUM_CLASSES
from src.dataset import (
    BioVerifyDataset,
    get_dataloaders,
    load_pubmedqa,
    prepare_tfidf_data,
    reformulate_dataset,
    split_dataset,
)
from src.model import TransformerClassifier, TFIDFClassifier, build_model, get_tokenizer
from src.utils import (
    AverageMeter,
    CSVLogger,
    Timer,
    compute_class_weights_tensor,
    get_device,
    load_checkpoint,
    save_checkpoint,
    save_config_snapshot,
    save_tfidf_checkpoint,
    set_seed,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _experiment_dir(model_name: str) -> str:
    return os.path.join(CONFIG.experiment.experiment_dir, model_name)


def _checkpoint_path(model_name: str) -> str:
    return os.path.join(_experiment_dir(model_name), "checkpoints", "best_model.pth")


def _pkl_checkpoint_path() -> str:
    return os.path.join(_experiment_dir("tfidf_lr"), "checkpoints", "best_model.pkl")


def _results_path(model_name: str) -> str:
    return os.path.join(_experiment_dir(model_name), "results.json")


def _compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Return accuracy, macro-F1, and contradiction-class F1."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    # contradiction is label index 1 (LABEL2ID["contradicted"] == 1)
    contra_f1 = f1_score(
        y_true, y_pred, labels=[LABEL2ID["contradicted"]], average="micro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "contradiction_f1": float(contra_f1),
    }


def _load_split_labels(processed_dir: str, split_file: str) -> List[int]:
    """Read label column from a split CSV and convert to int ids."""
    df = pd.read_csv(os.path.join(processed_dir, split_file))
    return df["label"].map(LABEL2ID).tolist()


# ---------------------------------------------------------------------------
# 1. Transformer training
# ---------------------------------------------------------------------------

def train_transformer(
    model_name: str = CONFIG.training.__class__.__name__,
    max_epochs: int = CONFIG.training.max_epochs,
    batch_size: int = CONFIG.training.batch_size,
    learning_rate: float = CONFIG.training.learning_rate,
    weight_decay: float = CONFIG.training.weight_decay,
    warmup_ratio: float = CONFIG.training.warmup_ratio,
    gradient_accumulation_steps: int = CONFIG.training.gradient_accumulation_steps,
    early_stopping_patience: int = CONFIG.training.early_stopping_patience,
    seed: int = CONFIG.training.seed,
    max_length: int = CONFIG.model.max_length,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = CONFIG.model.dropout_rate,
    max_train_steps: Optional[int] = None,   # if set, cap batches per epoch (fast dev run)
    max_val_steps: Optional[int] = None,     # if set, cap val batches per epoch
) -> Dict:
    """
    Fine-tune a transformer classifier on the BioVerify task.

    Returns a dict with best validation metrics.
    """
    set_seed(seed)
    device = get_device()

    # ---- Experiment directories ----
    exp_dir = _experiment_dir(model_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_logger = setup_logging(log_dir)

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"{'='*60}")

    # ---- Data ----
    train_loader, val_loader, _ = get_dataloaders(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )

    # Class weights from training labels
    train_labels = _load_split_labels(CONFIG.data.processed_dir, CONFIG.data.train_file)
    class_weights = compute_class_weights_tensor(train_labels, num_classes, device)

    # ---- Model ----
    model = build_model(model_name, num_classes=num_classes, dropout_rate=dropout_rate)
    model.to(device)

    # ---- Loss / Optimizer / Scheduler ----
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    effective_train_batches = min(len(train_loader), max_train_steps) if max_train_steps else len(train_loader)
    total_steps = (effective_train_batches // gradient_accumulation_steps) * max_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---- Save config snapshot ----
    config_snapshot = {
        "model_name": model_name,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "early_stopping_patience": early_stopping_patience,
        "seed": seed,
        "max_length": max_length,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
    save_config_snapshot(config_snapshot, os.path.join(log_dir, "training_config.json"))

    # ---- Training loop ----
    best_macro_f1 = -1.0
    patience_counter = 0
    best_metrics: Dict = {}

    with Timer("Total training") as total_timer:
        for epoch in range(1, max_epochs + 1):

            # -- Train phase --
            model.train()
            train_loss_meter = AverageMeter("train_loss")
            optimizer.zero_grad()

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}/{max_epochs} [train]",
                leave=False,
            )
            for step, batch in pbar:
                if max_train_steps and step >= max_train_steps:
                    break

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels) / gradient_accumulation_steps
                loss.backward()

                train_loss_meter.update(loss.item() * gradient_accumulation_steps, n=len(labels))

                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                pbar.set_postfix(loss=f"{train_loss_meter.avg:.4f}")

            # Handle any remaining accumulated gradients at end of epoch
            effective_steps = min(len(train_loader), max_train_steps) if max_train_steps else len(train_loader)
            if effective_steps % gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            current_lr = scheduler.get_last_lr()[0]

            # -- Validation phase --
            model.eval()
            val_loss_meter = AverageMeter("val_loss")
            all_preds: List[int] = []
            all_labels: List[int] = []

            with torch.no_grad():
                for val_step, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch}/{max_epochs} [val]", leave=False)):
                    if max_val_steps and val_step >= max_val_steps:
                        break
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    token_type_ids = batch.get("token_type_ids")
                    if token_type_ids is not None:
                        token_type_ids = token_type_ids.to(device)
                    labels = batch["label"].to(device)

                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(logits, labels)
                    val_loss_meter.update(loss.item(), n=len(labels))

                    preds = logits.argmax(dim=-1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            metrics = _compute_metrics(all_labels, all_preds)
            val_macro_f1 = metrics["macro_f1"]

            # -- Log --
            log_row = {
                "epoch": epoch,
                "train_loss": round(train_loss_meter.avg, 6),
                "val_loss": round(val_loss_meter.avg, 6),
                "val_accuracy": round(metrics["accuracy"], 6),
                "val_macro_f1": round(val_macro_f1, 6),
                "val_contradiction_f1": round(metrics["contradiction_f1"], 6),
                "learning_rate": round(current_lr, 8),
            }
            csv_logger.log(log_row)

            print(
                f"  Epoch {epoch:2d} | "
                f"train_loss={log_row['train_loss']:.4f}  "
                f"val_loss={log_row['val_loss']:.4f}  "
                f"acc={log_row['val_accuracy']:.4f}  "
                f"macro_f1={log_row['val_macro_f1']:.4f}  "
                f"contra_f1={log_row['val_contradiction_f1']:.4f}  "
                f"lr={current_lr:.2e}"
            )

            # -- Checkpoint + early stopping --
            if val_macro_f1 > best_macro_f1:
                best_macro_f1 = val_macro_f1
                best_metrics = {**metrics, "epoch": epoch}
                save_checkpoint(
                    model, optimizer, epoch, metrics, _checkpoint_path(model_name)
                )
                print(f"    ✓ New best macro-F1={best_macro_f1:.4f} — checkpoint saved")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
                    break

    # ---- Save results ----
    results = {
        "model_name": model_name,
        **best_metrics,
        "training_time_minutes": round(total_timer.elapsed / 60, 2),
        "config_snapshot": config_snapshot,
    }
    with open(_results_path(model_name), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Best val macro-F1: {best_macro_f1:.4f}  (epoch {best_metrics.get('epoch', '?')})")
    print(f"  Results → {_results_path(model_name)}")

    return best_metrics


# ---------------------------------------------------------------------------
# 2. TF-IDF training
# ---------------------------------------------------------------------------

def train_tfidf() -> Dict:
    """
    Fit TFIDFClassifier on training data and evaluate on validation set.
    Saves model checkpoint and results.json.
    """
    set_seed(CONFIG.training.seed)

    print(f"\n{'='*60}")
    print(f"  Training: TFIDF_LR")
    print(f"{'='*60}")

    processed_dir = CONFIG.data.processed_dir
    train_df = pd.read_csv(os.path.join(processed_dir, CONFIG.data.train_file))
    val_df = pd.read_csv(os.path.join(processed_dir, CONFIG.data.val_file))

    train_texts, train_labels = prepare_tfidf_data(train_df)
    val_texts, val_labels = prepare_tfidf_data(val_df)

    exp_dir = _experiment_dir("tfidf_lr")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with Timer("TF-IDF fit") as t:
        model = build_model("tfidf_lr")
        model.fit(train_texts, train_labels)

    val_preds = model.predict(val_texts)
    metrics = _compute_metrics(val_labels.tolist(), val_preds.tolist())

    print(
        f"  val_accuracy={metrics['accuracy']:.4f}  "
        f"macro_f1={metrics['macro_f1']:.4f}  "
        f"contra_f1={metrics['contradiction_f1']:.4f}"
    )

    # Save checkpoint
    pkl_path = _pkl_checkpoint_path()
    save_tfidf_checkpoint(model.pipeline, pkl_path)
    print(f"  Checkpoint → {pkl_path}")

    # Log a single-row CSV (consistent interface with transformer logs)
    log_path = os.path.join(log_dir, "train_log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_accuracy,val_macro_f1,val_contradiction_f1,learning_rate\n")
        f.write(f"1,—,—,{metrics['accuracy']:.6f},{metrics['macro_f1']:.6f},{metrics['contradiction_f1']:.6f},—\n")

    results = {
        "model_name": "tfidf_lr",
        **metrics,
        "training_time_minutes": round(t.elapsed / 60, 4),
    }
    with open(_results_path("tfidf_lr"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results  → {_results_path('tfidf_lr')}")

    return metrics


# ---------------------------------------------------------------------------
# 3. Training orchestrator
# ---------------------------------------------------------------------------

def _ensure_processed_splits() -> None:
    """Generate processed CSV splits from raw JSON if they don't exist yet."""
    train_path = os.path.join(CONFIG.data.processed_dir, CONFIG.data.train_file)
    if os.path.exists(train_path):
        return

    raw_path = os.path.join(CONFIG.data.data_dir, CONFIG.data.raw_data_file)
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Raw data not found at '{raw_path}'.\n"
            "Run Section 5 (Dataset Setup) in the Colab notebook first."
        )

    print("[train] Processed splits not found — generating from raw data...")
    from src.dataset import load_pubmedqa, reformulate_dataset, split_dataset
    samples = load_pubmedqa(raw_path)
    df = reformulate_dataset(samples)
    split_dataset(df)
    print("[train] Splits generated and saved.")


def train_model(model_name: str, **kwargs) -> Dict:
    """Dispatch to train_transformer or train_tfidf based on model_name."""
    _ensure_processed_splits()
    if model_name == "tfidf_lr":
        return train_tfidf()
    else:
        return train_transformer(model_name=model_name, **kwargs)


# ---------------------------------------------------------------------------
# CLI entry (sanity checks)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BioVerify training")
    parser.add_argument("--model", default="pubmedbert",
                        choices=["tfidf_lr", "distilbert", "biobert", "pubmedbert"])
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs (use 2 for sanity check)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Cap train batches per epoch (fast dev run)")
    parser.add_argument("--max_val_steps", type=int, default=None,
                        help="Cap val batches per epoch (fast dev run)")
    args = parser.parse_args()

    metrics = train_model(
        args.model,
        max_epochs=args.epochs,
        max_train_steps=args.max_steps,
        max_val_steps=args.max_val_steps,
    )
    print(f"\nFinal best metrics: {metrics}")
