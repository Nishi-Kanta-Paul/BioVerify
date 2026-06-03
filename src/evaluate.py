"""
Evaluation, metric computation, plot generation, and table generation
for BioVerify.

Components:
  compute_all_metrics()  — full metric dict from predictions + probabilities
  evaluate_model()       — run inference on test set, save results.json
  plot_confusion_matrix()
  plot_loss_curves()
  plot_f1_curves()
  plot_per_class_f1_comparison()
  plot_contradiction_metrics_comparison()
  plot_roc_curves()
  generate_main_results_table()
  generate_contradiction_focus_table()
"""

import json
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts and Colab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import CONFIG, CLASS_NAMES, ID2LABEL, LABEL2ID, NUM_CLASSES


# ---------------------------------------------------------------------------
# Paths helpers
# ---------------------------------------------------------------------------

def _fig_path(filename: str) -> str:
    os.makedirs(CONFIG.experiment.figures_dir, exist_ok=True)
    return os.path.join(CONFIG.experiment.figures_dir, filename)


def _table_path(filename: str) -> str:
    os.makedirs(CONFIG.experiment.tables_dir, exist_ok=True)
    return os.path.join(CONFIG.experiment.tables_dir, filename)


def _results_path(model_name: str) -> str:
    path = os.path.join(CONFIG.experiment.experiment_dir, model_name, "results.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. Metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_probs: Optional[np.ndarray] = None,
    class_names: List[str] = CLASS_NAMES,
) -> Dict[str, Any]:
    """
    Compute the full BioVerify metric suite.

    Args:
        y_true:      Ground-truth integer labels.
        y_pred:      Predicted integer labels.
        y_probs:     Softmax probabilities (N, num_classes); optional.
        class_names: Label names in id order.

    Returns:
        Structured dict matching STABLE §13 results.json schema.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    # ---- Overall ----
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # ---- Per-class ----
    per_class_precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    per_class_precision_d = {class_names[i]: float(per_class_precision[i]) for i in range(n_classes)}
    per_class_recall_d = {class_names[i]: float(per_class_recall[i]) for i in range(n_classes)}
    per_class_f1_d = {class_names[i]: float(per_class_f1[i]) for i in range(n_classes)}

    # ---- Contradiction class (safety-critical, emphasised) ----
    contra_idx = LABEL2ID["contradicted"]
    contradiction_precision = float(per_class_precision[contra_idx])
    contradiction_recall = float(per_class_recall[contra_idx])
    contradiction_f1 = float(per_class_f1[contra_idx])

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # ---- AUC-ROC (one-vs-rest, requires probabilities) ----
    per_class_auc: Dict[str, float] = {}
    macro_auc: Optional[float] = None
    if y_probs is not None:
        y_probs = np.array(y_probs)
        y_bin = label_binarize(y_true, classes=labels)
        for i, name in enumerate(class_names):
            try:
                per_class_auc[name] = float(roc_auc_score(y_bin[:, i], y_probs[:, i]))
            except ValueError:
                per_class_auc[name] = float("nan")
        try:
            macro_auc = float(roc_auc_score(y_bin, y_probs, average="macro", multi_class="ovr"))
        except ValueError:
            macro_auc = None

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "per_class_precision": per_class_precision_d,
        "per_class_recall": per_class_recall_d,
        "per_class_f1": per_class_f1_d,
        "contradiction_precision": contradiction_precision,
        "contradiction_recall": contradiction_recall,
        "contradiction_f1": contradiction_f1,
        "confusion_matrix": cm,
        "per_class_auc": per_class_auc if per_class_auc else None,
        "macro_auc": macro_auc,
    }
    return metrics


# ---------------------------------------------------------------------------
# 2. Model evaluation (inference → metrics → results.json)
# ---------------------------------------------------------------------------

def evaluate_transformer(
    model: "torch.nn.Module",
    test_loader: "torch.utils.data.DataLoader",
    model_name: str,
    device: "torch.device",
    extra_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run a transformer model over test_loader, compute metrics, save results.json.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_probs: List[List[float]] = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Evaluating {model_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"]

            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)

            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

    metrics = compute_all_metrics(all_labels, all_preds, np.array(all_probs))
    _save_results_json(model_name, metrics, extra_info)
    return metrics


def evaluate_tfidf(
    pipeline: Any,
    test_df: pd.DataFrame,
    model_name: str = "tfidf_lr",
    extra_info: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate a fitted sklearn TF-IDF pipeline on a test DataFrame.
    """
    from src.dataset import prepare_tfidf_data

    texts, y_true = prepare_tfidf_data(test_df)
    y_pred = pipeline.predict(texts)
    y_probs = pipeline.predict_proba(texts)

    metrics = compute_all_metrics(y_true.tolist(), y_pred.tolist(), y_probs)
    _save_results_json(model_name, metrics, extra_info)
    return metrics


def _save_results_json(
    model_name: str,
    metrics: Dict[str, Any],
    extra_info: Optional[Dict] = None,
) -> None:
    """
    Write results.json following the STABLE §13 schema.
    """
    record = {
        "model_name": model_name,
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "per_class_precision": metrics.get("per_class_precision"),
        "per_class_recall": metrics.get("per_class_recall"),
        "per_class_f1": metrics.get("per_class_f1"),
        "contradiction_precision": metrics.get("contradiction_precision"),
        "contradiction_recall": metrics.get("contradiction_recall"),
        "contradiction_f1": metrics.get("contradiction_f1"),
        "per_class_auc": metrics.get("per_class_auc"),
        "macro_auc": metrics.get("macro_auc"),
        "confusion_matrix": metrics.get("confusion_matrix"),
    }
    if extra_info:
        record.update(extra_info)

    path = _results_path(model_name)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"[evaluate] results.json → {path}")


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------

# --- 3a. Confusion matrix ---

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    model_name: str,
    class_names: List[str] = CLASS_NAMES,
) -> str:
    """Save annotated confusion matrix heatmap. Returns saved path."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {model_name}")

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    path = _fig_path(f"confusion_matrix_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Figure → {path}")
    return path


# --- 3b/3c. Training curves (loss + F1) ---

def plot_loss_curves(model_name: str, log_csv: Optional[str] = None) -> Optional[str]:
    """Plot train/val loss curves from train_log.csv. Returns path or None."""
    if log_csv is None:
        log_csv = os.path.join(
            CONFIG.experiment.experiment_dir, model_name, "logs", "train_log.csv"
        )
    if not os.path.exists(log_csv):
        print(f"[evaluate] No train_log.csv for {model_name} — skipping loss curves")
        return None

    df = pd.read_csv(log_csv)
    df = df[pd.to_numeric(df["train_loss"], errors="coerce").notna()]   # drop TF-IDF '—' rows
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train_loss"].astype(float), marker="o", label="Train loss")
    ax.plot(df["epoch"], df["val_loss"].astype(float), marker="s", label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"Training / Validation Loss — {model_name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    path = _fig_path(f"loss_curves_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Figure → {path}")
    return path


def plot_f1_curves(model_name: str, log_csv: Optional[str] = None) -> Optional[str]:
    """Plot val macro-F1 curve from train_log.csv. Returns path or None."""
    if log_csv is None:
        log_csv = os.path.join(
            CONFIG.experiment.experiment_dir, model_name, "logs", "train_log.csv"
        )
    if not os.path.exists(log_csv):
        return None

    df = pd.read_csv(log_csv)
    df = df[pd.to_numeric(df["val_macro_f1"], errors="coerce").notna()]
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["val_macro_f1"].astype(float), marker="o", label="Val macro-F1")
    ax.plot(df["epoch"], df["val_contradiction_f1"].astype(float), marker="s",
            label="Val contradiction-F1", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 score")
    ax.set_title(f"Validation F1 Curves — {model_name}")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    path = _fig_path(f"f1_curves_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Figure → {path}")
    return path


# --- 3d. Per-class F1 comparison (all models) ---

def plot_per_class_f1_comparison(results: Dict[str, Dict]) -> str:
    """
    Grouped bar chart of per-class F1 across all models.

    Args:
        results: {model_name: metrics_dict, ...}
    """
    model_names = list(results.keys())
    x = np.arange(len(CLASS_NAMES))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(model_names):
        f1_vals = [results[name]["per_class_f1"].get(c, 0.0) for c in CLASS_NAMES]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_vals, width * 0.9, label=name)
        for bar, val in zip(bars, f1_vals):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 — Model Comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = _fig_path("per_class_f1_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Figure → {path}")
    return path


# --- 3e. Contradiction metrics comparison ---

def plot_contradiction_metrics_comparison(results: Dict[str, Dict]) -> str:
    """
    Grouped bar chart: contradiction P / R / F1 per model.
    """
    model_names = list(results.keys())
    metric_keys = ["contradiction_precision", "contradiction_recall", "contradiction_f1"]
    metric_labels = ["Precision", "Recall", "F1"]
    x = np.arange(len(metric_labels))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(model_names):
        vals = [results[name].get(k, 0.0) for k in metric_keys]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=name)
        for bar, val in zip(bars, vals):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Contradiction-class Metrics — Model Comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = _fig_path("contradiction_metrics_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Figure → {path}")
    return path


# --- 3f. Per-class ROC curves ---

def plot_roc_curves(
    y_true: List[int],
    y_probs: np.ndarray,
    model_name: str,
    class_names: List[str] = CLASS_NAMES,
) -> Optional[str]:
    """Plot one-vs-rest ROC curves per class. Returns path or None."""
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    if y_bin.shape[1] != n_classes:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
        except ValueError:
            pass

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Per-class ROC Curves — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = _fig_path(f"roc_curves_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Figure → {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Tables
# ---------------------------------------------------------------------------

def generate_main_results_table(results: Dict[str, Dict]) -> str:
    """
    Write outputs/tables/main_results.csv — one row per model, all metrics.
    """
    rows = []
    for model_name, m in results.items():
        row = {
            "model": model_name,
            "accuracy": m.get("accuracy", ""),
            "macro_f1": m.get("macro_f1", ""),
            "macro_precision": m.get("macro_precision", ""),
            "macro_recall": m.get("macro_recall", ""),
        }
        for cls in CLASS_NAMES:
            row[f"precision_{cls}"] = (m.get("per_class_precision") or {}).get(cls, "")
            row[f"recall_{cls}"] = (m.get("per_class_recall") or {}).get(cls, "")
            row[f"f1_{cls}"] = (m.get("per_class_f1") or {}).get(cls, "")
        row["macro_auc"] = m.get("macro_auc", "")
        rows.append(row)

    df = pd.DataFrame(rows)
    path = _table_path("main_results.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"[evaluate] Table  → {path}")
    return path


def generate_contradiction_focus_table(results: Dict[str, Dict]) -> str:
    """
    Write outputs/tables/contradiction_focus.csv — contradiction P/R/F1 per model.
    """
    rows = []
    for model_name, m in results.items():
        rows.append({
            "model": model_name,
            "contradiction_precision": m.get("contradiction_precision", ""),
            "contradiction_recall": m.get("contradiction_recall", ""),
            "contradiction_f1": m.get("contradiction_f1", ""),
        })

    df = pd.DataFrame(rows)
    path = _table_path("contradiction_focus.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"[evaluate] Table  → {path}")
    return path


# ---------------------------------------------------------------------------
# 5. Full evaluation pipeline for a single model
# ---------------------------------------------------------------------------

def run_full_evaluation(model_name: str) -> Dict[str, Any]:
    """
    End-to-end evaluation for one model:
      1. Load checkpoint + test data
      2. Run inference → compute metrics
      3. Save results.json
      4. Generate all applicable plots
      5. Update main_results.csv and contradiction_focus.csv

    Returns the metrics dict.
    """
    import pickle
    from src.utils import get_device

    device = get_device()
    processed_dir = CONFIG.data.processed_dir
    test_df = pd.read_csv(os.path.join(processed_dir, CONFIG.data.test_file))

    # ---- Load model and run inference ----
    if model_name == "tfidf_lr":
        pkl_path = os.path.join(
            CONFIG.experiment.experiment_dir, "tfidf_lr", "checkpoints", "best_model.pkl"
        )
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"TF-IDF checkpoint not found: {pkl_path}")
        pipeline = pickle.load(open(pkl_path, "rb"))
        metrics = evaluate_tfidf(pipeline, test_df, model_name="tfidf_lr")
        y_true, y_pred, y_probs = _get_tfidf_predictions(pipeline, test_df)

    else:
        from src.model import TransformerClassifier, build_model
        from src.dataset import get_dataloaders

        ckpt_path = os.path.join(
            CONFIG.experiment.experiment_dir, model_name, "checkpoints", "best_model.pth"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        _, _, test_loader = get_dataloaders(
            model_name=model_name,
            batch_size=CONFIG.training.batch_size,
            max_length=CONFIG.model.max_length,
        )

        model = build_model(model_name)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        metrics = evaluate_transformer(model, test_loader, model_name, device)
        y_true, y_pred, y_probs = _get_transformer_predictions(model, test_loader, device)

    # ---- Plots ----
    plot_confusion_matrix(y_true, y_pred, model_name)
    if y_probs is not None:
        plot_roc_curves(y_true, np.array(y_probs), model_name)
    plot_loss_curves(model_name)
    plot_f1_curves(model_name)

    # ---- Comparison plots + tables (single-model version; updated by compare_results.py) ----
    results_dict = {model_name: metrics}
    plot_per_class_f1_comparison(results_dict)
    plot_contradiction_metrics_comparison(results_dict)
    generate_main_results_table(results_dict)
    generate_contradiction_focus_table(results_dict)

    _print_metrics_summary(model_name, metrics)
    return metrics


def _get_tfidf_predictions(pipeline, test_df):
    from src.dataset import prepare_tfidf_data
    texts, y_true = prepare_tfidf_data(test_df)
    y_pred = pipeline.predict(texts)
    y_probs = pipeline.predict_proba(texts)
    return y_true.tolist(), y_pred.tolist(), y_probs


def _get_transformer_predictions(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"]

            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
    return all_labels, all_preds, all_probs


def _print_metrics_summary(model_name: str, metrics: Dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Evaluation Results — {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Macro-F1:            {metrics['macro_f1']:.4f}")
    print(f"  Macro-Precision:     {metrics['macro_precision']:.4f}")
    print(f"  Macro-Recall:        {metrics['macro_recall']:.4f}")
    print(f"\n  Per-class F1:")
    for cls in CLASS_NAMES:
        f1 = (metrics.get("per_class_f1") or {}).get(cls, 0.0)
        prec = (metrics.get("per_class_precision") or {}).get(cls, 0.0)
        rec = (metrics.get("per_class_recall") or {}).get(cls, 0.0)
        print(f"    {cls:>14}: P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    print(f"\n  Contradiction-class (safety-critical):")
    print(f"    Precision: {metrics['contradiction_precision']:.4f}")
    print(f"    Recall:    {metrics['contradiction_recall']:.4f}")
    print(f"    F1:        {metrics['contradiction_f1']:.4f}")
    if metrics.get("macro_auc"):
        print(f"\n  Macro AUC-ROC: {metrics['macro_auc']:.4f}")
    print()


# ---------------------------------------------------------------------------
# CLI entry / sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BioVerify evaluation")
    parser.add_argument(
        "--model", default="tfidf_lr",
        choices=["tfidf_lr", "distilbert", "biobert", "pubmedbert"],
    )
    args = parser.parse_args()

    metrics = run_full_evaluation(args.model)
