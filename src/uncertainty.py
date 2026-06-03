"""
Confidence-based uncertainty detection for BioVerify.

Components:
  extract_confidences()         — run inference, return labels + probs + confidences
  tune_threshold()              — grid-search τ on val set
  apply_uncertainty_threshold() — override low-confidence predictions to "uncertain"
  compute_uncertainty_metrics() — ECE, low-conf rate, before/after comparison
  plot_confidence_distribution()
  plot_threshold_tradeoff_curve()
  generate_uncertainty_table()
  run_uncertainty_analysis()    — full pipeline
"""

import os
import sys
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import CONFIG, CLASS_NAMES, ID2LABEL, LABEL2ID, NUM_CLASSES


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _fig_path(filename: str) -> str:
    os.makedirs(CONFIG.experiment.figures_dir, exist_ok=True)
    return os.path.join(CONFIG.experiment.figures_dir, filename)


def _table_path(filename: str) -> str:
    os.makedirs(CONFIG.experiment.tables_dir, exist_ok=True)
    return os.path.join(CONFIG.experiment.tables_dir, filename)


_UNCERTAIN_ID = LABEL2ID["uncertain"]   # label index 2


# ---------------------------------------------------------------------------
# 1. Confidence extraction
# ---------------------------------------------------------------------------

def extract_confidences(
    model: "torch.nn.Module",
    dataloader: "torch.utils.data.DataLoader",
    device: "torch.device",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference and return (y_true, y_pred, y_probs, confidences).

    confidences[i] = max(softmax(logits)[i])  — per-sample max probability.
    """
    model.eval()
    all_true, all_pred, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting confidences", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["label"]

            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)

            all_true.extend(labels.tolist())
            all_pred.extend(logits.argmax(dim=-1).cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_probs = np.array(all_probs)
    confidences = y_probs.max(axis=1)
    return y_true, y_pred, y_probs, confidences


def extract_confidences_tfidf(
    pipeline: Any,
    texts: List[str],
    y_true_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Equivalent of extract_confidences for sklearn pipeline."""
    y_probs = pipeline.predict_proba(texts)
    y_pred = pipeline.predict(texts)
    confidences = y_probs.max(axis=1)
    return y_true_arr, np.array(y_pred), np.array(y_probs), np.array(confidences)


# ---------------------------------------------------------------------------
# 2. Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    search_grid: Optional[List[float]] = None,
    accuracy_floor: float = CONFIG.uncertainty.accuracy_floor,
    class_names: List[str] = CLASS_NAMES,
) -> Tuple[float, pd.DataFrame]:
    """
    Grid-search over τ values on the validation set.

    Selection criterion (STABLE §7 / implementation_plan §4):
      Maximise uncertainty-class recall, subject to overall accuracy ≥ accuracy_floor.
      If no τ meets the floor, fall back to CONFIG.uncertainty.default_threshold.

    Returns:
        best_threshold: chosen τ
        results_df:     DataFrame with one row per τ
    """
    if search_grid is None:
        search_grid = CONFIG.uncertainty.threshold_search_grid

    baseline_accuracy = float(accuracy_score(y_true, y_pred))
    baseline_macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    baseline_unc_recall = _uncertain_recall(y_true, y_pred)

    rows = []
    for tau in search_grid:
        y_overridden = apply_uncertainty_threshold(y_pred.copy(), confidences, tau)
        low_conf_rate = float((confidences < tau).mean())
        acc = float(accuracy_score(y_true, y_overridden))
        macro_f1 = float(f1_score(y_true, y_overridden, average="macro", zero_division=0))
        unc_recall = _uncertain_recall(y_true, y_overridden)

        rows.append({
            "threshold": tau,
            "low_confidence_rate": round(low_conf_rate, 4),
            "accuracy_after_override": round(acc, 4),
            "macro_f1_after_override": round(macro_f1, 4),
            "uncertainty_recall_after_override": round(unc_recall, 4),
            "meets_accuracy_floor": acc >= accuracy_floor,
        })

    results_df = pd.DataFrame(rows)

    # Selection: maximise uncertainty recall among τ that meet accuracy floor
    candidates = results_df[results_df["meets_accuracy_floor"]]
    if not candidates.empty:
        best_row = candidates.loc[candidates["uncertainty_recall_after_override"].idxmax()]
        best_threshold = float(best_row["threshold"])
        selection_reason = (
            f"highest uncertainty recall ({best_row['uncertainty_recall_after_override']:.3f}) "
            f"with accuracy ≥ {accuracy_floor}"
        )
    else:
        # Fallback: use default threshold
        best_threshold = CONFIG.uncertainty.default_threshold
        selection_reason = (
            f"fallback (no τ met accuracy floor {accuracy_floor}); "
            f"using default τ={best_threshold}"
        )

    print(f"\n[uncertainty] Threshold tuning results:")
    print(f"  Baseline — accuracy={baseline_accuracy:.4f}  macro_f1={baseline_macro_f1:.4f}  "
          f"uncertain_recall={baseline_unc_recall:.4f}")
    print(results_df.to_string(index=False))
    print(f"\n  → Best τ = {best_threshold}  ({selection_reason})")

    return best_threshold, results_df


def _uncertain_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall on the 'uncertain' class (safe to call even if class absent in predictions)."""
    if _UNCERTAIN_ID not in y_true:
        return float("nan")
    return float(recall_score(
        y_true, y_pred,
        labels=[_UNCERTAIN_ID],
        average="micro",
        zero_division=0,
    ))


# ---------------------------------------------------------------------------
# 3. Prediction override
# ---------------------------------------------------------------------------

def apply_uncertainty_threshold(
    y_pred: np.ndarray,
    confidences: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Override predictions to 'uncertain' (label 2) where confidence < threshold.

    Args:
        y_pred:      Original predicted label indices (modified in-place copy).
        confidences: Per-sample max softmax probability.
        threshold:   τ cutoff.

    Returns:
        y_pred_overridden: new array with low-confidence predictions set to _UNCERTAIN_ID.
    """
    y_overridden = y_pred.copy()
    low_conf_mask = confidences < threshold
    y_overridden[low_conf_mask] = _UNCERTAIN_ID
    return y_overridden


# ---------------------------------------------------------------------------
# 4. Uncertainty metrics (including ECE)
# ---------------------------------------------------------------------------

def compute_uncertainty_metrics(
    y_true: np.ndarray,
    y_pred_original: np.ndarray,
    y_pred_overridden: np.ndarray,
    confidences: np.ndarray,
    threshold: float,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute the full uncertainty metric suite.

    Returns dict with:
      low_confidence_rate, ECE,
      before/after accuracy, macro_f1, uncertainty_recall
    """
    low_conf_rate = float((confidences < threshold).mean())
    n_overridden = int((confidences < threshold).sum())

    # Before override
    acc_before = float(accuracy_score(y_true, y_pred_original))
    mf1_before = float(f1_score(y_true, y_pred_original, average="macro", zero_division=0))
    unc_recall_before = _uncertain_recall(y_true, y_pred_original)

    # After override
    acc_after = float(accuracy_score(y_true, y_pred_overridden))
    mf1_after = float(f1_score(y_true, y_pred_overridden, average="macro", zero_division=0))
    unc_recall_after = _uncertain_recall(y_true, y_pred_overridden)

    # ECE — Expected Calibration Error
    ece = _compute_ece(y_true, y_pred_original, confidences, n_bins=n_bins)

    metrics = {
        "threshold": threshold,
        "low_confidence_rate": round(low_conf_rate, 4),
        "n_overridden": n_overridden,
        "ece": round(ece, 4),
        "accuracy_before": round(acc_before, 4),
        "accuracy_after": round(acc_after, 4),
        "macro_f1_before": round(mf1_before, 4),
        "macro_f1_after": round(mf1_after, 4),
        "uncertainty_recall_before": (
            round(unc_recall_before, 4) if not np.isnan(unc_recall_before) else None
        ),
        "uncertainty_recall_after": (
            round(unc_recall_after, 4) if not np.isnan(unc_recall_after) else None
        ),
    }
    return metrics


def _compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error.

    ECE = Σ_b (|B_b| / N) × |acc(B_b) − conf(B_b)|

    where B_b = {samples whose confidence falls in bin b}.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Include upper boundary in last bin
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        if mask.sum() == 0:
            continue

        bin_acc = float((y_pred[mask] == y_true[mask]).mean())
        bin_conf = float(confidences[mask].mean())
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def plot_confidence_distribution(
    y_true: np.ndarray,
    confidences: np.ndarray,
    model_name: str,
    threshold: Optional[float] = None,
    class_names: List[str] = CLASS_NAMES,
) -> str:
    """
    Histogram of confidence scores, one overlay per true class.
    Optionally draws a vertical line at the chosen threshold τ.
    """
    colors = ["steelblue", "tomato", "forestgreen"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for cls_id, (cls_name, color) in enumerate(zip(class_names, colors)):
        mask = y_true == cls_id
        if mask.sum() == 0:
            continue
        ax.hist(
            confidences[mask],
            bins=20,
            range=(0, 1),
            alpha=0.55,
            color=color,
            label=f"{cls_name} (n={mask.sum()})",
            density=True,
        )

    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                   label=f"τ = {threshold}")

    ax.set_xlabel("Confidence (max softmax probability)")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence Distribution by True Class — {model_name}")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = _fig_path(f"confidence_distribution_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[uncertainty] Figure → {path}")
    return path


def plot_threshold_tradeoff_curve(
    results_df: pd.DataFrame,
    model_name: str,
    best_threshold: Optional[float] = None,
) -> str:
    """
    Dual-metric curve: x=threshold, left-y=uncertainty_recall, right-y=macro_f1 & accuracy.
    """
    tau = results_df["threshold"].values
    unc_recall = results_df["uncertainty_recall_after_override"].values
    macro_f1 = results_df["macro_f1_after_override"].values
    accuracy = results_df["accuracy_after_override"].values

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(tau, unc_recall, "o-", color="forestgreen", label="Uncertain recall")
    l2, = ax2.plot(tau, macro_f1, "s--", color="steelblue", label="Macro-F1")
    l3, = ax2.plot(tau, accuracy, "^:", color="tomato", label="Accuracy")

    if best_threshold is not None:
        ax1.axvline(best_threshold, color="black", linestyle="--", linewidth=1.2,
                    label=f"Best τ={best_threshold}")

    ax1.set_xlabel("Confidence threshold τ")
    ax1.set_ylabel("Uncertain-class recall", color="forestgreen")
    ax2.set_ylabel("Macro-F1 / Accuracy", color="steelblue")
    ax1.set_title(f"Threshold Trade-off — {model_name}")
    ax1.set_ylim(-0.05, 1.1)
    ax2.set_ylim(-0.05, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.4)

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    if best_threshold is not None:
        from matplotlib.lines import Line2D
        lines.append(Line2D([0], [0], color="black", linestyle="--"))
        labels.append(f"Best τ={best_threshold}")
    ax1.legend(lines, labels, loc="center left", fontsize=8)

    fig.tight_layout()
    path = _fig_path(f"threshold_tradeoff_curve_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[uncertainty] Figure → {path}")
    return path


# ---------------------------------------------------------------------------
# 6. Table generation
# ---------------------------------------------------------------------------

def generate_uncertainty_table(
    model_name: str,
    metrics: Dict[str, Any],
    append: bool = True,
) -> str:
    """
    Append (or write) one row to outputs/tables/uncertainty_detection.csv.

    Columns: model, threshold, low_conf_rate, uncertainty_recall_after,
             macro_f1_after, ece
    """
    path = _table_path("uncertainty_detection.csv")
    row = {
        "model": model_name,
        "threshold": metrics.get("threshold"),
        "low_conf_rate": metrics.get("low_confidence_rate"),
        "uncertainty_recall_before": metrics.get("uncertainty_recall_before"),
        "uncertainty_recall_after": metrics.get("uncertainty_recall_after"),
        "macro_f1_before": metrics.get("macro_f1_before"),
        "macro_f1_after": metrics.get("macro_f1_after"),
        "accuracy_before": metrics.get("accuracy_before"),
        "accuracy_after": metrics.get("accuracy_after"),
        "ece": metrics.get("ece"),
        "n_overridden": metrics.get("n_overridden"),
    }

    if append and os.path.exists(path):
        df = pd.read_csv(path)
        # Remove any prior row for this model, then append fresh
        df = df[df["model"] != model_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False, float_format="%.4f")
    print(f"[uncertainty] Table  → {path}")
    return path


# ---------------------------------------------------------------------------
# 7. Full uncertainty pipeline
# ---------------------------------------------------------------------------

def run_uncertainty_analysis(
    model_name: str,
    val_data: Any,          # (y_true, y_pred, y_probs, confidences) or (pipeline, val_texts, val_labels)
    test_data: Any,         # same structure as val_data but for test set
    is_tfidf: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """
    End-to-end uncertainty detection pipeline.

    1. Tune threshold on val set.
    2. Apply best threshold to test set.
    3. Compute all uncertainty metrics on test set.
    4. Generate plots and table.

    Args:
        model_name: short name string for file naming.
        val_data:   tuple of (y_true, y_pred, y_probs, confidences) for val set.
        test_data:  same tuple for test set.
        is_tfidf:   flag (not used in logic, kept for clarity).

    Returns:
        (best_threshold, test_metrics_dict)
    """
    val_true, val_pred, val_probs, val_confs = val_data
    test_true, test_pred, test_probs, test_confs = test_data

    # ---- 1. Tune on val ----
    best_tau, tuning_df = tune_threshold(
        val_true, val_pred, val_confs,
        search_grid=CONFIG.uncertainty.threshold_search_grid,
        accuracy_floor=CONFIG.uncertainty.accuracy_floor,
    )

    # ---- 2. Apply to test ----
    test_pred_overridden = apply_uncertainty_threshold(test_pred.copy(), test_confs, best_tau)

    # ---- 3. Metrics on test ----
    test_metrics = compute_uncertainty_metrics(
        test_true, test_pred, test_pred_overridden, test_confs, best_tau
    )
    test_metrics["model_name"] = model_name

    # ---- 4. Plots ----
    plot_confidence_distribution(test_true, test_confs, model_name, threshold=best_tau)
    plot_threshold_tradeoff_curve(tuning_df, model_name, best_threshold=best_tau)

    # ---- 5. Table ----
    generate_uncertainty_table(model_name, test_metrics)

    # ---- Print summary ----
    _print_uncertainty_summary(model_name, best_tau, test_metrics)

    return best_tau, test_metrics


def _print_uncertainty_summary(model_name: str, threshold: float, metrics: Dict) -> None:
    print(f"\n[uncertainty] ── {model_name.upper()} ──")
    print(f"  Best τ:                   {threshold}")
    print(f"  Low-confidence rate:      {metrics['low_confidence_rate']:.4f}  "
          f"({metrics['n_overridden']} samples overridden)")
    print(f"  ECE:                      {metrics['ece']:.4f}")
    print(f"  Accuracy:     before={metrics['accuracy_before']:.4f}  "
          f"after={metrics['accuracy_after']:.4f}")
    print(f"  Macro-F1:     before={metrics['macro_f1_before']:.4f}  "
          f"after={metrics['macro_f1_after']:.4f}")
    print(f"  Uncert recall before={metrics['uncertainty_recall_before']}  "
          f"after={metrics['uncertainty_recall_after']}")


# ---------------------------------------------------------------------------
# CLI / sanity checks
# ---------------------------------------------------------------------------

def _load_tfidf_pipeline_data(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper: load TF-IDF pipeline + run inference on a split CSV."""
    from src.dataset import prepare_tfidf_data

    processed_dir = CONFIG.data.processed_dir
    file_map = {
        "val": CONFIG.data.val_file,
        "test": CONFIG.data.test_file,
        "train": CONFIG.data.train_file,
    }
    df = pd.read_csv(os.path.join(processed_dir, file_map[split]))
    texts, y_true = prepare_tfidf_data(df)

    pkl_path = os.path.join(
        CONFIG.experiment.experiment_dir, "tfidf_lr", "checkpoints", "best_model.pkl"
    )
    pipeline = pickle.load(open(pkl_path, "rb"))
    return extract_confidences_tfidf(pipeline, texts, y_true)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BioVerify uncertainty analysis")
    parser.add_argument(
        "--model", default="tfidf_lr",
        choices=["tfidf_lr", "distilbert", "biobert", "pubmedbert"],
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Uncertainty Analysis — {args.model.upper()}")
    print(f"{'='*60}")

    if args.model == "tfidf_lr":
        val_data = _load_tfidf_pipeline_data("val")
        test_data = _load_tfidf_pipeline_data("test")
    else:
        from src.utils import get_device
        from src.model import build_model
        from src.dataset import get_dataloaders

        device = get_device()
        ckpt_path = os.path.join(
            CONFIG.experiment.experiment_dir, args.model, "checkpoints", "best_model.pth"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train model first.")

        _, val_loader, test_loader = get_dataloaders(
            model_name=args.model,
            batch_size=CONFIG.training.batch_size,
            max_length=CONFIG.model.max_length,
        )
        model = build_model(args.model)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        val_data = extract_confidences(model, val_loader, device)
        test_data = extract_confidences(model, test_loader, device)

    best_tau, metrics = run_uncertainty_analysis(
        model_name=args.model,
        val_data=val_data,
        test_data=test_data,
        is_tfidf=(args.model == "tfidf_lr"),
    )

    # ---- Final sanity assertions ----
    print("\n── Sanity checks ──")
    assert 0.0 <= metrics["ece"] <= 1.0, f"ECE out of range: {metrics['ece']}"
    print(f"  ECE ∈ [0, 1]: {metrics['ece']} ✓")

    y_true, y_pred, y_probs, confs = (
        _load_tfidf_pipeline_data("test") if args.model == "tfidf_lr"
        else (None, None, None, None)
    )
    if y_true is not None:
        y_overridden = apply_uncertainty_threshold(y_pred.copy(), confs, best_tau)
        # Samples that are overridden AND weren't already predicted uncertain
        n_changed = int((y_overridden != y_pred).sum())
        n_below_not_already_uncertain = int(
            ((confs < best_tau) & (y_pred != _UNCERTAIN_ID)).sum()
        )
        assert n_changed == n_below_not_already_uncertain, (
            f"Override count mismatch: n_changed={n_changed} vs "
            f"n_below_not_already_uncertain={n_below_not_already_uncertain}"
        )
        print(f"  Override correctness: {n_changed} predictions changed to 'uncertain' "
              f"(of {int((confs < best_tau).sum())} below τ; "
              f"{int((confs < best_tau).sum()) - n_changed} already were uncertain) ✓")
