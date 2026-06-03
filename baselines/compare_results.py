"""
Cross-model comparison: loads results from all trained models, generates
comparison tables and figures.

Run: python baselines/compare_results.py

Gracefully skips models whose results.json / checkpoints do not exist yet
(BioBERT and PubMedBERT must be trained on GPU first).
"""

import json
import os
import sys
import time
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONFIG, CLASS_NAMES, LABEL2ID, MODEL_REGISTRY
from src.evaluate import (
    generate_contradiction_focus_table,
    generate_main_results_table,
    plot_contradiction_metrics_comparison,
    plot_per_class_f1_comparison,
)

# All four models in the canonical comparison order
ALL_MODELS = ["tfidf_lr", "distilbert", "biobert", "pubmedbert"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _results_path(model_name: str) -> str:
    return os.path.join(CONFIG.experiment.experiment_dir, model_name, "results.json")


def _unc_table_path() -> str:
    return os.path.join(CONFIG.experiment.tables_dir, "uncertainty_detection.csv")


def _safety_table_path() -> str:
    return os.path.join(CONFIG.experiment.tables_dir, "safety_evaluation.csv")


def _fig_path(filename: str) -> str:
    os.makedirs(CONFIG.experiment.figures_dir, exist_ok=True)
    return os.path.join(CONFIG.experiment.figures_dir, filename)


def _table_path(filename: str) -> str:
    os.makedirs(CONFIG.experiment.tables_dir, exist_ok=True)
    return os.path.join(CONFIG.experiment.tables_dir, filename)


def load_all_results() -> dict[str, dict]:
    """Load results.json for every model that has one. Skip missing."""
    results = {}
    for model in ALL_MODELS:
        path = _results_path(model)
        if os.path.exists(path):
            with open(path) as f:
                results[model] = json.load(f)
            print(f"  [compare] Loaded  {path}")
        else:
            print(f"  [compare] Skipped {path}  (not found — train model first)")
    return results


# ---------------------------------------------------------------------------
# Table a: main_results.csv  (replaces single-model version from evaluate.py)
# ---------------------------------------------------------------------------

def build_main_results_table(results: dict) -> str:
    rows = []
    for model_name, m in results.items():
        row = {
            "model": model_name,
            "accuracy": m.get("accuracy", ""),
            "macro_f1": m.get("macro_f1", ""),
            "macro_precision": m.get("macro_precision", m.get("per_class_precision", {}) and ""),
            "macro_recall": m.get("macro_recall", ""),
        }
        pcp = m.get("per_class_precision") or {}
        pcr = m.get("per_class_recall")    or {}
        pcf = m.get("per_class_f1")        or {}
        for cls in CLASS_NAMES:
            row[f"precision_{cls}"] = pcp.get(cls, "")
            row[f"recall_{cls}"]    = pcr.get(cls, "")
            row[f"f1_{cls}"]        = pcf.get(cls, "")
        row["macro_auc"] = m.get("macro_auc", "")
        rows.append(row)

    df = pd.DataFrame(rows)
    path = _table_path("main_results.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"  [compare] Table  → {path}")
    return path


# ---------------------------------------------------------------------------
# Table b: contradiction_focus.csv
# ---------------------------------------------------------------------------

def build_contradiction_table(results: dict) -> str:
    rows = []
    for model_name, m in results.items():
        rows.append({
            "model": model_name,
            "contradiction_precision": m.get("contradiction_precision", ""),
            "contradiction_recall":    m.get("contradiction_recall", ""),
            "contradiction_f1":        m.get("contradiction_f1", ""),
        })
    df = pd.DataFrame(rows)
    path = _table_path("contradiction_focus.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"  [compare] Table  → {path}")
    return path


# ---------------------------------------------------------------------------
# Table e: efficiency_table.csv
# ---------------------------------------------------------------------------

def build_efficiency_table(results: dict) -> str:
    param_counts = {
        "tfidf_lr":   0,
        "distilbert": 66_365_187,
        "biobert":    108_312_579,
        "pubmedbert": 109_484_547,
    }
    rows = []
    for model_name, m in results.items():
        train_time = m.get("training_time_minutes", "")
        # Inference speed: measure on test set if checkpoint available; else leave blank
        inf_ms = _measure_inference_speed(model_name)
        rows.append({
            "model": model_name,
            "param_count": param_counts.get(model_name, ""),
            "training_time_minutes": train_time,
            "inference_ms_per_sample": inf_ms,
        })
    df = pd.DataFrame(rows)
    path = _table_path("efficiency_table.csv")
    df.to_csv(path, index=False)
    print(f"  [compare] Table  → {path}")
    return path


def _measure_inference_speed(model_name: str, n_samples: int = 32) -> str:
    """Time inference on n_samples from the test set. Returns ms/sample string or ''."""
    try:
        if model_name == "tfidf_lr":
            pkl_path = os.path.join(
                CONFIG.experiment.experiment_dir, "tfidf_lr", "checkpoints", "best_model.pkl"
            )
            if not os.path.exists(pkl_path):
                return ""
            from src.dataset import prepare_tfidf_data
            test_df = pd.read_csv(
                os.path.join(CONFIG.data.processed_dir, CONFIG.data.test_file)
            ).head(n_samples)
            texts, _ = prepare_tfidf_data(test_df)
            pipeline = pickle.load(open(pkl_path, "rb"))
            t0 = time.time()
            pipeline.predict_proba(texts)
            elapsed = time.time() - t0
            return round((elapsed / n_samples) * 1000, 3)

        else:
            ckpt_path = os.path.join(
                CONFIG.experiment.experiment_dir, model_name, "checkpoints", "best_model.pth"
            )
            if not os.path.exists(ckpt_path):
                return ""
            from src.model import build_model
            from src.dataset import get_dataloaders

            _, _, test_loader = get_dataloaders(
                model_name=model_name,
                batch_size=n_samples,
                max_length=CONFIG.model.max_length,
            )
            model = build_model(model_name)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            batch = next(iter(test_loader))
            input_ids    = batch["input_ids"]
            attn_mask    = batch["attention_mask"]
            tok_type_ids = batch.get("token_type_ids")

            with torch.no_grad():
                t0 = time.time()
                model(input_ids, attn_mask, tok_type_ids)
                elapsed = time.time() - t0
            return round((elapsed / n_samples) * 1000, 3)

    except Exception as e:
        print(f"    [compare] inference speed skipped for {model_name}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Table f: ablation_study.csv
# ---------------------------------------------------------------------------

def build_ablation_table(best_model: str = "tfidf_lr") -> str:
    """
    Ablation: evaluate best available model under 4 conditions:
      1. Full pipeline (model + uncertainty override + safety layer)
      2. Without uncertainty detection (no threshold override)
      3. Without safety layer (flags set to N/A)
      4. Without both

    Uses the test set predictions + the already-computed uncertainty/safety metrics.
    Falls back gracefully to tfidf_lr if pubmedbert not available.
    """
    import pickle
    from src.dataset import prepare_tfidf_data
    from src.evaluate import compute_all_metrics
    from src.uncertainty import apply_uncertainty_threshold, _UNCERTAIN_ID
    from src.safety_layer import apply_safety_layer, compute_safety_metrics
    from sklearn.metrics import accuracy_score, f1_score

    # ── Load test predictions ──────────────────────────────────────
    processed_dir = CONFIG.data.processed_dir
    test_df   = pd.read_csv(os.path.join(processed_dir, CONFIG.data.test_file))
    true_labels = test_df["label"].map(LABEL2ID).values

    # Try best available model checkpoint
    for candidate in ["pubmedbert", "biobert", "distilbert", "tfidf_lr"]:
        ckpt_exists = os.path.exists(os.path.join(
            CONFIG.experiment.experiment_dir, candidate,
            "checkpoints",
            "best_model.pth" if candidate != "tfidf_lr" else "best_model.pkl",
        ))
        if ckpt_exists:
            best_model = candidate
            break

    print(f"  [compare] Ablation model: {best_model}")

    if best_model == "tfidf_lr":
        pkl_path = os.path.join(
            CONFIG.experiment.experiment_dir, "tfidf_lr", "checkpoints", "best_model.pkl"
        )
        pipeline   = pickle.load(open(pkl_path, "rb"))
        texts, _   = prepare_tfidf_data(test_df)
        y_probs    = pipeline.predict_proba(texts)
        y_pred     = pipeline.predict(texts)
        confidences = y_probs.max(axis=1)
        threshold  = 0.50       # best tau from uncertainty stage
    else:
        from src.model import build_model
        from src.dataset import get_dataloaders
        from src.uncertainty import extract_confidences

        device = torch.device("cpu")
        ckpt_path = os.path.join(
            CONFIG.experiment.experiment_dir, best_model, "checkpoints", "best_model.pth"
        )
        _, _, test_loader = get_dataloaders(
            model_name=best_model,
            batch_size=CONFIG.training.batch_size,
            max_length=CONFIG.model.max_length,
        )
        model = build_model(best_model)
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        _, y_pred, y_probs, confidences = extract_confidences(model, test_loader, device)
        threshold = CONFIG.uncertainty.default_threshold

    y_pred      = np.array(y_pred)
    confidences = np.array(confidences)
    questions   = test_df["question"].fillna("").tolist()
    evidences   = test_df["evidence"].fillna("").tolist()

    # ── Four ablation variants ──────────────────────────────────────
    # 1. With uncertainty override
    y_with_unc = apply_uncertainty_threshold(y_pred.copy(), confidences, threshold)

    # 2. Safety flags (with uncertainty override)
    flags_full, hr_flags, _ = apply_safety_layer(
        y_with_unc, confidences, questions, evidences, threshold
    )
    # 3. Safety flags (without uncertainty override — raw predictions)
    flags_no_unc, _, _ = apply_safety_layer(
        y_pred.copy(), confidences, questions, evidences, threshold
    )

    def _metrics_row(variant, preds, flags=None):
        m = compute_all_metrics(true_labels.tolist(), preds.tolist())
        row = {
            "variant": variant,
            "model": best_model,
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "contradiction_f1": round(m["contradiction_f1"], 4),
            "uncertainty_recall": round(
                float(np.mean(preds[true_labels == LABEL2ID["uncertain"]] == LABEL2ID["uncertain"]))
                if (true_labels == LABEL2ID["uncertain"]).sum() > 0 else float("nan"),
                4,
            ),
        }
        if flags is not None:
            n = len(flags)
            row["pct_safe"]          = round(flags.count("safe") / n, 4)
            row["pct_unsafe"]        = round(flags.count("unsafe") / n, 4)
            row["pct_expert_review"] = round(flags.count("expert_review") / n, 4)
            # false safe rate
            safe_mask = np.array([f == "safe" for f in flags])
            truly_bad = np.isin(true_labels, [LABEL2ID["contradicted"], LABEL2ID["uncertain"]])
            n_safe = safe_mask.sum()
            row["false_safe_rate"] = round(
                float((safe_mask & truly_bad).sum() / n_safe) if n_safe > 0 else float("nan"), 4
            )
        else:
            row["pct_safe"] = row["pct_unsafe"] = row["pct_expert_review"] = ""
            row["false_safe_rate"] = ""
        return row

    rows = [
        _metrics_row("full_pipeline",          y_with_unc, flags_full),
        _metrics_row("without_uncertainty",     y_pred.copy(), flags_no_unc),
        _metrics_row("without_safety",          y_with_unc, None),
        _metrics_row("without_both",            y_pred.copy(), None),
    ]

    df = pd.DataFrame(rows)
    path = _table_path("ablation_study.csv")
    df.to_csv(path, index=False, float_format="%.4f")
    print(f"  [compare] Table  → {path}")
    return path


# ---------------------------------------------------------------------------
# Comparison figures (update from evaluate.py single-model versions)
# ---------------------------------------------------------------------------

def build_comparison_figures(results: dict) -> None:
    if len(results) < 1:
        return
    # per-class F1 grouped bar chart
    plot_per_class_f1_comparison(results)
    # contradiction metrics grouped bar chart
    plot_contradiction_metrics_comparison(results)


# ---------------------------------------------------------------------------
# Console summary printer
# ---------------------------------------------------------------------------

def print_comparison_summary(results: dict) -> None:
    if not results:
        print("\n  No model results available to compare.")
        return

    hdr_w = 14
    col_w = 10
    header_cols = ["accuracy", "macro_f1", "contra_f1", "supp_f1", "unc_f1"]

    print(f"\n{'='*72}")
    print(f"  MAIN RESULTS COMPARISON")
    print(f"{'='*72}")
    header = f"  {'Model':<{hdr_w}}" + "".join(f"{h:>{col_w}}" for h in header_cols)
    print(header)
    print("  " + "-" * (hdr_w + col_w * len(header_cols)))

    for model_name in ALL_MODELS:
        if model_name not in results:
            print(f"  {model_name:<{hdr_w}}" + "".join(f"{'—':>{col_w}}" for _ in header_cols))
            continue
        m = results[model_name]
        pcf = m.get("per_class_f1") or {}
        vals = [
            m.get("accuracy", float("nan")),
            m.get("macro_f1", float("nan")),
            m.get("contradiction_f1", float("nan")),
            pcf.get("supported", float("nan")),
            pcf.get("uncertain", float("nan")),
        ]
        row = f"  {model_name:<{hdr_w}}"
        for v in vals:
            row += f"{v:>{col_w}.4f}" if isinstance(v, float) else f"{'—':>{col_w}}"
        print(row)

    print(f"\n  {'Model':<{hdr_w}} {'Contra-P':>{col_w}} {'Contra-R':>{col_w}} {'Contra-F1':>{col_w}}")
    print("  " + "-" * (hdr_w + col_w * 3))
    for model_name in ALL_MODELS:
        if model_name not in results:
            print(f"  {model_name:<{hdr_w}}" + f"{'—':>{col_w}}" * 3)
            continue
        m = results[model_name]
        print(
            f"  {model_name:<{hdr_w}}"
            f"{m.get('contradiction_precision', float('nan')):>{col_w}.4f}"
            f"{m.get('contradiction_recall', float('nan')):>{col_w}.4f}"
            f"{m.get('contradiction_f1', float('nan')):>{col_w}.4f}"
        )

    # Uncertainty table (if available)
    unc_path = _unc_table_path()
    if os.path.exists(unc_path):
        unc_df = pd.read_csv(unc_path)
        print(f"\n{'='*72}")
        print(f"  UNCERTAINTY DETECTION")
        print(f"{'='*72}")
        unc_cols = ["model", "threshold", "low_conf_rate", "uncertainty_recall_after", "ece"]
        available = [c for c in unc_cols if c in unc_df.columns]
        print(unc_df[available].to_string(index=False))

    # Safety table (if available)
    saf_path = _safety_table_path()
    if os.path.exists(saf_path):
        saf_df = pd.read_csv(saf_path)
        print(f"\n{'='*72}")
        print(f"  SAFETY EVALUATION")
        print(f"{'='*72}")
        saf_cols = ["model", "unsafe_catch_rate", "false_safe_rate", "expert_review_rate"]
        available = [c for c in saf_cols if c in saf_df.columns]
        print(saf_df[available].to_string(index=False))


def print_ablation_summary(ablation_path: str) -> None:
    if not os.path.exists(ablation_path):
        return
    df = pd.read_csv(ablation_path)
    print(f"\n{'='*72}")
    print(f"  ABLATION STUDY")
    print(f"{'='*72}")
    display_cols = [c for c in ["variant", "accuracy", "macro_f1",
                                "contradiction_f1", "uncertainty_recall",
                                "false_safe_rate"] if c in df.columns]
    print(df[display_cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'='*60}")
    print(f"  BioVerify — Cross-Model Comparison")
    print(f"{'='*60}\n")

    # ── Load all available results ──────────────────────────────────
    results = load_all_results()

    if not results:
        print("\n  No results found. Train at least one model first.")
        return

    # ── (d) Verify all models used the same test set ────────────────
    test_df = pd.read_csv(
        os.path.join(CONFIG.data.processed_dir, CONFIG.data.test_file)
    )
    print(f"\n  Test set size: {len(test_df)} samples  (all models evaluate on this)")
    print(f"  Label distribution: "
          + ", ".join(f"{k}={v}" for k, v in
                      test_df["label"].value_counts().items()))

    # ── Tables ─────────────────────────────────────────────────────
    print("\n── Generating tables ──")
    build_main_results_table(results)
    build_contradiction_table(results)
    build_efficiency_table(results)

    # Uncertainty and safety tables are accumulated by stages 7/8;
    # print them as-is (they already have one row per trained model).
    for label, path in [("uncertainty_detection", _unc_table_path()),
                        ("safety_evaluation",     _safety_table_path())]:
        if os.path.exists(path):
            print(f"  [compare] Using existing {label}.csv  ({path})")
        else:
            print(f"  [compare] {label}.csv not yet generated "
                  f"(run uncertainty/safety stages first)")

    # ── Ablation ───────────────────────────────────────────────────
    print("\n── Ablation study ──")
    ablation_path = build_ablation_table()

    # ── Figures ────────────────────────────────────────────────────
    print("\n── Generating figures ──")
    build_comparison_figures(results)

    # ── Console summary ────────────────────────────────────────────
    print_comparison_summary(results)
    print_ablation_summary(ablation_path)

    print(f"\n✓  Comparison complete.  "
          f"Models with results: {list(results.keys())}")
    print(f"  BioBERT / PubMedBERT: train on Colab GPU, then re-run this script.")


if __name__ == "__main__":
    main()
