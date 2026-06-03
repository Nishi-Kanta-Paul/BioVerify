"""
Rule-based patient-safety layer for BioVerify.

Components:
  detect_high_risk()       — keyword-based high-risk topic detection
  assign_safety_flag()     — maps (prediction, confidence, is_high_risk) → safety flag
  apply_safety_layer()     — batch application over a test set
  compute_safety_metrics() — unsafe catch rate, false safe rate, flag distribution
  plot_safety_distribution()
  generate_safety_table()
  run_safety_analysis()    — full pipeline
"""

import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import CONFIG, CLASS_NAMES, HIGH_RISK_KEYWORDS, LABEL2ID


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAFETY_FLAGS = ("safe", "unsafe", "expert_review")

_SUPPORTED_ID    = LABEL2ID["supported"]
_CONTRADICTED_ID = LABEL2ID["contradicted"]
_UNCERTAIN_ID    = LABEL2ID["uncertain"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _fig_path(filename: str) -> str:
    os.makedirs(CONFIG.experiment.figures_dir, exist_ok=True)
    return os.path.join(CONFIG.experiment.figures_dir, filename)


def _table_path(filename: str) -> str:
    os.makedirs(CONFIG.experiment.tables_dir, exist_ok=True)
    return os.path.join(CONFIG.experiment.tables_dir, filename)


# ---------------------------------------------------------------------------
# 1. High-risk topic detection
# ---------------------------------------------------------------------------

def detect_high_risk(
    question: str,
    evidence: str,
    keywords_dict: Optional[Dict[str, List[str]]] = None,
) -> Tuple[bool, List[str]]:
    """
    Scan question and evidence text for high-risk medical keywords.

    Args:
        question:      The biomedical question string.
        evidence:      The PubMed abstract evidence string.
        keywords_dict: Category → keyword list. Defaults to CONFIG safety keywords.

    Returns:
        (is_high_risk, matched_categories)
        is_high_risk       True if any keyword from any category is found.
        matched_categories List of category names where a match was found.
    """
    if keywords_dict is None:
        keywords_dict = HIGH_RISK_KEYWORDS

    # Combined text, lower-cased for case-insensitive matching
    combined = (str(question) + " " + str(evidence)).lower()

    matched_categories: List[str] = []
    for category, keywords in keywords_dict.items():
        for kw in keywords:
            # Use word-boundary matching for single-word keywords to avoid
            # spurious hits (e.g. "dose" inside "predispose")
            pattern = (
                r"\b" + re.escape(kw.lower()) + r"\b"
                if " " not in kw            # single token → word boundary
                else re.escape(kw.lower())  # multi-word phrase → substring
            )
            if re.search(pattern, combined):
                matched_categories.append(category)
                break   # one match per category is sufficient

    is_high_risk = len(matched_categories) > 0
    return is_high_risk, matched_categories


# ---------------------------------------------------------------------------
# 2. Safety flag assignment
# ---------------------------------------------------------------------------

def assign_safety_flag(
    prediction_label: int,
    confidence: float,
    threshold: float,
    is_high_risk: bool,
) -> str:
    """
    Map (prediction, confidence, is_high_risk) → safety flag.

    Implements the 7-row table from STABLE §7:

    | Condition                                    | Flag          |
    |----------------------------------------------|---------------|
    | supported AND conf ≥ τ AND NOT high_risk     | safe          |
    | supported AND conf ≥ τ AND high_risk         | safe          |
    | contradicted AND high_risk                   | unsafe        |
    | contradicted AND NOT high_risk               | expert_review |
    | uncertain (or conf < τ) AND high_risk        | expert_review |
    | uncertain (or conf < τ) AND NOT high_risk    | expert_review |
    | any other case                               | expert_review |
    """
    low_confidence = confidence < threshold
    effective_uncertain = (prediction_label == _UNCERTAIN_ID) or low_confidence

    if prediction_label == _SUPPORTED_ID and not low_confidence:
        # Supported + confident → safe regardless of high-risk status
        return "safe"

    if prediction_label == _CONTRADICTED_ID and not low_confidence:
        return "unsafe" if is_high_risk else "expert_review"

    if effective_uncertain:
        # Both high-risk and non-high-risk uncertain → expert_review
        return "expert_review"

    # Catch-all
    return "expert_review"


# ---------------------------------------------------------------------------
# 3. Batch safety assignment
# ---------------------------------------------------------------------------

def apply_safety_layer(
    predictions: np.ndarray,
    confidences: np.ndarray,
    questions: List[str],
    evidences: List[str],
    threshold: float,
    keywords_dict: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[str], List[bool], List[List[str]]]:
    """
    Apply detect_high_risk + assign_safety_flag to every sample.

    Returns:
        safety_flags:        list of "safe" / "unsafe" / "expert_review"
        high_risk_flags:     parallel bool list
        matched_categories:  parallel list of matched category lists
    """
    if keywords_dict is None:
        keywords_dict = HIGH_RISK_KEYWORDS

    safety_flags: List[str] = []
    high_risk_flags: List[bool] = []
    matched_cats: List[List[str]] = []

    for pred, conf, q, ev in zip(predictions, confidences, questions, evidences):
        is_hr, cats = detect_high_risk(q, ev, keywords_dict)
        flag = assign_safety_flag(int(pred), float(conf), threshold, is_hr)
        safety_flags.append(flag)
        high_risk_flags.append(is_hr)
        matched_cats.append(cats)

    return safety_flags, high_risk_flags, matched_cats


# ---------------------------------------------------------------------------
# 4. Safety evaluation metrics
# ---------------------------------------------------------------------------

def compute_safety_metrics(
    predictions: np.ndarray,
    confidences: np.ndarray,
    true_labels: np.ndarray,
    safety_flags: List[str],
    high_risk_flags: List[bool],
) -> Dict[str, Any]:
    """
    Compute safety layer evaluation metrics.

    Metrics:
      a. flag_distribution       — count of safe / unsafe / expert_review
      b. unsafe_catch_rate       — among high-risk contradicted/uncertain true samples,
                                   % flagged as "unsafe" or "expert_review"
      c. false_safe_rate         — among samples flagged "safe",
                                   % that are actually contradicted or uncertain (critical error)
      d. expert_review_rate      — % of all samples sent to expert review
      e. high_risk_rate          — % of samples detected as high-risk
    """
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    confidences = np.array(confidences)
    safety_flags = list(safety_flags)
    high_risk_flags = list(high_risk_flags)
    n = len(predictions)

    # ---- a. Flag distribution ----
    flag_dist = {
        "safe": safety_flags.count("safe"),
        "unsafe": safety_flags.count("unsafe"),
        "expert_review": safety_flags.count("expert_review"),
    }

    # ---- b. Unsafe catch rate ----
    # Numerator: high-risk samples with true label contradicted/uncertain
    #            that got "unsafe" or "expert_review"
    # Denominator: all high-risk samples with true label contradicted/uncertain
    truly_risky_mask = (
        np.array(high_risk_flags)
        & np.isin(true_labels, [_CONTRADICTED_ID, _UNCERTAIN_ID])
    )
    truly_risky_count = int(truly_risky_mask.sum())
    if truly_risky_count > 0:
        caught_mask = truly_risky_mask & np.isin(safety_flags, ["unsafe", "expert_review"])
        unsafe_catch_rate = float(caught_mask.sum() / truly_risky_count)
    else:
        unsafe_catch_rate = float("nan")

    # ---- c. False safe rate (critical error) ----
    # Among samples flagged "safe", % that are actually contradicted or uncertain
    safe_mask = np.array([f == "safe" for f in safety_flags])
    n_safe = int(safe_mask.sum())
    if n_safe > 0:
        actually_unsafe_mask = safe_mask & np.isin(true_labels, [_CONTRADICTED_ID, _UNCERTAIN_ID])
        false_safe_rate = float(actually_unsafe_mask.sum() / n_safe)
    else:
        false_safe_rate = float("nan")

    # ---- d. Expert review rate ----
    expert_review_rate = float(safety_flags.count("expert_review") / n)

    # ---- e. High-risk rate ----
    high_risk_rate = float(sum(high_risk_flags) / n)

    metrics = {
        "flag_distribution": flag_dist,
        "unsafe_catch_rate": (
            round(unsafe_catch_rate, 4) if not np.isnan(unsafe_catch_rate) else None
        ),
        "false_safe_rate": (
            round(false_safe_rate, 4) if not np.isnan(false_safe_rate) else None
        ),
        "expert_review_rate": round(expert_review_rate, 4),
        "high_risk_rate": round(high_risk_rate, 4),
        "n_safe": flag_dist["safe"],
        "n_unsafe": flag_dist["unsafe"],
        "n_expert_review": flag_dist["expert_review"],
        "n_truly_risky": truly_risky_count,
    }
    return metrics


# ---------------------------------------------------------------------------
# 5. Plot generation
# ---------------------------------------------------------------------------

def plot_safety_distribution(
    safety_flags: List[str],
    model_name: str,
) -> str:
    """Bar chart of safe / unsafe / expert_review counts."""
    counts = {f: safety_flags.count(f) for f in SAFETY_FLAGS}
    colors = {"safe": "steelblue", "unsafe": "tomato", "expert_review": "goldenrod"}
    total = len(safety_flags)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        list(counts.keys()),
        list(counts.values()),
        color=[colors[f] for f in counts],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, (flag, count) in zip(bars, counts.items()):
        pct = 100 * count / total if total > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values()) * 0.02,
            f"{count}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("Number of samples")
    ax.set_title(f"Safety Flag Distribution — {model_name}")
    ax.set_ylim(0, max(counts.values()) * 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = _fig_path(f"safety_flag_distribution_{model_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[safety] Figure → {path}")
    return path


# ---------------------------------------------------------------------------
# 6. Table generation
# ---------------------------------------------------------------------------

def generate_safety_table(
    model_name: str,
    metrics: Dict[str, Any],
    threshold: float,
) -> str:
    """
    Append one row to outputs/tables/safety_evaluation.csv.
    """
    path = _table_path("safety_evaluation.csv")
    row = {
        "model": model_name,
        "threshold": threshold,
        "n_safe": metrics.get("n_safe"),
        "n_unsafe": metrics.get("n_unsafe"),
        "n_expert_review": metrics.get("n_expert_review"),
        "high_risk_rate": metrics.get("high_risk_rate"),
        "unsafe_catch_rate": metrics.get("unsafe_catch_rate"),
        "false_safe_rate": metrics.get("false_safe_rate"),
        "expert_review_rate": metrics.get("expert_review_rate"),
    }

    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[df["model"] != model_name]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(path, index=False, float_format="%.4f")
    print(f"[safety] Table  → {path}")
    return path


# ---------------------------------------------------------------------------
# 7. Full safety pipeline
# ---------------------------------------------------------------------------

def run_safety_analysis(
    predictions: np.ndarray,
    confidences: np.ndarray,
    true_labels: np.ndarray,
    threshold: float,
    test_df: pd.DataFrame,
    model_name: str,
    keywords_dict: Optional[Dict[str, List[str]]] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    End-to-end safety pipeline.

    1. Extract questions and evidences from test_df.
    2. Run batch safety assignment.
    3. Compute safety metrics.
    4. Generate plot and table.

    Returns:
        (safety_flags, metrics_dict)
    """
    if keywords_dict is None:
        keywords_dict = HIGH_RISK_KEYWORDS

    questions = test_df["question"].fillna("").tolist()
    evidences = test_df["evidence"].fillna("").tolist()

    # ---- 1. Batch assignment ----
    safety_flags, high_risk_flags, matched_cats = apply_safety_layer(
        predictions, confidences, questions, evidences, threshold, keywords_dict
    )

    # ---- 2. Metrics ----
    metrics = compute_safety_metrics(
        predictions, confidences, true_labels, safety_flags, high_risk_flags
    )
    metrics["model_name"] = model_name
    metrics["threshold"] = threshold

    # ---- 3. Plot ----
    plot_safety_distribution(safety_flags, model_name)

    # ---- 4. Table ----
    generate_safety_table(model_name, metrics, threshold)

    # ---- Print summary ----
    _print_safety_summary(model_name, metrics)

    return safety_flags, metrics


def _print_safety_summary(model_name: str, metrics: Dict[str, Any]) -> None:
    dist = metrics["flag_distribution"]
    total = sum(dist.values())
    print(f"\n[safety] ── {model_name.upper()} ──")
    print(f"  Flag distribution (n={total}):")
    for flag in SAFETY_FLAGS:
        count = dist[flag]
        pct = 100 * count / total if total else 0
        print(f"    {flag:>14}: {count:4d}  ({pct:.1f}%)")
    print(f"  High-risk rate:      {metrics['high_risk_rate']:.4f}")
    print(f"  Unsafe catch rate:   {metrics['unsafe_catch_rate']}")
    print(f"  False safe rate:     {metrics['false_safe_rate']}")
    print(f"  Expert review rate:  {metrics['expert_review_rate']:.4f}")


# ---------------------------------------------------------------------------
# CLI / sanity checks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import pickle
    from src.config import CONFIG
    from src.dataset import prepare_tfidf_data

    parser = argparse.ArgumentParser(description="BioVerify safety layer")
    parser.add_argument(
        "--model", default="tfidf_lr",
        choices=["tfidf_lr", "distilbert", "biobert", "pubmedbert"],
    )
    args = parser.parse_args()

    # ---- (a) Keyword detection sanity checks ----
    print("\n" + "=" * 60)
    print("  Sanity check (a): keyword detection")
    print("=" * 60)
    test_cases = [
        (
            "What is the best dosage of metformin for diabetes?",
            "Metformin is a first-line drug for type 2 diabetes management.",
            ["drug_dosage", "severe_disease"],    # expected categories (subset)
        ),
        (
            "Does exercise improve sleep quality?",
            "A randomised trial found 30 minutes of moderate exercise improved sleep scores.",
            [],                                   # expected: no high-risk categories
        ),
        (
            "Is it safe to take ibuprofen during pregnancy?",
            "NSAIDs including ibuprofen may affect fetal development, especially in the third trimester.",
            ["pregnancy", "drug_dosage"],         # expected categories (subset)
        ),
    ]
    for q, ev, expected_cats in test_cases:
        is_hr, cats = detect_high_risk(q, ev)
        status = "HIGH RISK" if is_hr else "SAFE"
        print(f"\n  Q: {q[:70]}")
        print(f"  → {status}  |  matched: {cats}")
        # Check expected categories are a subset of detected
        missing = [c for c in expected_cats if c not in cats]
        if missing:
            print(f"  WARNING: expected categories not detected: {missing}")
        else:
            print(f"  ✓ All expected categories detected")

    # ---- (b) Flag assignment sanity checks ----
    print("\n" + "=" * 60)
    print("  Sanity check (b): flag assignment logic")
    print("=" * 60)
    flag_cases = [
        # (pred_label,       conf, tau, high_risk, expected_flag)
        (_SUPPORTED_ID,    0.90, 0.70, False, "safe"),
        (_SUPPORTED_ID,    0.90, 0.70, True,  "safe"),           # supported+confident = safe even if high-risk
        (_CONTRADICTED_ID, 0.85, 0.70, True,  "unsafe"),
        (_CONTRADICTED_ID, 0.85, 0.70, False, "expert_review"),
        (_UNCERTAIN_ID,    0.80, 0.70, True,  "expert_review"),
        (_UNCERTAIN_ID,    0.80, 0.70, False, "expert_review"),
        (_SUPPORTED_ID,    0.40, 0.70, False, "expert_review"),  # low-conf supported
        (_SUPPORTED_ID,    0.40, 0.70, True,  "expert_review"),  # low-conf high-risk
    ]
    all_pass = True
    label_names = {_SUPPORTED_ID: "supported", _CONTRADICTED_ID: "contradicted", _UNCERTAIN_ID: "uncertain"}
    for pred, conf, tau, hr, expected in flag_cases:
        flag = assign_safety_flag(pred, conf, tau, hr)
        ok = "✓" if flag == expected else "✗"
        if flag != expected:
            all_pass = False
        print(f"  {ok}  pred={label_names[pred]:>14}  conf={conf}  τ={tau}  "
              f"high_risk={str(hr):<5}  → {flag:<14}  (expected {expected})")
    print(f"\n  All flag-assignment checks: {'PASSED ✓' if all_pass else 'FAILED ✗'}")

    # ---- (c) Full safety analysis on test set ----
    print("\n" + "=" * 60)
    print(f"  Sanity check (c): full pipeline — {args.model}")
    print("=" * 60)

    processed_dir = CONFIG.data.processed_dir
    test_df = pd.read_csv(
        os.path.join(processed_dir, CONFIG.data.test_file)
    )
    from src.config import LABEL2ID as L2I
    true_labels = test_df["label"].map(L2I).values

    if args.model == "tfidf_lr":
        pkl_path = os.path.join(
            CONFIG.experiment.experiment_dir, "tfidf_lr", "checkpoints", "best_model.pkl"
        )
        pipeline = pickle.load(open(pkl_path, "rb"))
        texts, _ = prepare_tfidf_data(test_df)
        y_pred = pipeline.predict(texts)
        y_probs = pipeline.predict_proba(texts)
        confidences = y_probs.max(axis=1)
        threshold = 0.50   # best τ from Stage 7

    else:
        import torch
        from src.utils import get_device
        from src.model import build_model
        from src.dataset import get_dataloaders
        from src.uncertainty import extract_confidences

        device = get_device()
        ckpt_path = os.path.join(
            CONFIG.experiment.experiment_dir, args.model, "checkpoints", "best_model.pth"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train model first.")

        _, _, test_loader = get_dataloaders(
            model_name=args.model,
            batch_size=CONFIG.training.batch_size,
            max_length=CONFIG.model.max_length,
        )
        model = build_model(args.model)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        _, y_pred, _, confidences = extract_confidences(model, test_loader, device)
        threshold = CONFIG.uncertainty.default_threshold

    safety_flags, metrics = run_safety_analysis(
        predictions=y_pred,
        confidences=confidences,
        true_labels=true_labels,
        threshold=threshold,
        test_df=test_df,
        model_name=args.model,
    )
