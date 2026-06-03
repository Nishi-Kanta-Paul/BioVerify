"""
Baseline: TF-IDF + Logistic Regression

Thin wrapper — configures model_name="tfidf_lr" and calls shared pipeline functions.
Run: python baselines/train_tfidf_lr.py
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONFIG, LABEL2ID
from src.train import train_model
from src.evaluate import run_full_evaluation
from src.dataset import prepare_tfidf_data
from src.uncertainty import run_uncertainty_analysis, extract_confidences_tfidf
from src.safety_layer import run_safety_analysis

MODEL_NAME = "tfidf_lr"


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  BioVerify Baseline — TF-IDF + Logistic Regression")
    print(f"{'='*60}\n")

    # ── 1. Train ──────────────────────────────────────────────────
    print("Step 1: Training")
    train_model(MODEL_NAME)

    # ── 2. Evaluate on test set ───────────────────────────────────
    print("\nStep 2: Evaluation")
    metrics = run_full_evaluation(MODEL_NAME)

    # ── 3. Uncertainty analysis ───────────────────────────────────
    print("\nStep 3: Uncertainty analysis")
    processed_dir = CONFIG.data.processed_dir
    pkl_path = os.path.join(
        CONFIG.experiment.experiment_dir, MODEL_NAME, "checkpoints", "best_model.pkl"
    )
    pipeline = pickle.load(open(pkl_path, "rb"))

    val_df  = pd.read_csv(os.path.join(processed_dir, CONFIG.data.val_file))
    test_df = pd.read_csv(os.path.join(processed_dir, CONFIG.data.test_file))
    val_texts,  val_labels  = prepare_tfidf_data(val_df)
    test_texts, test_labels = prepare_tfidf_data(test_df)

    val_data  = extract_confidences_tfidf(pipeline, val_texts,  val_labels)
    test_data = extract_confidences_tfidf(pipeline, test_texts, test_labels)

    best_tau, unc_metrics = run_uncertainty_analysis(
        model_name=MODEL_NAME,
        val_data=val_data,
        test_data=test_data,
    )

    # ── 4. Safety analysis ────────────────────────────────────────
    print("\nStep 4: Safety analysis")
    _, y_pred, y_probs, confidences = test_data
    true_labels = np.array(test_labels)

    run_safety_analysis(
        predictions=y_pred,
        confidences=confidences,
        true_labels=true_labels,
        threshold=best_tau,
        test_df=test_df,
        model_name=MODEL_NAME,
    )

    print(f"\n✓  {MODEL_NAME} pipeline complete.")


if __name__ == "__main__":
    main()
