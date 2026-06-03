"""
Baseline: DistilBERT (general-domain transformer)

Thin wrapper — configures model_name="distilbert" and calls shared pipeline functions.
Run: python baselines/train_distilbert.py [--epochs N] [--max_steps N]

On GPU (Colab): run without --max_steps for full training.
On CPU (local sanity check): use --max_steps 4 --epochs 2 to verify the pipeline only.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONFIG, LABEL2ID
from src.train import train_model
from src.evaluate import run_full_evaluation
from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import get_device
from src.uncertainty import run_uncertainty_analysis, extract_confidences
from src.safety_layer import run_safety_analysis

MODEL_NAME = "distilbert"


def main(max_epochs: int = CONFIG.training.max_epochs,
         max_steps: int | None = None,
         max_val_steps: int | None = None) -> None:

    print(f"\n{'='*60}")
    print(f"  BioVerify Baseline — DistilBERT")
    print(f"{'='*60}\n")

    device = get_device()

    # ── 1. Train ──────────────────────────────────────────────────
    print("Step 1: Training")
    train_model(
        MODEL_NAME,
        max_epochs=max_epochs,
        max_train_steps=max_steps,
        max_val_steps=max_val_steps,
    )

    # ── 2. Evaluate on test set ───────────────────────────────────
    print("\nStep 2: Evaluation")
    metrics = run_full_evaluation(MODEL_NAME)

    # ── 3. Uncertainty analysis ───────────────────────────────────
    print("\nStep 3: Uncertainty analysis")
    ckpt_path = os.path.join(
        CONFIG.experiment.experiment_dir, MODEL_NAME, "checkpoints", "best_model.pth"
    )
    _, val_loader, test_loader = get_dataloaders(
        model_name=MODEL_NAME,
        batch_size=CONFIG.training.batch_size,
        max_length=CONFIG.model.max_length,
    )
    model = build_model(MODEL_NAME)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    val_data  = extract_confidences(model, val_loader,  device)
    test_data = extract_confidences(model, test_loader, device)

    best_tau, unc_metrics = run_uncertainty_analysis(
        model_name=MODEL_NAME,
        val_data=val_data,
        test_data=test_data,
    )

    # ── 4. Safety analysis ────────────────────────────────────────
    print("\nStep 4: Safety analysis")
    processed_dir = CONFIG.data.processed_dir
    test_df = pd.read_csv(os.path.join(processed_dir, CONFIG.data.test_file))

    test_true, test_pred, _, test_confs = test_data

    run_safety_analysis(
        predictions=test_pred,
        confidences=test_confs,
        true_labels=test_true,
        threshold=best_tau,
        test_df=test_df,
        model_name=MODEL_NAME,
    )

    print(f"\n✓  {MODEL_NAME} pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DistilBERT baseline")
    parser.add_argument("--epochs",        type=int, default=CONFIG.training.max_epochs)
    parser.add_argument("--max_steps",     type=int, default=None,
                        help="Cap train batches per epoch (fast dev run)")
    parser.add_argument("--max_val_steps", type=int, default=None)
    args = parser.parse_args()
    main(max_epochs=args.epochs,
         max_steps=args.max_steps,
         max_val_steps=args.max_val_steps)
