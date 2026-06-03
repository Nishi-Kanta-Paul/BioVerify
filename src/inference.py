"""
Inference pipeline for BioVerify.

Runs the full three-stage pipeline for any input:
  1. Transformer/TF-IDF prediction → verification label + confidence
  2. Confidence-based uncertainty override
  3. Rule-based patient-safety flagging

Usage (CLI):
  python src/inference.py --model tfidf_lr --input data/sample_input.json
  python src/inference.py --model distilbert --checkpoint experiments/distilbert/checkpoints/best_model.pth --input data/sample_input.json
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONFIG, CLASS_NAMES, ID2LABEL, LABEL2ID, MODEL_REGISTRY
from src.safety_layer import assign_safety_flag, detect_high_risk
from src.utils import get_device


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline(
    model_name: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load the trained model (transformer or TF-IDF) and its tokeniser.

    Returns a dict:
      {
        "model":      fitted model object,
        "tokenizer":  AutoTokenizer or None (for TF-IDF),
        "model_name": model_name,
        "device":     device,
        "is_tfidf":   bool,
        "threshold":  float,   # best tau from uncertainty stage if available
      }
    """
    import pickle

    if device is None:
        device = get_device()

    # Resolve checkpoint path
    if checkpoint_path is None:
        suffix = "best_model.pkl" if model_name == "tfidf_lr" else "best_model.pth"
        checkpoint_path = os.path.join(
            CONFIG.experiment.experiment_dir, model_name, "checkpoints", suffix
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train the model first: python src/main.py --mode train --model {model_name}"
        )

    # Load threshold from uncertainty table if available
    threshold = CONFIG.uncertainty.default_threshold
    unc_table = os.path.join(CONFIG.experiment.tables_dir, "uncertainty_detection.csv")
    if os.path.exists(unc_table):
        df = pd.read_csv(unc_table)
        row = df[df["model"] == model_name]
        if not row.empty and "threshold" in row.columns:
            threshold = float(row.iloc[0]["threshold"])

    if model_name == "tfidf_lr":
        with open(checkpoint_path, "rb") as f:
            pipeline = pickle.load(f)
        return {
            "model": pipeline,
            "tokenizer": None,
            "model_name": model_name,
            "device": device,
            "is_tfidf": True,
            "threshold": threshold,
        }
    else:
        from src.model import build_model
        from transformers import AutoTokenizer

        model = build_model(model_name)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        hf_id = MODEL_REGISTRY[model_name]
        tokenizer = AutoTokenizer.from_pretrained(hf_id)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_name": model_name,
            "device": device,
            "is_tfidf": False,
            "threshold": threshold,
        }


# ---------------------------------------------------------------------------
# Single-sample prediction
# ---------------------------------------------------------------------------

def predict_single(
    pipeline_state: Dict[str, Any],
    question: str,
    evidence: str,
    candidate_answer: str,
) -> Dict[str, Any]:
    """
    Run the full BioVerify pipeline on one (question, evidence, candidate_answer) triple.

    Returns a dict matching STABLE §12 output schema:
      {
        question, candidate_answer,
        verification_label, confidence,
        is_high_risk, high_risk_categories,
        safety_flag, reasoning
      }
    """
    model       = pipeline_state["model"]
    tokenizer   = pipeline_state["tokenizer"]
    device      = pipeline_state["device"]
    is_tfidf    = pipeline_state["is_tfidf"]
    threshold   = pipeline_state["threshold"]
    model_name  = pipeline_state["model_name"]

    # ── Step 1: Get logits / probabilities ──────────────────────────────
    t0 = time.time()

    if is_tfidf:
        text = f"{question} {evidence} {candidate_answer}"
        probs = model.predict_proba([text])[0]            # shape (3,)
        pred_id = int(np.argmax(probs))
        confidence = float(probs.max())
    else:
        sep = tokenizer.sep_token or "[SEP]"
        text_a = question
        text_b = f"{evidence} {sep} {candidate_answer}"
        encoding = tokenizer(
            text_a, text_b,
            max_length=CONFIG.model.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        input_ids    = encoding["input_ids"].to(device)
        attn_mask    = encoding["attention_mask"].to(device)
        tok_type_ids = encoding.get("token_type_ids")
        if tok_type_ids is not None:
            tok_type_ids = tok_type_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, attn_mask, tok_type_ids)
        probs      = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_id    = int(np.argmax(probs))
        confidence = float(probs.max())

    inference_ms = (time.time() - t0) * 1000

    # ── Step 2: Uncertainty override ──────────────────────────────────
    if confidence < threshold:
        final_pred_id = LABEL2ID["uncertain"]
        overridden    = True
    else:
        final_pred_id = pred_id
        overridden    = False

    verification_label = ID2LABEL[final_pred_id]

    # ── Step 3: Safety flagging ────────────────────────────────────────
    is_high_risk, matched_cats = detect_high_risk(question, evidence)
    safety_flag = assign_safety_flag(final_pred_id, confidence, threshold, is_high_risk)

    # ── Reasoning string ──────────────────────────────────────────────
    reasoning_parts = [
        f"Model: {model_name}.",
        f"Predicted '{ID2LABEL[pred_id]}' with confidence {confidence:.3f} (τ={threshold}).",
    ]
    if overridden:
        reasoning_parts.append("Overridden to 'uncertain' due to low confidence.")
    if is_high_risk:
        reasoning_parts.append(f"High-risk categories: {matched_cats}.")
    reasoning_parts.append(f"Safety flag assigned: {safety_flag}.")

    return {
        "question":            question,
        "evidence":            evidence[:200] + "..." if len(evidence) > 200 else evidence,
        "candidate_answer":    candidate_answer,
        "verification_label":  verification_label,
        "confidence":          round(confidence, 4),
        "threshold_used":      threshold,
        "overridden_to_uncertain": overridden,
        "is_high_risk":        is_high_risk,
        "high_risk_categories": matched_cats,
        "safety_flag":         safety_flag,
        "reasoning":           " ".join(reasoning_parts),
        "inference_ms":        round(inference_ms, 2),
        "model_name":          model_name,
    }


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(
    pipeline_state: Dict[str, Any],
    inputs: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """
    Run the full pipeline on a list of input dicts.
    Each input must have keys: question, evidence, candidate_answer.
    """
    results = []
    for i, inp in enumerate(inputs):
        result = predict_single(
            pipeline_state,
            question         = inp.get("question", ""),
            evidence         = inp.get("evidence", ""),
            candidate_answer = inp.get("candidate_answer", ""),
        )
        result["sample_id"] = i
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# File-based inference runner
# ---------------------------------------------------------------------------

def run_inference(
    model_name: str,
    input_path: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """
    Load JSON input (single dict or list of dicts), run full pipeline,
    save to outputs/predictions/.

    Returns list of result dicts.
    """
    # Load input
    with open(input_path) as f:
        raw = json.load(f)
    inputs = raw if isinstance(raw, list) else [raw]
    print(f"[inference] {len(inputs)} sample(s) loaded from '{input_path}'")

    # Load pipeline
    state = load_pipeline(model_name, checkpoint_path, device)

    # Run
    results = predict_batch(state, inputs)

    # Save outputs
    pred_dir = CONFIG.experiment.predictions_dir
    os.makedirs(pred_dir, exist_ok=True)
    json_path = os.path.join(pred_dir, "inference_results.json")
    csv_path  = os.path.join(pred_dir, "inference_results.csv")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[inference] Saved → {json_path}")

    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"[inference] Saved → {csv_path}")

    # Print summary
    for r in results:
        print(
            f"\n  Q: {r['question'][:80]}"
            f"\n  label={r['verification_label']}  conf={r['confidence']:.3f}  "
            f"flag={r['safety_flag']}  high_risk={r['is_high_risk']}"
            f"\n  reasoning: {r['reasoning']}"
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BioVerify inference — run the full pipeline on new inputs."
    )
    parser.add_argument("--model",      default="tfidf_lr",
                        choices=["tfidf_lr", "distilbert", "biobert", "pubmedbert"])
    parser.add_argument("--checkpoint", default=None,
                        help="Path to checkpoint file (defaults to experiments/<model>/checkpoints/)")
    parser.add_argument("--input",      required=True,
                        help="Path to JSON input file (single dict or list)")
    args = parser.parse_args()

    run_inference(args.model, args.input, checkpoint_path=args.checkpoint)
