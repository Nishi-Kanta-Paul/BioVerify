"""
BioVerify — CLI entry point.

Modes:
  train       Train a model.
  evaluate    Evaluate a trained model on the test set.
  infer       Run inference on new input JSON.
  uncertainty Run confidence-based uncertainty analysis.
  safety      Run patient-safety layer analysis.
  compare     Generate cross-model comparison tables and figures.

Examples:
  python src/main.py --mode train     --model pubmedbert
  python src/main.py --mode train     --model distilbert --epochs 5
  python src/main.py --mode evaluate  --model tfidf_lr
  python src/main.py --mode infer     --model tfidf_lr   --input data/sample_input.json
  python src/main.py --mode uncertainty --model tfidf_lr
  python src/main.py --mode safety    --model tfidf_lr
  python src/main.py --mode compare
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import CONFIG, MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def _mode_train(args: argparse.Namespace) -> None:
    from src.train import train_model
    kwargs = {"max_epochs": args.epochs}
    if args.max_steps:
        kwargs["max_train_steps"] = args.max_steps
    train_model(args.model, **kwargs)


def _mode_evaluate(args: argparse.Namespace) -> None:
    from src.evaluate import run_full_evaluation
    run_full_evaluation(args.model)


def _mode_infer(args: argparse.Namespace) -> None:
    from src.inference import run_inference
    from src.utils import get_device

    if not args.input:
        print("ERROR: --input <path> is required for infer mode.")
        sys.exit(1)
    run_inference(
        model_name      = args.model,
        input_path      = args.input,
        checkpoint_path = args.checkpoint,
        device          = get_device(),
    )


def _mode_uncertainty(args: argparse.Namespace) -> None:
    import pickle
    import numpy as np
    import pandas as pd
    import torch

    from src.config import LABEL2ID
    from src.uncertainty import (
        extract_confidences,
        extract_confidences_tfidf,
        run_uncertainty_analysis,
    )

    model_name    = args.model
    processed_dir = CONFIG.data.processed_dir

    if model_name == "tfidf_lr":
        from src.dataset import prepare_tfidf_data

        pkl_path = os.path.join(
            CONFIG.experiment.experiment_dir, "tfidf_lr", "checkpoints", "best_model.pkl"
        )
        pipeline = pickle.load(open(pkl_path, "rb"))
        val_df  = pd.read_csv(os.path.join(processed_dir, CONFIG.data.val_file))
        test_df = pd.read_csv(os.path.join(processed_dir, CONFIG.data.test_file))
        val_texts,  val_labels  = prepare_tfidf_data(val_df)
        test_texts, test_labels = prepare_tfidf_data(test_df)
        val_data  = extract_confidences_tfidf(pipeline, val_texts,  val_labels)
        test_data = extract_confidences_tfidf(pipeline, test_texts, test_labels)
    else:
        from src.model import build_model
        from src.dataset import get_dataloaders
        from src.utils import get_device

        device    = get_device()
        ckpt_path = os.path.join(
            CONFIG.experiment.experiment_dir, model_name, "checkpoints", "best_model.pth"
        )
        _, val_loader, test_loader = get_dataloaders(
            model_name=model_name,
            batch_size=CONFIG.training.batch_size,
            max_length=CONFIG.model.max_length,
        )
        model = build_model(model_name)
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        val_data  = extract_confidences(model, val_loader,  device)
        test_data = extract_confidences(model, test_loader, device)

    run_uncertainty_analysis(model_name=model_name, val_data=val_data, test_data=test_data)


def _mode_safety(args: argparse.Namespace) -> None:
    import pickle
    import numpy as np
    import pandas as pd
    import torch

    from src.config import LABEL2ID
    from src.safety_layer import run_safety_analysis

    model_name    = args.model
    processed_dir = CONFIG.data.processed_dir
    test_df       = pd.read_csv(os.path.join(processed_dir, CONFIG.data.test_file))
    true_labels   = test_df["label"].map(LABEL2ID).values

    # Load best threshold from uncertainty table
    threshold = CONFIG.uncertainty.default_threshold
    unc_table = os.path.join(CONFIG.experiment.tables_dir, "uncertainty_detection.csv")
    if os.path.exists(unc_table):
        df  = pd.read_csv(unc_table)
        row = df[df["model"] == model_name]
        if not row.empty:
            threshold = float(row.iloc[0]["threshold"])

    if model_name == "tfidf_lr":
        from src.dataset import prepare_tfidf_data
        pkl_path = os.path.join(
            CONFIG.experiment.experiment_dir, "tfidf_lr", "checkpoints", "best_model.pkl"
        )
        pipeline   = pickle.load(open(pkl_path, "rb"))
        texts, _   = prepare_tfidf_data(test_df)
        y_pred     = pipeline.predict(texts)
        y_probs    = pipeline.predict_proba(texts)
        confidences = y_probs.max(axis=1)
    else:
        from src.model import build_model
        from src.dataset import get_dataloaders
        from src.uncertainty import extract_confidences
        from src.utils import get_device

        device    = get_device()
        ckpt_path = os.path.join(
            CONFIG.experiment.experiment_dir, model_name, "checkpoints", "best_model.pth"
        )
        _, _, test_loader = get_dataloaders(
            model_name=model_name,
            batch_size=CONFIG.training.batch_size,
            max_length=CONFIG.model.max_length,
        )
        model = build_model(model_name)
        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        _, y_pred, _, confidences = extract_confidences(model, test_loader, device)

    run_safety_analysis(
        predictions  = np.array(y_pred),
        confidences  = np.array(confidences),
        true_labels  = true_labels,
        threshold    = threshold,
        test_df      = test_df,
        model_name   = model_name,
    )


def _mode_compare(_args: argparse.Namespace) -> None:
    # Import and call compare_results main() directly
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "compare_results",
        os.path.join(os.path.dirname(__file__), "..", "baselines", "compare_results.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

USAGE_EXAMPLES = """
Examples:
  python src/main.py --mode train      --model pubmedbert
  python src/main.py --mode train      --model distilbert --epochs 5
  python src/main.py --mode train      --model tfidf_lr
  python src/main.py --mode evaluate   --model tfidf_lr
  python src/main.py --mode evaluate   --model pubmedbert
  python src/main.py --mode infer      --model tfidf_lr --input data/sample_input.json
  python src/main.py --mode uncertainty --model tfidf_lr
  python src/main.py --mode safety     --model tfidf_lr
  python src/main.py --mode compare
"""

_MODE_HANDLERS = {
    "train":       _mode_train,
    "evaluate":    _mode_evaluate,
    "infer":       _mode_infer,
    "uncertainty": _mode_uncertainty,
    "safety":      _mode_safety,
    "compare":     _mode_compare,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bioverify",
        description="BioVerify — Evidence-Aware Biomedical Answer Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_EXAMPLES,
    )
    parser.add_argument(
        "--mode", required=True,
        choices=list(_MODE_HANDLERS.keys()),
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--model", default="pubmedbert",
        choices=["tfidf_lr", "distilbert", "biobert", "pubmedbert"],
        help="Model to use (default: pubmedbert).",
    )
    parser.add_argument(
        "--epochs", type=int, default=CONFIG.training.max_epochs,
        help=f"Max training epochs (default: {CONFIG.training.max_epochs}).",
    )
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Cap train batches per epoch — for fast dev runs.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Override checkpoint path (inference mode).",
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to JSON input file (infer mode).",
    )
    return parser


def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()
    handler = _MODE_HANDLERS[args.mode]
    handler(args)


if __name__ == "__main__":
    main()
