"""
Model definitions for BioVerify.

Components:
  TransformerClassifier  — fine-tunable HuggingFace encoder + classification head
  TFIDFClassifier        — sklearn TF-IDF + Logistic Regression pipeline
  build_model()          — factory dispatching on model name
  get_tokenizer()        — tokeniser factory
"""

import os
import pickle
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import CONFIG, MODEL_REGISTRY, NUM_CLASSES


# ---------------------------------------------------------------------------
# 1. Transformer Classifier
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """
    Encoder + classification head for evidence-answer verification.

    Architecture (STABLE §6):
      [CLS] hidden state (B, hidden_dim)
        → Dropout(dropout_rate)
        → Linear(hidden_dim, num_classes)
        → logits (B, num_classes)
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = CONFIG.model.dropout_rate,
    ):
        super().__init__()
        hf_id = _resolve_hf_id(model_name)
        self.model_name = model_name
        self.hf_id = hf_id

        self.encoder = AutoModel.from_pretrained(hf_id)
        hidden_dim = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, seq_len)
            attention_mask: (B, seq_len)
            token_type_ids: (B, seq_len) or None

        Returns:
            logits: (B, num_classes)
        """
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        # token_type_ids are supported by BERT variants but not DistilBERT
        if token_type_ids is not None and self._accepts_token_type_ids():
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        cls_hidden = outputs.last_hidden_state[:, 0, :]          # (B, hidden_dim)
        logits = self.classifier(self.dropout(cls_hidden))        # (B, num_classes)
        return logits

    def _accepts_token_type_ids(self) -> bool:
        """Return True if the encoder accepts token_type_ids."""
        sig_params = self.encoder.forward.__code__.co_varnames
        return "token_type_ids" in sig_params

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to softmax probabilities. Shape: (B, num_classes)."""
        return F.softmax(logits, dim=-1)

    def get_predictions(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (predicted_class_ids, confidence_scores).

        confidence = max(softmax(logits))  per sample.
        """
        probs = self.get_probabilities(logits)
        confidence, predictions = probs.max(dim=-1)
        return predictions, confidence

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 2. TF-IDF Classifier
# ---------------------------------------------------------------------------

class TFIDFClassifier:
    """
    sklearn TF-IDF + Logistic Regression pipeline (STABLE §6).

    Input:  list of plain text strings ("question evidence candidate_answer")
    Output: predicted labels and class probabilities
    """

    def __init__(
        self,
        max_features: int = CONFIG.tfidf.max_features,
        ngram_range: Tuple[int, int] = CONFIG.tfidf.ngram_range,
        sublinear_tf: bool = CONFIG.tfidf.sublinear_tf,
        C: float = CONFIG.tfidf.lr_C,
        class_weight: str = CONFIG.tfidf.lr_class_weight,
        max_iter: int = CONFIG.tfidf.lr_max_iter,
        solver: str = CONFIG.tfidf.lr_solver,
    ):
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        sublinear_tf=sublinear_tf,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        class_weight=class_weight,
                        max_iter=max_iter,
                        solver=solver,
                    ),
                ),
            ]
        )
        self.is_fitted = False

    def fit(self, texts, labels) -> "TFIDFClassifier":
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        return self

    def predict(self, texts):
        if not self.is_fitted:
            raise RuntimeError("TFIDFClassifier must be fitted before calling predict().")
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        if not self.is_fitted:
            raise RuntimeError("TFIDFClassifier must be fitted before calling predict_proba().")
        return self.pipeline.predict_proba(texts)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        print(f"[TFIDFClassifier] Saved to {path}")

    def load(self, path: str) -> "TFIDFClassifier":
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.is_fitted = True
        print(f"[TFIDFClassifier] Loaded from {path}")
        return self


# ---------------------------------------------------------------------------
# 3. Model factory
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = CONFIG.model.dropout_rate,
) -> "TransformerClassifier | TFIDFClassifier":
    """
    Factory returning the appropriate model instance.

    Args:
        model_name:   'tfidf_lr' | 'distilbert' | 'biobert' | 'pubmedbert'
        num_classes:  number of output classes (default: 3)
        dropout_rate: dropout applied before linear head (transformers only)

    Returns:
        TransformerClassifier or TFIDFClassifier
    """
    if model_name == "tfidf_lr":
        model = TFIDFClassifier()
        print(f"[build_model] TFIDFClassifier  (sklearn pipeline, no GPU)")
        return model

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {['tfidf_lr'] + list(MODEL_REGISTRY.keys())}"
        )

    model = TransformerClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )
    total = model.count_parameters()
    trainable = model.count_trainable_parameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[build_model] {model_name:<12} | HF: {model.hf_id}\n"
        f"              params: {total:,}  trainable: {trainable:,}  device: {device}"
    )
    return model


# ---------------------------------------------------------------------------
# 4. Tokeniser factory
# ---------------------------------------------------------------------------

def get_tokenizer(model_name: str) -> AutoTokenizer:
    """Return an AutoTokenizer for the given short model name."""
    hf_id = _resolve_hf_id(model_name)
    return AutoTokenizer.from_pretrained(hf_id)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _resolve_hf_id(model_name: str) -> str:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Valid names: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


# ---------------------------------------------------------------------------
# 5. Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    param_summary: Dict[str, int] = {}

    # ---- Transformer models ----
    for model_name in ["distilbert", "biobert", "pubmedbert"]:
        print("=" * 60)
        print(f"  {model_name.upper()}")
        print("=" * 60)

        model = build_model(model_name)
        model.eval()
        model.to(device)

        total_params = model.count_parameters()
        param_summary[model_name] = total_params

        # Dummy input
        B, L = 2, 512
        input_ids = torch.ones(B, L, dtype=torch.long, device=device)
        attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
        token_type_ids = torch.zeros(B, L, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)

        assert logits.shape == (B, NUM_CLASSES), \
            f"Expected logits shape ({B}, {NUM_CLASSES}), got {tuple(logits.shape)}"

        probs = model.get_probabilities(logits)
        preds, confs = model.get_predictions(logits)

        print(f"  logits shape:       {tuple(logits.shape)}  ✓")
        print(f"  probs (first sample):{probs[0].tolist()}")
        print(f"  prediction:         {preds.tolist()}  confidence: {confs.tolist()}")
        print()

    # ---- TF-IDF ----
    print("=" * 60)
    print("  TFIDF_LR")
    print("=" * 60)
    tfidf_model = build_model("tfidf_lr")
    dummy_texts = [
        "Does aspirin reduce heart attack risk? Clinical trial evidence shows a 44% reduction. Yes aspirin reduces risk.",
        "Does ibuprofen cause kidney damage? Studies show NSAIDs associated with renal impairment. No the evidence does not support this conclusion.",
        "Is vitamin D effective for COVID prevention? Evidence is inconclusive at this time.",
    ]
    dummy_labels = [0, 1, 2]   # supported, contradicted, uncertain

    tfidf_model.fit(dummy_texts, dummy_labels)
    preds = tfidf_model.predict(dummy_texts)
    probas = tfidf_model.predict_proba(dummy_texts)
    print(f"  predictions:  {preds.tolist()}")
    print(f"  probabilities (shape): {probas.shape}")
    print(f"  proba row 0:  {probas[0].tolist()}")
    param_summary["tfidf_lr"] = 0

    # ---- Summary table ----
    print("\n" + "=" * 60)
    print("  PARAMETER COUNT SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<14} {'Total Params':>15}  {'HF ID'}")
    print(f"  {'-'*14} {'-'*15}  {'-'*45}")
    for name, count in param_summary.items():
        hf = MODEL_REGISTRY.get(name, "sklearn (no params)")
        count_str = f"{count:,}" if count else "—"
        print(f"  {name:<14} {count_str:>15}  {hf}")


if __name__ == "__main__":
    run_sanity_checks()
