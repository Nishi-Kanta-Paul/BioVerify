# BioVerify

**Evidence-Aware Biomedical Answer Verification with Uncertainty Detection for Patient-Safe Healthcare NLP**

BioVerify is a lightweight, reproducible framework that classifies candidate biomedical answers as **supported**, **contradicted**, or **uncertain** based on PubMed abstract evidence, then applies confidence-based uncertainty detection and a rule-based patient-safety layer to produce a final safety flag.

---

## Pipeline Overview

```
PubMedQA (ori_pqal.json)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 — Candidate-Answer Construction & Reformulation    │
│  yes/no → 2 rows (supported + contradicted)                 │
│  maybe  → 1 row  (uncertain)            ~1,890 samples      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2 — Encoder + Classification Head                    │
│  [CLS] Question [SEP] Evidence [SEP] Candidate [SEP]        │
│  → Dropout(0.3) → Linear(768,3) → {supported/contra/unc}   │
│                                                             │
│  Models compared:                                           │
│   • TF-IDF + Logistic Regression (baseline)                 │
│   • DistilBERT  (general transformer baseline)              │
│   • BioBERT     (biomedical transformer)                    │
│   • PubMedBERT  (proposed — trained from scratch on PubMed) │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3 — Confidence-Based Uncertainty Detection           │
│  conf = max(softmax(logits))                                │
│  if conf < τ  →  override to uncertain / expert_review      │
│  τ tuned on val set by maximising uncertain-class recall    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4 — Rule-Based Patient-Safety Layer                  │
│  Keyword matching across 6 high-risk medical categories     │
│  → safe  /  unsafe  /  expert_review                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
      {verification_label, confidence, safety_flag, reasoning}
```

---

## Installation

```bash
# Clone the repository
git clone <repo_url>
cd BioVerify

# Install dependencies (Python 3.9+)
pip install -r requirements.txt
```

---

## Data Preparation

BioVerify uses the **PubMedQA labeled subset (PQA-L)** — 1,000 expert-annotated biomedical Q&A samples.

1. Download `ori_pqal.json` from the official repository:
   ```bash
   # Option A — direct download
   curl -L "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json" \
        -o data/pubmedqa/ori_pqal.json

   # Option B — clone PubMedQA repo and copy
   git clone https://github.com/pubmedqa/pubmedqa /tmp/pubmedqa
   cp /tmp/pubmedqa/data/ori_pqal.json data/pubmedqa/ori_pqal.json
   ```

2. Verify the file exists at `data/pubmedqa/ori_pqal.json`.

The dataset is **reformulated automatically** on first run:
- `yes`/`no` samples → 2 rows each (supported + contradicted)
- `maybe` samples → 1 row (uncertain)
- Result: ~1,890 rows saved to `data/pubmedqa/processed/`

---

## Usage

### Full pipeline (recommended — run on GPU)

```bash
bash scripts/train.sh
```

### Individual commands

```bash
# Train a model
python src/main.py --mode train --model pubmedbert
python src/main.py --mode train --model tfidf_lr

# Evaluate on test set
python src/main.py --mode evaluate --model pubmedbert

# Uncertainty analysis (tune τ on val, apply to test)
python src/main.py --mode uncertainty --model pubmedbert

# Safety analysis
python src/main.py --mode safety --model pubmedbert

# Cross-model comparison (generates all tables and figures)
python baselines/compare_results.py

# Inference on new inputs
python src/main.py --mode infer --model pubmedbert --input data/sample_input.json
```

### Inference input format

Create a JSON file (`data/sample_input.json`):

```json
[
  {
    "question": "Does aspirin reduce the risk of heart attack?",
    "evidence": "A randomized trial of 22,071 participants showed aspirin reduced first MI risk by 44%.",
    "candidate_answer": "Yes, aspirin reduces heart attack risk based on trial evidence."
  }
]
```

Expected output:

```json
{
  "question": "Does aspirin reduce the risk of heart attack?",
  "verification_label": "supported",
  "confidence": 0.94,
  "is_high_risk": true,
  "high_risk_categories": ["drug_dosage", "treatment_decision"],
  "safety_flag": "safe",
  "reasoning": "Predicted 'supported' with confidence 0.940 (τ=0.700). High-risk categories: ['drug_dosage', 'treatment_decision']. Safety flag assigned: safe."
}
```

---

## Project Structure

```
BioVerify/
├── data/
│   └── pubmedqa/
│       ├── ori_pqal.json              # PubMedQA PQA-L (download separately)
│       └── processed/                 # Auto-generated reformulated splits
├── src/
│   ├── config.py                      # All hyperparameters and paths
│   ├── dataset.py                     # Data loading, reformulation, DataLoaders
│   ├── model.py                       # TransformerClassifier + TFIDFClassifier
│   ├── train.py                       # Training loop (transformer + TF-IDF)
│   ├── evaluate.py                    # Metrics, plots, tables
│   ├── uncertainty.py                 # Confidence-based uncertainty detection
│   ├── safety_layer.py                # Rule-based patient-safety layer
│   ├── inference.py                   # Full inference pipeline
│   ├── utils.py                       # Seed, device, checkpointing, logging
│   └── main.py                        # CLI entry point
├── baselines/
│   ├── train_tfidf_lr.py              # TF-IDF + LR full pipeline wrapper
│   ├── train_distilbert.py            # DistilBERT full pipeline wrapper
│   └── compare_results.py             # Cross-model comparison
├── experiments/<model_name>/
│   ├── checkpoints/best_model.{pth,pkl}
│   ├── logs/train_log.csv
│   └── results.json
├── outputs/
│   ├── figures/                       # All plots (PNG)
│   ├── tables/                        # All CSVs
│   └── predictions/                   # Inference results
├── scripts/
│   └── train.sh                       # Full reproducible pipeline script
├── requirements.txt
└── README.md
```

---

## Results

*Populated after full training on GPU. BioBERT and PubMedBERT require a T4 or equivalent.*

### Main Results (test set, 283 samples)

| Model | Accuracy | Macro-F1 | Contradiction F1 | Uncertain F1 |
|---|---|---|---|---|
| TF-IDF + LR | — | — | — | — |
| DistilBERT | — | — | — | — |
| BioBERT | — | — | — | — |
| PubMedBERT | — | — | — | — |

### Safety Layer Results

| Model | Unsafe Catch Rate | False Safe Rate |
|---|---|---|
| TF-IDF + LR | — | — |
| DistilBERT | — | — |
| BioBERT | — | — |
| PubMedBERT | — | — |

*Run `python baselines/compare_results.py` after training to populate these tables.*

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Max sequence length | 512 |
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 16 (grad accum × 2 = effective 32) |
| Max epochs | 10 |
| Early stopping patience | 3 |
| Confidence threshold τ | Tuned on val set |
| Seed | 42 |

---

## Ablation Study

The framework ablates four conditions to measure each component's contribution:

| Variant | Description |
|---|---|
| Full pipeline | Model + uncertainty override + safety layer |
| Without uncertainty | Skip confidence threshold override |
| Without safety | Skip keyword-based safety flagging |
| Without both | Raw model predictions only |

---

## Citation

```bibtex
@misc{bioVerify2024,
  title   = {Evidence-Aware Biomedical Answer Verification with Uncertainty Detection
             for Patient-Safe Healthcare NLP},
  author  = {},
  year    = {2026},
  note    = {BioVerify framework}
}
```

---

## Environment

- Python 3.9+
- PyTorch 2.x
- HuggingFace Transformers 4.x / 5.x
- scikit-learn 1.x
- GPU recommended for transformer training (Colab T4 or equivalent)
