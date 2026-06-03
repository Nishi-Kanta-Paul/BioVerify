#!/bin/bash
# BioVerify — Full reproducible training pipeline
# Run on a GPU machine (Colab T4 recommended for BioBERT / PubMedBERT).
# Usage:  bash scripts/train.sh
set -e

echo "================================================================"
echo "  BioVerify: Full Pipeline"
echo "================================================================"

# ── 0. Dependencies ─────────────────────────────────────────────────
echo ""
echo "--- Installing dependencies ---"
pip install -r requirements.txt --quiet

# ── 1. Data check ───────────────────────────────────────────────────
echo ""
echo "--- Checking data ---"
if [ ! -f "data/pubmedqa/ori_pqal.json" ]; then
  echo "ERROR: data/pubmedqa/ori_pqal.json not found."
  echo "Download it from https://github.com/pubmedqa/pubmedqa"
  echo "and place it at data/pubmedqa/ori_pqal.json"
  exit 1
fi
echo "Data file found: data/pubmedqa/ori_pqal.json"

# ── 2. Train all models ─────────────────────────────────────────────
echo ""
echo "--- Training TF-IDF + Logistic Regression ---"
python baselines/train_tfidf_lr.py

echo ""
echo "--- Training DistilBERT ---"
python src/main.py --mode train --model distilbert

echo ""
echo "--- Training BioBERT ---"
python src/main.py --mode train --model biobert

echo ""
echo "--- Training PubMedBERT ---"
python src/main.py --mode train --model pubmedbert

# ── 3. Evaluate all models ──────────────────────────────────────────
echo ""
echo "--- Evaluating DistilBERT ---"
python src/main.py --mode evaluate --model distilbert

echo ""
echo "--- Evaluating BioBERT ---"
python src/main.py --mode evaluate --model biobert

echo ""
echo "--- Evaluating PubMedBERT ---"
python src/main.py --mode evaluate --model pubmedbert

# ── 4. Uncertainty analysis (all transformer models) ────────────────
echo ""
echo "--- Uncertainty analysis: DistilBERT ---"
python src/main.py --mode uncertainty --model distilbert

echo ""
echo "--- Uncertainty analysis: BioBERT ---"
python src/main.py --mode uncertainty --model biobert

echo ""
echo "--- Uncertainty analysis: PubMedBERT ---"
python src/main.py --mode uncertainty --model pubmedbert

# ── 5. Safety analysis (all models) ────────────────────────────────
echo ""
echo "--- Safety analysis: DistilBERT ---"
python src/main.py --mode safety --model distilbert

echo ""
echo "--- Safety analysis: BioBERT ---"
python src/main.py --mode safety --model biobert

echo ""
echo "--- Safety analysis: PubMedBERT ---"
python src/main.py --mode safety --model pubmedbert

# ── 6. Cross-model comparison ───────────────────────────────────────
echo ""
echo "--- Generating comparison tables and figures ---"
python baselines/compare_results.py

# ── 7. Sample inference ─────────────────────────────────────────────
echo ""
echo "--- Running sample inference (PubMedBERT) ---"
if [ -f "data/sample_input.json" ]; then
  python src/main.py --mode infer --model pubmedbert --input data/sample_input.json
else
  echo "Skipping inference: data/sample_input.json not found."
fi

echo ""
echo "================================================================"
echo "  Pipeline complete. Results are in outputs/"
echo "================================================================"
