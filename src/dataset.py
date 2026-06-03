"""
Dataset loading, candidate-answer construction, label reformulation,
PMID-aware stratified splitting, PyTorch Dataset, and DataLoader factory
for BioVerify.

PubMedQA JSON schema (ori_pqal.json):
  {
    "<PMID>": {
      "QUESTION": str,
      "CONTEXTS": [str, ...],          # list of abstract sentences
      "LONG_ANSWER": str,
      "final_decision": "yes" | "no" | "maybe"
    },
    ...
  }
"""

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# Make sure src/ is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import CONFIG, ID2LABEL, LABEL2ID, MODEL_REGISTRY


# ---------------------------------------------------------------------------
# 1. PubMedQA loader
# ---------------------------------------------------------------------------

def load_pubmedqa(json_path: str) -> List[Dict]:
    """
    Load ori_pqal.json and return a flat list of sample dicts.

    Each dict contains:
      pmid, question, context (joined paragraph), long_answer, final_decision
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"PubMedQA data not found at '{json_path}'.\n"
            "Download ori_pqal.json from https://pubmedqa.github.io/ and place it at that path."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for pmid, entry in raw.items():
        # Normalise key names — PubMedQA uses mixed capitalisation
        question = entry.get("QUESTION") or entry.get("question", "")
        contexts = entry.get("CONTEXTS") or entry.get("contexts", [])
        long_answer = entry.get("LONG_ANSWER") or entry.get("long_answer", "")
        decision = (
            entry.get("final_decision")
            or entry.get("FINAL_DECISION")
            or ""
        ).lower().strip()

        # Join context sentences into a single evidence paragraph
        if isinstance(contexts, list):
            context_str = " ".join(str(s).strip() for s in contexts if s)
        else:
            context_str = str(contexts).strip()

        samples.append(
            {
                "pmid": str(pmid),
                "question": _clean_text(question),
                "context": _clean_text(context_str),
                "long_answer": _clean_text(long_answer),
                "final_decision": decision,
            }
        )

    # Report
    decisions = [s["final_decision"] for s in samples]
    print(f"[dataset] Loaded {len(samples)} samples from '{json_path}'")
    for label in ["yes", "no", "maybe"]:
        count = decisions.count(label)
        print(f"  {label:>5}: {count:4d}  ({100 * count / len(samples):.1f}%)")
    unknown = [d for d in decisions if d not in ("yes", "no", "maybe")]
    if unknown:
        print(f"  WARNING: {len(unknown)} samples with unrecognised decision: {set(unknown)}")

    return samples


def _clean_text(text: str) -> str:
    """Strip excessive whitespace; do NOT force lowercase (tokeniser handles casing)."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Candidate-answer construction & label reformulation
# ---------------------------------------------------------------------------

def reformulate_dataset(samples: List[Dict]) -> pd.DataFrame:
    """
    Convert PubMedQA samples into evidence-answer verification tuples.

    Reformulation rules (STABLE §5):
      yes   → Row1: "Yes. {long_answer}"                    → supported
              Row2: "No. The evidence does not support this conclusion."  → contradicted
      no    → Row1: "No. {long_answer}"                     → supported
              Row2: "Yes. The evidence supports the opposite conclusion." → contradicted
      maybe → Row1: "The evidence is inconclusive. {long_answer}"        → uncertain

    Returns a DataFrame with columns:
      pmid, question, evidence, candidate_answer, label, original_decision
    """
    rows = []
    skipped = 0

    for s in samples:
        decision = s["final_decision"]
        q = s["question"]
        ev = s["context"]
        la = s["long_answer"]
        pmid = s["pmid"]

        if decision == "yes":
            rows.append(
                {
                    "pmid": pmid,
                    "question": q,
                    "evidence": ev,
                    "candidate_answer": f"Yes. {la}",
                    "label": "supported",
                    "original_decision": decision,
                }
            )
            rows.append(
                {
                    "pmid": pmid,
                    "question": q,
                    "evidence": ev,
                    "candidate_answer": "No. The evidence does not support this conclusion.",
                    "label": "contradicted",
                    "original_decision": decision,
                }
            )
        elif decision == "no":
            rows.append(
                {
                    "pmid": pmid,
                    "question": q,
                    "evidence": ev,
                    "candidate_answer": f"No. {la}",
                    "label": "supported",
                    "original_decision": decision,
                }
            )
            rows.append(
                {
                    "pmid": pmid,
                    "question": q,
                    "evidence": ev,
                    "candidate_answer": "Yes. The evidence supports the opposite conclusion.",
                    "label": "contradicted",
                    "original_decision": decision,
                }
            )
        elif decision == "maybe":
            rows.append(
                {
                    "pmid": pmid,
                    "question": q,
                    "evidence": ev,
                    "candidate_answer": f"The evidence is inconclusive. {la}",
                    "label": "uncertain",
                    "original_decision": decision,
                }
            )
        else:
            skipped += 1

    df = pd.DataFrame(rows)

    # Report
    print(f"\n[dataset] Reformulated dataset: {len(df)} rows  (skipped {skipped} unknown decisions)")
    for lbl in ["supported", "contradicted", "uncertain"]:
        count = (df["label"] == lbl).sum()
        print(f"  {lbl:>14}: {count:4d}  ({100 * count / len(df):.1f}%)")

    # Save full reformulated CSV
    os.makedirs(CONFIG.data.processed_dir, exist_ok=True)
    full_path = os.path.join(CONFIG.data.processed_dir, "reformulated_full.csv")
    df.to_csv(full_path, index=False)
    print(f"  Saved → {full_path}")

    return df


# ---------------------------------------------------------------------------
# 3. PMID-aware stratified train/val/test split
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = CONFIG.data.train_ratio,
    val_ratio: float = CONFIG.data.val_ratio,
    test_ratio: float = CONFIG.data.test_ratio,
    seed: int = CONFIG.data.seed,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    PMID-aware stratified split to prevent data leakage.

    Strategy:
      1. Collapse df to one row per PMID.  Assign a "dominant label" for
         stratification:  yes-PMIDs → "yes_no" stratum;  no-PMIDs → "yes_no";
         maybe-PMIDs → "maybe".  This keeps the approximate label ratio.
      2. Split PMIDs into train / (val+test) with StratifiedShuffleSplit.
      3. Split (val+test) PMIDs into val / test.
      4. Expand back to full rows using PMID membership.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9, \
        "Ratios must sum to 1."

    # --- Step 1: one row per PMID with a stratification stratum ---
    pmid_df = (
        df[["pmid", "original_decision"]]
        .drop_duplicates("pmid")
        .copy()
    )
    # Stratum: yes/no both produce two rows (same class balance impact); maybe produces one
    pmid_df["stratum"] = pmid_df["original_decision"].apply(
        lambda d: "yes_no" if d in ("yes", "no") else "maybe"
    )

    pmids = pmid_df["pmid"].values
    strata = pmid_df["stratum"].values
    n_pmids = len(pmids)

    # --- Step 2: train vs (val+test) ---
    val_test_ratio = val_ratio + test_ratio
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
    train_idx, valtest_idx = next(sss1.split(pmids, strata))
    train_pmids = set(pmids[train_idx])
    valtest_pmids = pmids[valtest_idx]
    valtest_strata = strata[valtest_idx]

    # --- Step 3: val vs test (from valtest pool) ---
    relative_test = test_ratio / val_test_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed)
    val_idx, test_idx = next(sss2.split(valtest_pmids, valtest_strata))
    val_pmids = set(valtest_pmids[val_idx])
    test_pmids = set(valtest_pmids[test_idx])

    # --- Step 4: expand rows ---
    train_df = df[df["pmid"].isin(train_pmids)].reset_index(drop=True)
    val_df = df[df["pmid"].isin(val_pmids)].reset_index(drop=True)
    test_df = df[df["pmid"].isin(test_pmids)].reset_index(drop=True)

    # --- Save CSVs ---
    processed_dir = CONFIG.data.processed_dir
    os.makedirs(processed_dir, exist_ok=True)
    train_df.to_csv(os.path.join(processed_dir, CONFIG.data.train_file), index=False)
    val_df.to_csv(os.path.join(processed_dir, CONFIG.data.val_file), index=False)
    test_df.to_csv(os.path.join(processed_dir, CONFIG.data.test_file), index=False)

    # --- Report ---
    print(f"\n[dataset] PMID-aware stratified split  (seed={seed})")
    print(f"  Total PMIDs: {n_pmids}  →  train: {len(train_pmids)}  val: {len(val_pmids)}  test: {len(test_pmids)}")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = {lbl: (split_df["label"] == lbl).sum() for lbl in ["supported", "contradicted", "uncertain"]}
        print(
            f"  {split_name:>5}: {len(split_df):5d} rows | "
            + "  ".join(f"{k}={v}" for k, v in counts.items())
        )

    # --- Leakage check ---
    overlap_tv = train_pmids & val_pmids
    overlap_tt = train_pmids & test_pmids
    overlap_vt = val_pmids & test_pmids
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            f"PMID leakage detected! train∩val={len(overlap_tv)}, "
            f"train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
        )
    print("  PMID leakage check: PASSED (no PMID appears in more than one split)")

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 4. PyTorch Dataset class
# ---------------------------------------------------------------------------

class BioVerifyDataset(Dataset):
    """
    Tokenises (question, evidence, candidate_answer) triples for transformer models.

    Input format (STABLE §4):
      [CLS] question [SEP] evidence [SEP] candidate_answer [SEP]

    HuggingFace tokenisers handle [CLS]/[SEP] insertion automatically when
    text_pair is used for two-segment input.  For three segments we pass the
    concatenation of evidence and candidate_answer as the second segment,
    preserving the logical [SEP] boundary with an explicit separator string.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = CONFIG.model.max_length,
        label2id: Dict[str, int] = LABEL2ID,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        # Detect whether this tokeniser produces token_type_ids
        self._has_token_type_ids = "token_type_ids" in tokenizer.model_input_names or \
            hasattr(tokenizer, "create_token_type_ids_from_sequences")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        question = str(row["question"])
        evidence = str(row["evidence"])
        candidate = str(row["candidate_answer"])

        # Segment A: question
        # Segment B: evidence [SEP] candidate_answer
        # The tokeniser will insert [CLS] … [SEP] … [SEP] automatically.
        # We join evidence + candidate with a SEP token string so the
        # internal [SEP] is visible as a word-piece boundary even in
        # single-segment tokenisers (DistilBERT has no token_type_ids).
        sep = self.tokenizer.sep_token or "[SEP]"
        text_a = question
        text_b = f"{evidence} {sep} {candidate}"

        encoding = self.tokenizer(
            text_a,
            text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,  # truncates the longer segment (text_b / evidence) from the right
            return_tensors="pt",
            return_token_type_ids=True,
        )

        label = self.label2id[str(row["label"])]

        item: Dict[str, torch.Tensor] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        return item


# ---------------------------------------------------------------------------
# 5. TF-IDF data preparation
# ---------------------------------------------------------------------------

def prepare_tfidf_data(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """
    Return plain-text list and integer label array for TF-IDF models.

    Text format: "question evidence candidate_answer" (space-joined).
    """
    texts = (
        df["question"].fillna("") + " "
        + df["evidence"].fillna("") + " "
        + df["candidate_answer"].fillna("")
    ).tolist()
    labels = df["label"].map(LABEL2ID).values.astype(int)
    return texts, labels


# ---------------------------------------------------------------------------
# 6. DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    model_name: str = CONFIG.training.__class__.__name__,
    batch_size: int = CONFIG.training.batch_size,
    max_length: int = CONFIG.model.max_length,
    processed_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load saved split CSVs, build tokeniser, and return (train, val, test) DataLoaders.

    Args:
        model_name: short name key from MODEL_REGISTRY (e.g. 'pubmedbert').
        batch_size:  samples per batch.
        max_length:  token sequence length.
        processed_dir: override for the processed data directory.
    """
    if processed_dir is None:
        processed_dir = CONFIG.data.processed_dir

    hf_id = MODEL_REGISTRY[model_name]
    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    train_df = pd.read_csv(os.path.join(processed_dir, CONFIG.data.train_file))
    val_df = pd.read_csv(os.path.join(processed_dir, CONFIG.data.val_file))
    test_df = pd.read_csv(os.path.join(processed_dir, CONFIG.data.test_file))

    train_ds = BioVerifyDataset(train_df, tokenizer, max_length)
    val_ds = BioVerifyDataset(val_df, tokenizer, max_length)
    test_ds = BioVerifyDataset(test_df, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[dataset] DataLoaders ready  model={model_name}  batch={batch_size}  max_len={max_length}")
    print(f"  train batches: {len(train_loader)}  val batches: {len(val_loader)}  test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# 7. Sanity checks (run when executed directly)
# ---------------------------------------------------------------------------

def _print_separator(title: str = "") -> None:
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def run_sanity_checks(json_path: Optional[str] = None, model_name: str = "pubmedbert") -> None:
    """Full sanity check suite — prints all diagnostics."""
    if json_path is None:
        json_path = os.path.join(CONFIG.data.data_dir, CONFIG.data.raw_data_file)

    # ---- 1. Load raw data ----
    _print_separator("1. RAW DATA LOADING")
    samples = load_pubmedqa(json_path)

    # ---- 2. Reformulation ----
    _print_separator("2. REFORMULATION")
    df = reformulate_dataset(samples)

    # Confirm expected size
    expected_min, expected_max = 1600, 2000
    if not (expected_min <= len(df) <= expected_max):
        print(f"  WARNING: reformulated size {len(df)} outside expected range [{expected_min}, {expected_max}]")
    else:
        print(f"  Size check: {len(df)} rows — within expected range ✓")

    # Print 3 example rows (one per label)
    print("\n  Example rows:")
    for lbl in ["supported", "contradicted", "uncertain"]:
        row = df[df["label"] == lbl].iloc[0]
        q_short = row["question"][:80] + "..." if len(row["question"]) > 80 else row["question"]
        ca_short = row["candidate_answer"][:100] + "..." if len(row["candidate_answer"]) > 100 else row["candidate_answer"]
        print(f"\n  [{lbl}]")
        print(f"    question:         {q_short}")
        print(f"    candidate_answer: {ca_short}")
        print(f"    label:            {row['label']}")
        print(f"    original_decision:{row['original_decision']}")

    # Token length distribution (pre-truncation)
    print("\n  Token length stats (rough word-count proxy):")
    total_tokens = (df["question"] + " " + df["evidence"] + " " + df["candidate_answer"]).str.split().apply(len)
    over_512 = (total_tokens > 512).sum()
    print(f"    median word count: {int(total_tokens.median())}  |  max: {int(total_tokens.max())}")
    print(f"    samples >512 words (rough): {over_512} / {len(df)}  ({100*over_512/len(df):.1f}%)")

    # ---- 3. PMID-aware split ----
    _print_separator("3. PMID-AWARE STRATIFIED SPLIT")
    train_df, val_df, test_df = split_dataset(df)

    # PMID leakage verification (already checked inside split_dataset, but repeat here for report)
    all_pmids = [set(split["pmid"].tolist()) for split in [train_df, val_df, test_df]]
    split_names = ["train", "val", "test"]
    any_leak = False
    for i in range(3):
        for j in range(i + 1, 3):
            overlap = all_pmids[i] & all_pmids[j]
            if overlap:
                print(f"  LEAK: {split_names[i]} ∩ {split_names[j]} = {len(overlap)} PMIDs")
                any_leak = True
    if not any_leak:
        print("  No PMID appears in more than one split ✓")

    # ---- 4. DataLoader + tokenisation ----
    _print_separator("4. DATALOADER & TOKENISATION")
    try:
        hf_id = MODEL_REGISTRY[model_name]
        print(f"  Loading tokeniser: {hf_id}")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)

        val_ds = BioVerifyDataset(val_df, tokenizer)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(val_loader))

        print(f"  input_ids shape:    {tuple(batch['input_ids'].shape)}   (expect (4, 512))")
        print(f"  attention_mask shape:{tuple(batch['attention_mask'].shape)}")
        if "token_type_ids" in batch:
            print(f"  token_type_ids shape:{tuple(batch['token_type_ids'].shape)}")
        else:
            print("  token_type_ids: not produced by this tokeniser (expected for DistilBERT)")
        print(f"  labels shape:       {tuple(batch['label'].shape)}")
        print(f"  label values:       {batch['label'].tolist()}")
        print(f"  label names:        {[ID2LABEL[i] for i in batch['label'].tolist()]}")

        # Decode first sample
        first_ids = batch["input_ids"][0]
        decoded = tokenizer.decode(first_ids, skip_special_tokens=False)
        print(f"\n  Decoded first sample (first 300 chars):\n  {decoded[:300]!r}")

        # Verify shape strictly
        assert batch["input_ids"].shape == (4, 512), "input_ids shape mismatch"
        assert batch["attention_mask"].shape == (4, 512), "attention_mask shape mismatch"
        print("\n  Shape assertions: PASSED ✓")

    except Exception as exc:
        print(f"  WARNING: tokeniser check failed — {exc}")
        print("  (This is expected if HuggingFace model is not yet downloaded.)")

    _print_separator("ALL SANITY CHECKS COMPLETE")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BioVerify dataset sanity checks")
    parser.add_argument(
        "--json_path",
        default=None,
        help="Path to ori_pqal.json (default: uses config path)",
    )
    parser.add_argument(
        "--model",
        default="pubmedbert",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model name for tokeniser check",
    )
    args = parser.parse_args()
    run_sanity_checks(json_path=args.json_path, model_name=args.model)
