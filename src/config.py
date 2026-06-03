"""
Central hyperparameter registry for BioVerify.
All paths are relative to the project root.
Import this module to access any configuration value.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

LABEL2ID: Dict[str, int] = {
    "supported": 0,
    "contradicted": 1,
    "uncertain": 2,
}

ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

CLASS_NAMES: List[str] = ["supported", "contradicted", "uncertain"]
NUM_CLASSES: int = 3


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, str] = {
    "distilbert": "distilbert-base-uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
}

DEFAULT_MODEL: str = "pubmedbert"


# ---------------------------------------------------------------------------
# Model architecture config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    hidden_dim: int = 768
    dropout_rate: float = 0.3
    max_length: int = 512
    num_classes: int = NUM_CLASSES


# ---------------------------------------------------------------------------
# Transformer training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    max_epochs: int = 10
    early_stopping_patience: int = 3
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    seed: int = 42


# ---------------------------------------------------------------------------
# TF-IDF + Logistic Regression config
# ---------------------------------------------------------------------------

@dataclass
class TFIDFConfig:
    max_features: int = 10_000
    ngram_range: Tuple[int, int] = (1, 2)
    sublinear_tf: bool = True
    lr_C: float = 1.0
    lr_max_iter: int = 1_000
    lr_class_weight: str = "balanced"
    lr_solver: str = "lbfgs"


# ---------------------------------------------------------------------------
# Uncertainty detection config
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyConfig:
    threshold_search_grid: List[float] = field(
        default_factory=lambda: [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    )
    # Minimum overall accuracy that must be maintained when tuning τ
    accuracy_floor: float = 0.80
    # Default fallback threshold if no τ meets accuracy_floor on val set
    default_threshold: float = 0.70


# ---------------------------------------------------------------------------
# Patient-safety layer config
# ---------------------------------------------------------------------------

HIGH_RISK_KEYWORDS: Dict[str, List[str]] = {
    "emergency": [
        "emergency", "urgent", "life-threatening", "cardiac arrest",
        "stroke", "anaphylaxis", "seizure", "resuscitation", "cpr",
        "911", "ambulance", "unconscious", "hemorrhage", "shock",
    ],
    "drug_dosage": [
        "dosage", "dose", "mg", "milligram", "microgram", "mcg",
        "overdose", "drug interaction", "contraindication", "prescription",
        "medication", "pharmaceutical", "adverse drug", "toxicity",
        "lethal dose", "therapeutic index",
        "ibuprofen", "aspirin", "metformin", "paracetamol", "acetaminophen",
        "nsaid", "antibiotic", "analgesic", "antidepressant",
    ],
    "pregnancy": [
        "pregnancy", "pregnant", "fetal", "fetus", "trimester",
        "breastfeeding", "teratogenic", "prenatal", "postnatal",
        "miscarriage", "gestational", "lactation", "neonatal",
        "obstetric", "maternal",
    ],
    "surgery": [
        "surgery", "surgical", "operation", "transplant", "anesthesia",
        "postoperative", "preoperative", "incision", "resection",
        "laparoscopic", "biopsy", "amputation", "implant", "prosthesis",
        "invasive procedure",
    ],
    "severe_disease": [
        "cancer", "tumor", "malignant", "hiv", "aids", "sepsis",
        "renal failure", "cirrhosis", "myocardial infarction", "heart attack",
        "pulmonary embolism", "meningitis", "ebola", "terminal",
        "metastatic", "end-stage",
        "diabetes", "diabetic", "hypertension", "chronic kidney disease",
        "liver failure", "organ failure", "autoimmune",
    ],
    "treatment_decision": [
        "treatment", "therapy", "chemotherapy", "radiation",
        "prescribe", "discontinue", "immunotherapy", "clinical trial",
        "off-label", "second opinion", "taper", "withdraw", "initiate treatment",
        "stop medication", "change therapy",
    ],
}


@dataclass
class SafetyConfig:
    # Categories from HIGH_RISK_KEYWORDS
    high_risk_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: HIGH_RISK_KEYWORDS
    )


# ---------------------------------------------------------------------------
# Data paths config
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    data_dir: str = "data/pubmedqa"
    raw_data_file: str = "ori_pqal.json"
    processed_dir: str = "data/pubmedqa/processed"
    train_file: str = "reformulated_train.csv"
    val_file: str = "reformulated_val.csv"
    test_file: str = "reformulated_test.csv"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


# ---------------------------------------------------------------------------
# Experiment & output paths config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    experiment_dir: str = "experiments"
    output_dir: str = "outputs"
    figures_dir: str = "outputs/figures"
    tables_dir: str = "outputs/tables"
    predictions_dir: str = "outputs/predictions"


# ---------------------------------------------------------------------------
# Composite config
# ---------------------------------------------------------------------------

@dataclass
class BioVerifyConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tfidf: TFIDFConfig = field(default_factory=TFIDFConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# ---------------------------------------------------------------------------
# Module-level default instance (importable directly)
# ---------------------------------------------------------------------------

CONFIG = BioVerifyConfig()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_hf_model_id(model_name: str) -> str:
    """Return the HuggingFace model ID for a given short model name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]


def get_experiment_path(model_name: str) -> str:
    """Return the experiment directory path for a given model."""
    return f"{CONFIG.experiment.experiment_dir}/{model_name}"


if __name__ == "__main__":
    import pprint

    print("=== BioVerify Config ===\n")

    print("MODEL_REGISTRY:")
    pprint.pprint(MODEL_REGISTRY)

    print("\nDEFAULT_MODEL:", DEFAULT_MODEL)
    print("NUM_CLASSES:", NUM_CLASSES)
    print("LABEL2ID:", LABEL2ID)
    print("ID2LABEL:", ID2LABEL)
    print("CLASS_NAMES:", CLASS_NAMES)

    print("\nModelConfig:")
    pprint.pprint(CONFIG.model.__dict__)

    print("\nTrainingConfig:")
    pprint.pprint(CONFIG.training.__dict__)

    print("\nTFIDFConfig:")
    pprint.pprint(CONFIG.tfidf.__dict__)

    print("\nUncertaintyConfig:")
    pprint.pprint(CONFIG.uncertainty.__dict__)

    print("\nSafety keyword categories and counts:")
    for category, keywords in HIGH_RISK_KEYWORDS.items():
        print(f"  {category}: {len(keywords)} keywords")

    print("\nDataConfig:")
    pprint.pprint(CONFIG.data.__dict__)

    print("\nExperimentConfig:")
    pprint.pprint(CONFIG.experiment.__dict__)
