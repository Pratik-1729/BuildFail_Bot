import os
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from app.ml.model_evaluation import evaluate_model
from app.ml.preprocess import clean_text
import logging, json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths (using centralized config)
# ---------------------------------------------------------------------
from app.config.settings import (
    PROJECT_ROOT,
    DATA_DIR,
    PROCESSED_DIR,
    FEEDBACK_DIR,
    MODELS_DIR,
    TOKENIZER_DIR,
    INGESTED_LOGS_FILE,
    MODEL_METRICS_FILE,
    DISTILBERT_BEST_DIR
)

ROOT_DIR = str(PROJECT_ROOT)
BASE_DATA_PATH = str(PROCESSED_DIR / "cleaned_logs.csv")
INGESTED_PATH = str(INGESTED_LOGS_FILE)
MODEL_DIR = str(MODELS_DIR)
BEST_MODEL_DIR = str(DISTILBERT_BEST_DIR)
METRICS_FILE = str(MODEL_METRICS_FILE)


# ------------------------------------------------------------------
# Utility: prepare data for retraining
# ------------------------------------------------------------------
def prepare_retrain_data():
    """Combine base data and new ingested logs into a unified training dataframe."""
    if not os.path.exists(BASE_DATA_PATH):
        raise FileNotFoundError(f"Base dataset not found at {BASE_DATA_PATH}")

    base_df = pd.read_csv(BASE_DATA_PATH)
    logger.info(f"📘 Loaded base dataset: {base_df.shape[0]} rows")

    # Load ingested logs (optional)
    if not os.path.exists(INGESTED_PATH):
        logger.warning(" No ingested logs found, using only base dataset.")
        return base_df

    try:
        ingested_df = pd.read_csv(INGESTED_PATH, on_bad_lines="skip")
        logger.info(f"Loaded ingested logs: {ingested_df.shape[0]} rows")
    except Exception as e:
        logger.error(f"Error reading ingested logs: {e}")
        return base_df

    # Extract meaningful columns for retraining
    # Handle both "log_excerpt" and "clean_log_excerpt" column names
    log_col = "clean_log_excerpt" if "clean_log_excerpt" in ingested_df.columns else "log_excerpt"
    if log_col not in ingested_df.columns:
        logger.warning(f"⚠️ No log excerpt column found. Available columns: {list(ingested_df.columns)}")
        return base_df
    
    required_cols = [log_col, "label"]
    for col in required_cols:
        if col not in ingested_df.columns:
            ingested_df[col] = ""

    # Clean + align schema
    ingested_df["clean_text"] = ingested_df[log_col].astype(str).apply(clean_text)
    ingested_df["error_segment"] = ingested_df[log_col].astype(str).str[:200]  # short context window
    ingested_df["label"] = ingested_df["label"].fillna("unknown").str.lower()

    # Keep only required training columns
    ingested_df = ingested_df[["clean_text", "error_segment", "label"]]

    # Combine datasets
    combined_df = pd.concat([base_df, ingested_df], ignore_index=True)
    combined_df.drop_duplicates(subset=["clean_text"], inplace=True)
    logger.info(f" Combined dataset size: {combined_df.shape[0]} rows")

    return combined_df


# ------------------------------------------------------------------
# Main retraining pipeline
# ------------------------------------------------------------------
def run_retraining():
    """Retrain model with new ingested logs and promote best model."""
    logger.info("Starting retraining pipeline...")
    df = prepare_retrain_data()

    # Encode labels
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    labels = list(le.classes_)
    logger.info(f"Labels for training: {labels}")

    # Train/test split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = DistilBertTokenizer.from_pretrained(TOKENIZER_DIR)

    def tokenize(batch):
        return tokenizer(
            batch["clean_text"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    import datasets
    train_ds = datasets.Dataset.from_pandas(train_df)
    val_ds = datasets.Dataset.from_pandas(val_df)
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label_enc"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label_enc"])

    # Load base model
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=len(labels))

    training_args = TrainingArguments(
        output_dir="models/output",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        learning_rate=5e-5,
        logging_dir="logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Training
    trainer.train()

    # Evaluate new model
    val_acc = evaluate_model(trainer, val_ds)
    logger.info(f" Validation Accuracy (Retrained): {val_acc:.4f}")

    # Compare with existing best metrics
    if not os.path.exists(METRICS_FILE):
        metrics = {"best_accuracy": 0.0, "active_model": "base"}
    else:
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)

    if val_acc > metrics.get("best_accuracy", 0.0):
        logger.info(f" New model is better! ({val_acc:.4f} > {metrics.get('best_accuracy', 0.0):.4f})")
        trainer.save_model(BEST_MODEL_DIR)
        metrics["best_accuracy"] = val_acc
        metrics["active_model"] = "retrained"
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f" Best model updated at: {BEST_MODEL_DIR}")
    else:
        logger.info("ℹRetained existing best model (no accuracy improvement).")

    logger.info(f" Metrics saved at {METRICS_FILE}")
    return val_acc
