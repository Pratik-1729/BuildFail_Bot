"""
Retrains (fine-tunes) DistilBERT on the updated dataset (real + synthetic logs).
Tracks accuracy metrics and automatically updates the best model record.
"""

import os
import json
import torch
import datetime
from torch.utils.data import DataLoader
from app.ml.preprocess import LogDataset, encode_labels, tokenize_dataset
from tqdm import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_scheduler
)
from app.ml.inference import get_active_model_path  # ✅ ensures continuity
from app.config.settings import INGESTED_LOGS_FILE
from sklearn.model_selection import train_test_split
import shutil, tempfile
import pandas as pd
# ---------------------------------------------------------------------
# Paths (using centralized config)
# ---------------------------------------------------------------------
from app.config.settings import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    DISTILBERT_RETRAINED_DIR,
    DISTILBERT_BEST_DIR,
    PROCESSED_DATA_FILE,
    MODEL_METRICS_FILE,
    TOKENIZER_DIR
)

ROOT_DIR = str(PROJECT_ROOT)
RETRAINED_DIR = str(DISTILBERT_RETRAINED_DIR)
BEST_MODEL_DIR = str(DISTILBERT_BEST_DIR)
PROCESSED_FILE = str(PROCESSED_DATA_FILE)
METRICS_FILE = str(MODEL_METRICS_FILE)

# ---------------------------------------------------------------------
# Training parameters
# ---------------------------------------------------------------------
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5


# ---------------------------------------------------------------------
# Load and combine datasets (base + new ingested logs)
# ---------------------------------------------------------------------
def prepare_combined_dataset():
    """Combine base dataset with new ingested logs for retraining."""
    import pandas as pd
    from app.config.settings import INGESTED_LOGS_FILE
    from app.ml.preprocess import load_dataset, encode_labels, tokenize_dataset
    from sklearn.model_selection import train_test_split
    
    # Load base dataset
    base_df = load_dataset()
    print(f"📘 Loaded base dataset: {len(base_df)} rows")
    
    # Load new ingested logs
    ingested_file = str(INGESTED_LOGS_FILE)
    new_logs_df = None
    if os.path.exists(ingested_file):
        try:
            ingested_df = pd.read_csv(ingested_file, on_bad_lines="skip", engine="python")
            if not ingested_df.empty and "clean_log_excerpt" in ingested_df.columns:
                # Map ingested logs to training format
                new_logs_df = pd.DataFrame({
                    "clean_text": ingested_df["clean_log_excerpt"].astype(str).fillna(""),
                    "error_segment": ingested_df["clean_log_excerpt"].astype(str).str[:200].fillna(""),
                    "label": ingested_df["label"].astype(str).fillna("unknown").str.lower()
                })
                # Remove empty logs
                new_logs_df = new_logs_df[new_logs_df["clean_text"].str.strip() != ""]
                print(f"📥 Loaded {len(new_logs_df)} new ingested logs")
        except Exception as e:
            print(f"⚠️ Could not load ingested logs: {e}")
    
    # Combine datasets
    if new_logs_df is not None and not new_logs_df.empty:
        # Create text column for new logs (same format as base)
        new_logs_df["text"] = (
            new_logs_df["clean_text"].astype(str) + " " + 
            new_logs_df["error_segment"].astype(str)
        ).str.strip()
        
        # Map to base format
        combined_df = pd.concat([base_df, new_logs_df[["text", "label"]]], ignore_index=True)
        # Remove duplicates
        combined_df = combined_df.drop_duplicates(subset=["text"], keep="last")
        print(f"✅ Combined dataset: {len(combined_df)} rows ({len(base_df)} base + {len(new_logs_df)} new)")
    else:
        combined_df = base_df
        print(f"✅ Using base dataset only: {len(combined_df)} rows")
    
    return combined_df


def load_preprocessed():
    """Load or create preprocessed data with new logs."""
    # Check if we need to reprocess (if new logs exist)
    ingested_file = str(INGESTED_LOGS_FILE)
    needs_reprocess = os.path.exists(ingested_file) and os.path.getsize(ingested_file) > 0
    
    if needs_reprocess or not os.path.exists(PROCESSED_FILE):
        print("🔄 Preparing combined dataset with new logs...")
        combined_df = prepare_combined_dataset()
        
        # Encode labels
        labels, label2id, id2label = encode_labels(combined_df["label"])
        
        # Load or create tokenizer
        from transformers import DistilBertTokenizerFast
        if os.path.exists(str(TOKENIZER_DIR)) and os.listdir(str(TOKENIZER_DIR)):
            tokenizer = DistilBertTokenizerFast.from_pretrained(str(TOKENIZER_DIR))
        else:
            tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
            tokenizer.save_pretrained(str(TOKENIZER_DIR))
        
        # Tokenize
        encodings = tokenize_dataset(combined_df, tokenizer)
        
        # Split train/val
        train_indices, val_indices = train_test_split(
            range(len(combined_df)), test_size=0.2, random_state=42
        )
        
        # Build tensors
        input_ids = torch.tensor(encodings["input_ids"])
        attention_masks = torch.tensor(encodings["attention_mask"])
        labels_tensor = torch.tensor(labels)
        
        train_data = {
            "input_ids": input_ids[train_indices],
            "attention_mask": attention_masks[train_indices],
            "labels": labels_tensor[train_indices],
        }
        val_data = {
            "input_ids": input_ids[val_indices],
            "attention_mask": attention_masks[val_indices],
            "labels": labels_tensor[val_indices],
        }
        
        # Save processed data
        os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
        torch.save({
            "train_data": train_data,
            "val_data": val_data,
            "label2id": label2id,
            "id2label": id2label
        }, PROCESSED_FILE)
        print(f"✅ Saved preprocessed data to {PROCESSED_FILE}")
    
    # Load preprocessed data
    data = torch.load(PROCESSED_FILE, map_location="cpu")
    train_data, val_data = data["train_data"], data["val_data"]

    train_dataset = LogDataset(
        {k: v for k, v in train_data.items() if k != "labels"},
        train_data["labels"]
    )
    val_dataset = LogDataset(
        {k: v for k, v in val_data.items() if k != "labels"},
        val_data["labels"]
    )

    label2id = data["label2id"]
    id2label = data["id2label"]

    print(f"✅ Loaded preprocessed tensors: {len(train_dataset)} train, {len(val_dataset)} val")
    return train_dataset, val_dataset, label2id, id2label


# ---------------------------------------------------------------------
# Metrics handling
# ---------------------------------------------------------------------
def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return {"base_accuracy": 0.0, "retrained_accuracy": 0.0, "active_model": "base"}


def save_metrics(metrics):
    # Directories are created by settings.py, no need to create here
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=4)


# ---------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------
def train_model():
    train_dataset, val_dataset, label2id, id2label = load_preprocessed()

    # ✅ Load tokenizer from consistent directory
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(TOKENIZER_DIR))

    # ✅ Load the currently active model
    model_path = get_active_model_path()
    print(f"🔄 Loading from active model: {model_path}")

    # Handle None model_path (no trained model exists)
    if model_path and os.path.exists(model_path) and os.path.isdir(model_path):
        try:
            model = DistilBertForSequenceClassification.from_pretrained(
                model_path, num_labels=len(label2id),
                id2label=id2label, label2id=label2id
            )
            print(f"✅ Loaded existing model from {model_path}")
        except Exception as e:
            print(f"⚠️ Error loading model from {model_path}: {e}")
            print("🆕 Falling back to base DistilBERT.")
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=len(label2id),
                id2label=id2label, label2id=label2id
            )
    else:
        print("🆕 No existing fine-tuned model found — using base DistilBERT.")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(label2id),
            id2label=id2label, label2id=label2id
        )

    # ✅ Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Use PyTorch's AdamW to avoid deprecation warning
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # ---------------- Training loop ----------------
    model.train()
    progress_bar = tqdm(range(num_training_steps))
    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch + 1}/{EPOCHS}")
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        avg_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch + 1} complete. Avg Loss: {avg_loss:.4f}")

    # ---------------- Validation ---------------- 
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    # Calculate accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate F1 score
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    
    print(f"\n📊 Validation Metrics (Retrained):")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1_score:.4f}")

    # ---------------- Metrics & Model Management ---------------- 
    metrics = load_metrics()
    prev_best = metrics.get("base_accuracy", 0.0)
    
    # If this is the first training (prev_best is 0 or unrealistic), always save
    is_first_training = prev_best == 0.0 or prev_best >= 1.0

    metrics["retrained_accuracy"] = accuracy
    metrics["last_retrain_time"] = datetime.datetime.utcnow().isoformat()

    # Always save the model to DISTILBERT_TRAINED_DIR for inference
    os.makedirs(RETRAINED_DIR, exist_ok=True)
    model.save_pretrained(RETRAINED_DIR, safe_serialization=False)
    tokenizer.save_pretrained(RETRAINED_DIR)
    print(f"✅ Model saved to {RETRAINED_DIR}")

    # Also save to DISTILBERT_TRAINED_DIR (used by inference)
    from app.config.settings import DISTILBERT_TRAINED_DIR
    TRAINED_DIR = str(DISTILBERT_TRAINED_DIR)
    os.makedirs(TRAINED_DIR, exist_ok=True)
    model.save_pretrained(TRAINED_DIR, safe_serialization=False)
    tokenizer.save_pretrained(TRAINED_DIR)
    print(f"✅ Model saved to {TRAINED_DIR} (for inference)")

    if is_first_training or accuracy > prev_best:
        if is_first_training:
            print(f"🆕 First training - saving model as baseline.")
        else:
            print(f"🏆 New model is better! ({accuracy:.4f} > {prev_best:.4f})")
        
        metrics["base_accuracy"] = accuracy
        metrics["active_model"] = "retrained"

        # Save best model atomically
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir, safe_serialization=False)
            tokenizer.save_pretrained(tmpdir)
            # Replace directory atomically (must be inside with block)
            if os.path.exists(BEST_MODEL_DIR):
                shutil.rmtree(BEST_MODEL_DIR)
            shutil.copytree(tmpdir, BEST_MODEL_DIR)

        print(f"✅ Best model updated at: {BEST_MODEL_DIR}")
    else:
        print(f"⚖️ Retained old best model. ({accuracy:.4f} <= {prev_best:.4f})")
        metrics["active_model"] = "retrained"  # Still use the newly trained model

    # Save metrics
    save_metrics(metrics)
    print(f"📈 Metrics saved at {METRICS_FILE}")

    # Return both accuracy and F1 score as a dict
    return {
        "accuracy": accuracy,
        "f1_score": float(f1_score),
        "precision": float(precision),
        "recall": float(recall)
    }


# ---------------------------------------------------------------------
# Run Directly
# ---------------------------------------------------------------------
if __name__ == "__main__":
    acc = train_model()
    print(f"\n🎯 Final retraining accuracy: {acc['accuracy']:.4f}")
