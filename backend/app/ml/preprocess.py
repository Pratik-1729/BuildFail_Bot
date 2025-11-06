"""
Handles preprocessing for both initial and retraining phases.
Merges text fields, tokenizes, and prepares datasets for DistilBERT.
Now fully serialization-safe (saves tensors, not Dataset objects).
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset, TensorDataset

# ---------------------------------------------------------------------
# Configuration (using centralized settings)
# ---------------------------------------------------------------------
from app.config.settings import (
    PROJECT_ROOT,
    DATA_DIR,
    PROCESSED_DIR,
    TOKENIZER_DIR,
    PROCESSED_DATA_FILE
)

ROOT_DIR = str(PROJECT_ROOT)
FINAL_DATA_FILE = str(PROCESSED_DIR / "cleaned_logs.csv")
TOKENIZER_PATH = str(TOKENIZER_DIR)
PROCESSED_OUTPUT = str(PROCESSED_DATA_FILE)

# ---------------------------------------------------------------------
# Dataset class for runtime use
# ---------------------------------------------------------------------
class LogDataset(Dataset):
    """PyTorch dataset for tokenized logs."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Use clone().detach() to avoid warnings when copying tensors
        item = {key: val[idx].clone().detach() if isinstance(val[idx], torch.Tensor) else torch.tensor(val[idx]) 
                for key, val in self.encodings.items()}
        if isinstance(self.labels[idx], torch.Tensor):
            item["labels"] = self.labels[idx].clone().detach()
        else:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# ---------------------------------------------------------------------
# Text cleaning utility
# ---------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean log text by removing noise and normalizing."""
    if not text:
        return ""
    import re
    # Remove ANSI color codes
    text = re.sub(r"\x1B\[[0-9;]*[A-Za-z]", "", text)
    # Remove null bytes and carriage returns
    text = text.replace("\x00", "").replace("\r", "")
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Remove timestamps
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", "", text)
    # Remove GitHub Actions markers
    text = re.sub(r"##\[[^\]]+\]", "", text)
    text = re.sub(r"::group::|::endgroup::", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------
# Load dataset (real + synthetic merged)
# ---------------------------------------------------------------------
def load_dataset():
    if not os.path.exists(FINAL_DATA_FILE):
        raise FileNotFoundError(f"❌ {FINAL_DATA_FILE} not found. Run dataset merge first.")

    df = pd.read_csv(FINAL_DATA_FILE)
    print(f"✅ Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")

    # Handle different column names
    if "clean_text" in df.columns and "error_segment" in df.columns:
        df["text"] = (
            df["clean_text"].astype(str).fillna("") + " " +
            df["error_segment"].astype(str).fillna("")
        ).str.strip()
    elif "text" not in df.columns:
        # If no text column, create from available columns
        text_cols = [col for col in df.columns if col in ["text", "log", "content", "message"]]
        if text_cols:
            df["text"] = df[text_cols[0]].astype(str).fillna("")
        else:
            raise ValueError("No text column found in dataset")

    df = df[df["label"].notna()]
    return df

# ---------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------
def encode_labels(labels):
    label2id = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
    id2label = {v: k for k, v in label2id.items()}
    encoded = labels.map(label2id).values
    return encoded, label2id, id2label

# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------
def tokenize_dataset(df, tokenizer):
    encodings = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding=True,
        max_length=128
    )
    return encodings

# ---------------------------------------------------------------------
# Main preprocessing entry point (tensor-based saving)
# ---------------------------------------------------------------------
def preprocess_and_save():
    df = load_dataset()
    labels, label2id, id2label = encode_labels(df["label"])

    # Load or initialize tokenizer
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
        print("✅ Loaded existing tokenizer.")
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        tokenizer.save_pretrained(TOKENIZER_PATH)
        print("📦 Saved new tokenizer.")

    encodings = tokenize_dataset(df, tokenizer)

    # Split train/validation indices
    train_indices, val_indices = train_test_split(
        range(len(df)), test_size=0.2, random_state=42
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

    # ✅ Save plain tensors & metadata — no class references
    torch.save({
        "train_data": train_data,
        "val_data": val_data,
        "label2id": label2id,
        "id2label": id2label
    }, PROCESSED_OUTPUT)

    print(f"✅ Preprocessing complete. Saved tensor data to {PROCESSED_OUTPUT}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    preprocess_and_save()
