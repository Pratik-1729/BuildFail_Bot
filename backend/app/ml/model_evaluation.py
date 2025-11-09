import numpy as np
import torch
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Evaluate model on validation dataset
# ------------------------------------------------------------------
def evaluate_model(trainer, val_dataset):
    """
    Evaluate the retrained model using HuggingFace Trainer.
    Returns validation accuracy as float (0–1).
    """
    try:
        logger.info("Evaluating model performance on validation set...")
        preds_output = trainer.predict(val_dataset)

        preds = np.argmax(preds_output.predictions, axis=1)
        labels = preds_output.label_ids

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )

        logger.info(f"Metrics → Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
        return round(acc, 4)

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return 0.0


# ------------------------------------------------------------------
# (Optional) Evaluate individual model manually
# ------------------------------------------------------------------
def evaluate_standalone(model, tokenizer, dataset, device="cpu"):
    """
    Standalone evaluation for manual testing without Trainer.
    Useful when debugging inference performance or pipeline changes.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataset:
            inputs = tokenizer(
                batch["clean_text"],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            ).to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.append(batch["label_enc"])

    acc = accuracy_score(all_labels, all_preds)
    logger.info(f"Standalone evaluation accuracy: {acc:.4f}")
    return acc
