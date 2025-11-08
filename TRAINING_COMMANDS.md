# Training Commands Guide - BuildFail Bot

Complete guide for training the initial model and retraining.

## 🎯 Training Options

You have **3 ways** to train models:

1. **Initial Training** - Train from scratch using base dataset
2. **Retraining** - Update model with new logs (via API or script)
3. **Advanced Training (V2)** - Robust model with feedback weighting

---

## 📚 Option 1: Initial Training (From Scratch)

### Step 1: Preprocess Data

First, prepare your dataset:

```bash
# From project root
python -m backend.app.ml.preprocess
```

**What it does:**
- Loads `data/processed/cleaned_logs.csv`
- Tokenizes the data
- Splits into train/validation sets
- Saves preprocessed tensors to `data/processed/processed_data.pt`

**Requirements:**
- `data/processed/cleaned_logs.csv` must exist
- Should have columns: `clean_text`, `error_segment`, `label`

### Step 2: Train Initial Model

```bash
# From project root
python -m backend.app.ml.train_model
```

**What it does:**
- Loads preprocessed data
- Starts from base DistilBERT (no existing model needed)
- Trains for 2 epochs
- Saves model to:
  - `models/distilbert_trained/` (for inference)
  - `models/distilbert_retrained/` (backup)
  - `models/distilbert_best/` (if accuracy improves)
- Calculates accuracy, F1 score, precision, recall
- Saves metrics to `data/model_metrics.json`

**Output:**
```
🔄 Loading from active model: None
🆕 No existing fine-tuned model found — using base DistilBERT.
🚀 Epoch 1/2
✅ Epoch 1 complete. Avg Loss: 0.5089
🚀 Epoch 2/2
✅ Epoch 2 complete. Avg Loss: 0.0832
📊 Validation Metrics (Retrained):
   Accuracy: 0.9879
   Precision: 0.9850
   Recall: 0.9879
   F1 Score: 0.9864
✅ Model saved to models/distilbert_trained
✅ Model saved to models/distilbert_retrained
🏆 New model is better! (0.9879 > 0.0000)
✅ Best model updated at: models/distilbert_best
📈 Metrics saved at data/model_metrics.json
```

### Complete Initial Training Workflow

```bash
# 1. Ensure you have cleaned_logs.csv in data/processed/
#    (This should be your initial dataset)

# 2. Preprocess data
python -m backend.app.ml.preprocess

# 3. Train initial model
python -m backend.app.ml.train_model

# 4. (Optional) Upload to Hugging Face Hub
python scripts/upload_to_hf_hub.py --username YOUR_HF_USERNAME
```

---

## 🔄 Option 2: Retraining (Update Existing Model)

### Via API (Recommended)

```bash
# Start backend server first
python -m uvicorn backend.app.main:app --reload

# Then trigger retraining via API
curl -X POST http://localhost:8000/api/retrain
```

**Or use Dashboard:**
1. Open http://localhost:3000
2. Go to "Retrain" tab
3. Click "Start Retraining"

**What it does:**
- Combines base dataset + new ingested logs
- Retrains model (continues from existing model if available)
- Updates best model if accuracy improves
- Saves metrics and history

### Via Direct Script

```bash
# Retrain using the training script directly
python -m backend.app.ml.train_model
```

This will:
- Automatically combine base dataset with new ingested logs
- Use existing model as starting point (if available)
- Train and save updated model

---

## 🚀 Option 3: Advanced Training (V2 Model)

For more robust training with feedback weighting:

```bash
# Navigate to V2 directory
cd ml_models_v2

# Install V2 dependencies (if not already)
pip install -r requirements.txt

# Train with feedback weighting
python trainer.py --model distilbert_v2 --epochs 5 --use-feedback

# Or train without feedback (faster)
python trainer.py --model distilbert_v2 --epochs 3 --no-use-feedback
```

**Features:**
- Feedback-weighted learning
- Longer context (512 tokens)
- Better data combination
- Separate from production model

See `ml_models_v2/ROBUST_MODEL_GUIDE.md` for details.

---

## 📋 Training Parameters

### Current Training Settings (train_model.py)

```python
EPOCHS = 2
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
MAX_SEQUENCE_LENGTH = 256
```

### To Modify Training Parameters

Edit `backend/app/ml/train_model.py`:

```python
# Around line 47-49
EPOCHS = 3  # Increase for more training
BATCH_SIZE = 16  # Increase if you have more GPU memory
LEARNING_RATE = 2e-5  # Adjust learning rate
```

---

## 🔍 Check Training Status

### View Metrics

```bash
# Check model metrics
cat data/model_metrics.json
```

### View Retraining History

```bash
# Check retraining history
cat data/retrain_history.json
```

### Via Dashboard

- Go to http://localhost:3000
- Click "Retrain" tab
- View "Retraining History" table

---

## 🐛 Troubleshooting

### "FileNotFoundError: cleaned_logs.csv not found"

**Solution:**
1. Ensure `data/processed/cleaned_logs.csv` exists
2. If not, create it with columns: `clean_text`, `error_segment`, `label`
3. Or use the base dataset you mentioned having

### "No preprocessed data found"

**Solution:**
Run preprocessing first:
```bash
python -m backend.app.ml.preprocess
```

### "Model training takes too long"

**Solutions:**
- Reduce `EPOCHS` in `train_model.py`
- Increase `BATCH_SIZE` if you have GPU
- Use CPU-optimized PyTorch build

### "Out of memory during training"

**Solutions:**
- Reduce `BATCH_SIZE` (e.g., from 8 to 4)
- Reduce `MAX_SEQUENCE_LENGTH` (e.g., from 256 to 128)
- Use smaller model (already using DistilBERT which is small)

---

## 📊 Training Output Locations

After training, models are saved to:

- **Main model**: `models/distilbert_trained/`
- **Retrained backup**: `models/distilbert_retrained/`
- **Best model**: `models/distilbert_best/`
- **Metrics**: `data/model_metrics.json`
- **History**: `data/retrain_history.json`
- **Preprocessed data**: `data/processed/processed_data.pt`

---

## 🎯 Quick Reference

### Initial Training (First Time)
```bash
python -m backend.app.ml.preprocess
python -m backend.app.ml.train_model
```

### Retraining (After New Logs)
```bash
# Via API (recommended)
curl -X POST http://localhost:8000/api/retrain

# Or directly
python -m backend.app.ml.train_model
```

### Advanced Training (V2)
```bash
cd ml_models_v2
python trainer.py --model distilbert_v2 --epochs 5
```

---

## ✅ Verification

After training, verify the model works:

```bash
# Start backend
python -m uvicorn backend.app.main:app --reload

# Test prediction
curl -X POST http://localhost:8000/api/logs/manual \
  -H "Content-Type: application/json" \
  -d '{"log_text": "Error: ModuleNotFoundError: No module named numpy"}'
```

You should get a prediction with label, confidence, and suggestion.

