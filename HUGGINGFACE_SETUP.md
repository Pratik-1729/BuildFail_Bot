# Hugging Face Hub Setup Guide

Instead of storing large model files in GitHub, we upload them to **Hugging Face Hub** which is designed for ML models and datasets.

## 🚀 Quick Start

### 1. Get Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with **"Write"** permissions
3. Copy the token

### 2. Set Environment Variable

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN = "your_token_here"
```

**Windows (Command Prompt):**
```cmd
set HF_TOKEN=your_token_here
```

**Linux/Mac:**
```bash
export HF_TOKEN=your_token_here
```

**Or add to `.env` file:**
```
HF_TOKEN=your_token_here
```

### 3. Install Hugging Face Hub

```bash
pip install huggingface-hub
```

Or it's already in `requirements.txt`:
```bash
pip install -r requirements.txt
```

## 📤 Upload Models to Hugging Face

### Upload All Models

```bash
python scripts/upload_to_hf_hub.py --username YOUR_HF_USERNAME
```

This will:
- Upload `models/distilbert_best/` → `YOUR_USERNAME/buildfail-bot-distilbert_best`
- Upload `models/distilbert_trained/` → `YOUR_USERNAME/buildfail-bot-distilbert_trained`
- Upload `models/distilbert_retrained/` → `YOUR_USERNAME/buildfail-bot-distilbert_retrained`
- Upload `results/` → `YOUR_USERNAME/buildfail-bot-results` (as dataset)

### Upload Specific Model

```bash
python scripts/upload_to_hf_hub.py \
  --username YOUR_HF_USERNAME \
  --model-path models/distilbert_best \
  --repo-id YOUR_USERNAME/buildfail-bot-best
```

### Upload as Private Repos

```bash
python scripts/upload_to_hf_hub.py --username YOUR_HF_USERNAME --private
```

### Upload Only Models (Skip Results)

```bash
python scripts/upload_to_hf_hub.py --username YOUR_HF_USERNAME --models-only
```

### Upload Only Results

```bash
python scripts/upload_to_hf_hub.py --username YOUR_HF_USERNAME --results-only
```

## 📥 Download Models from Hugging Face

### Download All Models (Recommended)

```bash
python scripts/download_models.py --username YOUR_HF_USERNAME
```

This downloads all models to the correct local directories:
- `models/distilbert_best/`
- `models/distilbert_trained/`
- `models/distilbert_retrained/`

### Download Specific Model

```bash
# Download only the best model
python scripts/download_models.py --username YOUR_HF_USERNAME --model best

# Download only the trained model
python scripts/download_models.py --username YOUR_HF_USERNAME --model trained

# Download only the retrained model
python scripts/download_models.py --username YOUR_HF_USERNAME --model retrained
```

### Download with Token (for private repos)

```bash
python scripts/download_models.py --username YOUR_HF_USERNAME --token YOUR_TOKEN
```

Or set environment variable:
```bash
export HF_TOKEN=your_token_here
python scripts/download_models.py --username YOUR_HF_USERNAME
```

## 🔄 Manual Download Workflow (No Code Changes Needed)

**No need to modify inference.py!** Just download models manually before starting the app:

1. **Download models once:**
   ```bash
   python scripts/download_models.py --username YOUR_HF_USERNAME
   ```

2. **Start application normally:**
   ```bash
   python -m uvicorn backend.app.main:app --reload
   ```

The inference code will automatically find and use the downloaded models. No code changes needed!

## 📋 Example Workflow

### After Training a New Model

```bash
# 1. Train model (saves to models/)
python -m backend.app.ml.train_model

# 2. Upload to HF Hub
python scripts/upload_to_hf_hub.py --username YOUR_USERNAME

# 3. Commit code to GitHub (without models/)
git add .
git commit -m "Update model training"
git push origin main
```

### On a New Machine

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/BuildFail_Bot.git
cd BuildFail_Bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models from HF Hub (one command downloads all)
python scripts/download_models.py --username YOUR_HF_USERNAME

# 4. Run application
python -m uvicorn backend.app.main:app --reload
```

## 🔐 Security

- **Public repos**: Anyone can download your models
- **Private repos**: Only you (and collaborators) can access
- **Token**: Keep your HF_TOKEN secret (add to `.env`, never commit)

## 📊 Benefits

✅ **No GitHub size limits** - HF Hub is designed for large files  
✅ **Version control** - Models are versioned on HF Hub  
✅ **Easy sharing** - Share models with team or public  
✅ **Fast downloads** - CDN-backed downloads  
✅ **Free storage** - Generous free tier  

## 🆘 Troubleshooting

### "Token not found"
- Set `HF_TOKEN` environment variable
- Or use `--token` flag

### "Permission denied"
- Check token has "Write" permissions
- Regenerate token if needed

### "Repo already exists"
- Delete repo on HF Hub first: https://huggingface.co/YOUR_USERNAME/repo-name/settings
- Or use a different repo name

### "Upload timeout"
- Large models may take time
- Check your internet connection
- Try uploading one model at a time

