"""
Centralized configuration for BuildFail Bot backend.
All paths and settings are defined here for consistency.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------
# Base Directory Resolution
# ---------------------------------------------------------------------
# Go from app/config/settings.py → app → backend → BuildFail_Bot
BACKEND_DIR = Path(__file__).parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

# ---------------------------------------------------------------------
# Data Directories
# ---------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
FEEDBACK_DIR = DATA_DIR / "feedback"
MODEL_DIR = DATA_DIR / "model"
TOKENIZER_DIR = MODEL_DIR / "tokenizer"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------
# Data Files
# ---------------------------------------------------------------------
INGESTED_LOGS_FILE = FEEDBACK_DIR / "ingested_logs.csv"
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.csv"
MODEL_METRICS_FILE = DATA_DIR / "model_metrics.json"
RETRAIN_HISTORY_FILE = DATA_DIR / "retrain_history.json"
PROCESSED_DATA_FILE = PROCESSED_DIR / "processed_data.pt"

# ---------------------------------------------------------------------
# Model Directories
# ---------------------------------------------------------------------
DISTILBERT_BEST_DIR = MODELS_DIR / "distilbert_best"
DISTILBERT_TRAINED_DIR = MODELS_DIR / "distilbert_trained"
DISTILBERT_RETRAINED_DIR = MODELS_DIR / "distilbert_retrained"

# ---------------------------------------------------------------------
# Create directories if they don't exist
# ---------------------------------------------------------------------
for dir_path in [FEEDBACK_DIR, MODEL_DIR, TOKENIZER_DIR, PROCESSED_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Environment Variables
# ---------------------------------------------------------------------
# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://localhost:{API_PORT}")

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS]

# GitHub Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Model Configuration
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "distilbert-base-uncased")
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------
# Export all paths as strings for compatibility
# ---------------------------------------------------------------------
def get_paths():
    """Return all paths as dictionary with string values."""
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "BACKEND_DIR": str(BACKEND_DIR),
        "DATA_DIR": str(DATA_DIR),
        "FEEDBACK_DIR": str(FEEDBACK_DIR),
        "MODEL_DIR": str(MODEL_DIR),
        "TOKENIZER_DIR": str(TOKENIZER_DIR),
        "MODELS_DIR": str(MODELS_DIR),
        "INGESTED_LOGS_FILE": str(INGESTED_LOGS_FILE),
        "FEEDBACK_FILE": str(FEEDBACK_FILE),
        "MODEL_METRICS_FILE": str(MODEL_METRICS_FILE),
        "RETRAIN_HISTORY_FILE": str(RETRAIN_HISTORY_FILE),
        "PROCESSED_DATA_FILE": str(PROCESSED_DATA_FILE),
        "DISTILBERT_BEST_DIR": str(DISTILBERT_BEST_DIR),
        "DISTILBERT_TRAINED_DIR": str(DISTILBERT_TRAINED_DIR),
        "DISTILBERT_RETRAINED_DIR": str(DISTILBERT_RETRAINED_DIR),
    }

