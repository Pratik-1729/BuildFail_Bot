"""
Legacy config file - use app.config.settings instead.
Kept for backward compatibility.
"""
from app.config.settings import (
    INGESTED_LOGS_FILE as DATA_FILE,
    FEEDBACK_DIR as DATA_DIR,
    PROJECT_ROOT as BASE_DIR
)