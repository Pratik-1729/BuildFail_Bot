# app/routers/dashboard.py
from fastapi import APIRouter, HTTPException
import pandas as pd
import os
import logging
from app.config.settings import INGESTED_LOGS_FILE

router = APIRouter()
logger = logging.getLogger(__name__)

DATA_FILE = str(INGESTED_LOGS_FILE)

@router.get("/logs")
def get_logs():
    """Return cleaned and structured logs for dashboard visualization."""
    logger.info(f"Dashboard requesting logs from: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        # Return empty list if file doesn't exist yet
        logger.warning(f"Log file not found: {DATA_FILE}")
        return {"logs": []}

    try:
        # Read and clean CSV
        df = pd.read_csv(DATA_FILE, on_bad_lines="skip", engine="python")
        
        # Return empty list if file is empty
        if df.empty:
            logger.warning(f"CSV file is empty: {DATA_FILE}")
            return {"logs": []}
        
        # Remove duplicate headers (rows where timestamp == 'timestamp')
        if "timestamp" in df.columns:
            df = df[df["timestamp"] != "timestamp"]
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        if df.empty:
            logger.warning(f"CSV file has no valid data after cleaning: {DATA_FILE}")
            return {"logs": []}
        
        logger.info(f"Reading {len(df)} logs from {DATA_FILE}")
        
        df.fillna("-", inplace=True)

        # Ensure expected schema
        expected_cols = [
            "timestamp", "repo", "run_id", "status",
            "label", "confidence", "failed_step",
            "suggestion", "clean_log_excerpt"
        ]

        for col in expected_cols:
            if col not in df.columns:
                df[col] = "-"

        # Sort by timestamp (newest first)
        if "timestamp" in df.columns:
            df.sort_values(by="timestamp", ascending=False, inplace=True)

        # Keep only last 300 records for performance
        df = df.tail(300)

        # Convert confidence to float safely
        if "confidence" in df.columns:
            df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0)

        return {"logs": df.to_dict(orient="records")}

    except Exception as e:
        # Log error but return empty list instead of raising exception
        logger.error(f"Error reading logs from {DATA_FILE}: {str(e)}", exc_info=True)
        return {"logs": []}
