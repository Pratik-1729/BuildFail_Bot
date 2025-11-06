from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import datetime
import logging
from app.config.settings import FEEDBACK_FILE, FEEDBACK_DIR

# Initialize router and logger
router = APIRouter()
logger = logging.getLogger(__name__)

# Feedback storage path
DATA_DIR = str(FEEDBACK_DIR)


# --- Schema ---
class FeedbackInput(BaseModel):
    label: str
    confidence: float
    log_excerpt: str
    rating: int  # Rating out of 5


# --- POST endpoint: store feedback ---
@router.post("", include_in_schema=False)  # Handle without trailing slash
@router.post("/")
def submit_feedback(data: FeedbackInput):
    """
    Store user feedback from the Chrome extension or dashboard.
    """
    try:
        record = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "label": data.label,
            "confidence": data.confidence,
            "log_excerpt": data.log_excerpt[:250],
            "rating": data.rating,
        }

        df = pd.DataFrame([record])
        header = not os.path.exists(str(FEEDBACK_FILE))
        df.to_csv(str(FEEDBACK_FILE), mode="a", index=False, header=header)

        logger.info(f"Feedback recorded: {record}")
        return {"status": "success", "message": "Feedback recorded successfully."}

    except Exception as e:
        logger.exception("Error saving feedback")
        raise HTTPException(status_code=500, detail=str(e))


# --- GET endpoint: retrieve all feedback ---
@router.get("", include_in_schema=False)  # Handle without trailing slash
@router.get("/")
def get_feedback():
    """
    Fetch all stored feedback records for dashboard visualization.
    """
    try:
        feedback_file_path = str(FEEDBACK_FILE)
        logger.info(f"Dashboard requesting feedback from: {feedback_file_path}")
        
        if not os.path.exists(feedback_file_path):
            logger.warning(f"Feedback file not found: {feedback_file_path}")
            return {"feedback": []}

        df = pd.read_csv(feedback_file_path, on_bad_lines="skip", engine="python")
        logger.info(f"Reading {len(df)} feedback entries from {feedback_file_path}")
        
        if df.empty:
            return {"feedback": []}
        
        # Remove duplicate headers (rows where timestamp == 'timestamp')
        if "timestamp" in df.columns:
            df = df[df["timestamp"] != "timestamp"]
        
        # Ensure rating column exists and is numeric
        if "rating" not in df.columns:
            logger.warning("Rating column not found, adding default rating")
            df["rating"] = 0
        
        # Convert rating to numeric, handling any corrupted values
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        
        # Remove rows with invalid ratings
        df = df[df["rating"].notna()]
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        if df.empty:
            logger.warning("No valid feedback entries after cleaning")
            return {"feedback": []}
        
        feedback = df.to_dict(orient="records")

        # Add aggregate metrics for dashboard stats
        avg_rating = round(df["rating"].mean(), 2) if not df.empty and df["rating"].notna().any() else 0
        total_feedback = len(df)

        return {
            "feedback": feedback,
            "summary": {
                "average_rating": avg_rating,
                "total_feedback": total_feedback
            },
        }

    except Exception as e:
        logger.exception("Error reading feedback file")
        raise HTTPException(status_code=500, detail=str(e))
