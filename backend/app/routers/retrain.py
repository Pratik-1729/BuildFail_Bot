from fastapi import APIRouter, BackgroundTasks, HTTPException
import logging
import traceback
import os, json, datetime
from app.ml.train_model import train_model
from app.ml.inference import load_model_and_tokenizer
from app.config.settings import RETRAIN_HISTORY_FILE, MODEL_METRICS_FILE

router = APIRouter()
logger = logging.getLogger(__name__)

# Paths
RETRAIN_LOG = str(RETRAIN_HISTORY_FILE)
METRICS_PATH = str(MODEL_METRICS_FILE)


# ----------------------------------------------------------------------
# 🔁 Background Retraining Task
# ----------------------------------------------------------------------
def run_retrain_and_evaluate():
    """Retrain DistilBERT model using ingested logs + base dataset."""
    start_time = datetime.datetime.utcnow().isoformat()
    status = {
        "start_time": start_time,
        "status": "in_progress",
        "steps": [],
    }

    try:
        logger.info("🚀 Initiating retraining pipeline...")
        status["steps"].append("Retraining started")

        # Step 1 — Train model
        train_result = train_model()
        status["steps"].append("Training complete")
        logger.info("✅ Model retraining finished successfully.")

        # Step 2 — Extract metrics from training result
        # train_result is now a dict with accuracy, f1_score, precision, recall
        if isinstance(train_result, dict):
            accuracy = train_result.get("accuracy", 0.0)
            f1_score = train_result.get("f1_score", 0.0)
        else:
            # Backward compatibility: if train_result is just a float (accuracy)
            accuracy = float(train_result) if train_result else 0.0
            f1_score = 0.0

        # Load saved metrics for active_model info
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                saved_metrics = json.load(f)
                active_model = saved_metrics.get("active_model", "base")
        else:
            active_model = "base"

        # Structure metrics for dashboard
        metrics = {
            "accuracy": accuracy,
            "f1_score": f1_score,
            "active_model": active_model
        }

        logger.info(f"📊 Evaluation results — Accuracy: {accuracy:.4f}, F1: {f1_score:.4f}")
        status["steps"].append("Evaluation metrics loaded")

        # Step 3 — Reload best model for inference
        try:
            # Clear cache to force reload
            import app.ml.inference as inference_module
            inference_module._cached_model = None
            inference_module._cached_tokenizer = None
            
            load_model_and_tokenizer()
            status["steps"].append("Model reloaded for live inference")
            logger.info("♻️ Active model reloaded successfully after retraining.")
        except Exception as e:
            logger.warning(f"Could not reload model after retraining: {e}")
            status["steps"].append(f"Model reload warning: {str(e)}")
            # Don't fail the retraining if model reload fails

        # Step 4 — Success status
        status.update({
            "status": "success",
            "metrics": metrics,
            "train_result": train_result,
            "end_time": datetime.datetime.utcnow().isoformat(),
        })

    except Exception as e:
        logger.error("❌ Retraining pipeline failed:")
        logger.error(traceback.format_exc())
        status.update({
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "end_time": datetime.datetime.utcnow().isoformat(),
        })

    # Save retraining history
    history = []
    if os.path.exists(RETRAIN_LOG):
        with open(RETRAIN_LOG, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    history.append(status)
    with open(RETRAIN_LOG, "w") as f:
        json.dump(history, f, indent=4)

    logger.info("📝 Retraining history updated.")


# ----------------------------------------------------------------------
# ⚡ FastAPI Endpoint — Trigger Retraining
# ----------------------------------------------------------------------
@router.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Trigger retraining of the DistilBERT model using latest feedback and ingested logs.
    Runs asynchronously in background.
    """
    try:
        background_tasks.add_task(run_retrain_and_evaluate)
        logger.info("🧠 Retraining job scheduled in background.")
        return {
            "status": "accepted",
            "message": "Retraining initiated. It may take several minutes to complete.",
        }
    except Exception as e:
        logger.error(f"❌ Failed to start retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------------------------------------------------
# 🧾 Endpoint — Get Retraining History
# ----------------------------------------------------------------------
@router.get("/retrain/history")
def get_retrain_history():
    """Fetch retraining history log."""
    if not os.path.exists(RETRAIN_LOG):
        return {"history": []}

    with open(RETRAIN_LOG, "r") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = []

    return {"history": history}
