from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.routers import logs, feedback, dashboard
from app.ml.inference import predict_log_status
from app.routers import retrain
from app.config.settings import CORS_ORIGINS

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="BuildFail Bot API",
    description="API for predicting build or deployment log statuses using fine-tuned Transformer model.",
    version="1.0.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(logs.router, prefix="/api/logs", tags=["Logs"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(feedback.router, prefix="/api/feedback", tags=["Feedback"])
app.include_router(retrain.router, prefix="/api", tags=["Retrain"])

# Polling service (optional)
try:
    from app.routers import polling
    app.include_router(polling.router, prefix="/api/polling", tags=["Polling"])
except ImportError:
    pass  # Polling service is optional
# app.include_router(feedback.router, prefix="/api", tags=["Feedback"])
# --- Input schema ---
class LogInput(BaseModel):
    text: str

# --- Root endpoint ---
@app.get("/")
def read_root():
    return {"message": " BuildFail Bot API is running successfully!"}

# --- Prediction endpoint ---
@app.post("/predict")
def predict_log(input_data: LogInput):
    """
    Receives log text and returns predicted build status + confidence.
    """
    result = predict_log_status(input_data.text)
    return {"prediction": result}

# --- Health check endpoint ---
@app.get("/health")
def health_check():
    return {"status": "healthy"}
