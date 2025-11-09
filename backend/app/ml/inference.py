import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import json
import os
import logging
import re
from app.config.settings import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    TOKENIZER_DIR,
    MODEL_METRICS_FILE,
    DISTILBERT_BEST_DIR,
    DISTILBERT_TRAINED_DIR
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Directory Paths (using centralized config)
# ------------------------------------------------------------------
ROOT_DIR = str(PROJECT_ROOT)
METRICS_PATH = str(MODEL_METRICS_FILE)

# Global cache to avoid reloading model repeatedly
_cached_model = None
_cached_tokenizer = None


# ------------------------------------------------------------------
# Utility: Select active/best model
# ------------------------------------------------------------------
def get_active_model_path():
    """Select the currently active model (retrained if better)."""
    if not os.path.exists(METRICS_PATH):
        logger.warning("Metrics file not found. Defaulting to best model.")
        # Check if best model exists, otherwise use trained
        if os.path.exists(str(DISTILBERT_BEST_DIR)) and os.listdir(str(DISTILBERT_BEST_DIR)):
            return str(DISTILBERT_BEST_DIR)
        elif os.path.exists(str(DISTILBERT_TRAINED_DIR)) and os.listdir(str(DISTILBERT_TRAINED_DIR)):
            return str(DISTILBERT_TRAINED_DIR)
        else:
            return None  # Will trigger base model fallback

    try:
        with open(METRICS_PATH, "r") as f:
            data = json.load(f)

        active = data.get("active_model", "base")
        if active == "retrained":
            model_path = str(DISTILBERT_BEST_DIR)
        else:
            model_path = str(DISTILBERT_TRAINED_DIR)
        
        # Verify path exists and has files
        if model_path and os.path.exists(model_path) and os.listdir(model_path):
            return model_path
        else:
            logger.warning(f"Model path does not exist or is empty: {model_path}")
            return None
    except Exception as e:
        logger.error(f"Error reading metrics file: {e}")
        return None


# ------------------------------------------------------------------
# Model + Tokenizer Loader
# ------------------------------------------------------------------
def load_model_and_tokenizer():
    """Load the tokenizer and model dynamically (cached for performance)."""
    global _cached_model, _cached_tokenizer

    if _cached_model is not None and _cached_tokenizer is not None:
        return _cached_model, _cached_tokenizer

    model_path = get_active_model_path()
    
    # If no model path, use base model
    if model_path is None:
        logger.info("No trained model found. Using base DistilBERT model.")
        try:
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=4  # success, failed, skipped, unknown
            )
            _cached_model, _cached_tokenizer = model, tokenizer
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise FileNotFoundError(f"Could not load base model: {e}")
    
    logger.info(f"Loading active model from {model_path}")

    # Check if model path exists
    if not os.path.exists(model_path) or not os.path.isdir(model_path):
        logger.warning(f"Model path does not exist: {model_path}. Using base model.")
        # Use base model if trained model doesn't exist
        try:
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=4  # success, failed, skipped, unknown
            )
            _cached_model, _cached_tokenizer = model, tokenizer
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise FileNotFoundError(f"Could not load base model: {e}")

    try:
        # Try to load tokenizer from tokenizer directory first
        if os.path.exists(str(TOKENIZER_DIR)) and os.listdir(str(TOKENIZER_DIR)):
            tokenizer = DistilBertTokenizer.from_pretrained(str(TOKENIZER_DIR))
        else:
            # Fallback to model directory or base model
            if os.path.exists(model_path):
                tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            else:
                tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception as e:
        logger.warning(f"Tokenizer not found at {TOKENIZER_DIR}, using base: {e}")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    try:
        # Load model from local path
        if os.path.exists(model_path) and os.path.isdir(model_path):
            model = DistilBertForSequenceClassification.from_pretrained(model_path)
        else:
            # Fallback to base model
            logger.warning(f"Model path invalid, using base model: {model_path}")
            model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=4
            )
    except Exception as e:
        logger.error(f"Model not found or corrupted: {e}")
        # Fallback to base model
        logger.info("Falling back to base DistilBERT model")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=4
        )

    _cached_model, _cached_tokenizer = model, tokenizer
    return model, tokenizer


# ------------------------------------------------------------------
# Smart Suggestion Engine v3 - Context-Aware with Error Extraction
# ------------------------------------------------------------------
def extract_error_details(log_text: str) -> dict:
    """Extract specific error details from log text."""
    text = log_text
    text_lower = log_text.lower()
    details = {
        "error_type": None,
        "error_message": None,
        "file_path": None,
        "line_number": None,
        "module_name": None,
        "command": None,
        "traceback": None
    }
    
    # Extract traceback
    traceback_match = re.search(r"Traceback.*?(?=\n\n|\n[A-Z]|\Z)", text, re.DOTALL)
    if traceback_match:
        details["traceback"] = traceback_match.group(0)[:500]  # Limit length
    
    # Extract file path and line number
    file_line_match = re.search(r'File\s+["\']([^"\']+)["\'],\s*line\s+(\d+)', text)
    if file_line_match:
        details["file_path"] = file_line_match.group(1)
        details["line_number"] = file_line_match.group(2)
    
    # Extract error type and message
    error_match = re.search(r'(\w+Error|\w+Exception|Error|Exception):\s*(.+?)(?:\n|$)', text)
    if error_match:
        details["error_type"] = error_match.group(1)
        details["error_message"] = error_match.group(2).strip()[:200]
    
    # Extract module name from ImportError
    module_match = re.search(r"(?:No module named|cannot import name|ImportError:)\s+['\"]?([a-zA-Z0-9_.]+)", text)
    if module_match:
        details["module_name"] = module_match.group(1)
    
    # Extract command that failed
    command_match = re.search(r"(?:Command|Executing|Running):\s*(.+?)(?:\n|failed|error)", text, re.IGNORECASE)
    if command_match:
        details["command"] = command_match.group(1).strip()[:100]
    
    return details


def generate_suggestion(log_text: str, label: str, include_fix_command: bool = False) -> dict:
    """Generate contextual suggestions with specific error details."""
    text = log_text
    text_lower = log_text.lower()
    
    # Extract specific error details
    error_details = extract_error_details(log_text)
    
    from app.utils.rca_detector import detect_root_cause, generate_fix_command
    
    # Use RCA detector for categorization
    category, rca_suggestion = detect_root_cause(log_text)
    
    # Build context-aware suggestion
    suggestion_parts = []
    
    # 1. Specific error-based suggestions with extracted details
    if error_details.get("error_type"):
        error_type = error_details["error_type"]
        
        if "ImportError" in error_type or "ModuleNotFoundError" in error_type:
            module = error_details.get("module_name") or "<module-name>"
            suggestion_parts.append(f"Missing Python module '{module}'. Install it with: pip install {module}")
            if error_details.get("file_path"):
                suggestion_parts.append(f"Error occurred in: {error_details['file_path']}")
        
        elif "SyntaxError" in error_type or "IndentationError" in error_type:
            if error_details.get("file_path") and error_details.get("line_number"):
                suggestion_parts.append(f"Syntax error in {error_details['file_path']} at line {error_details['line_number']}")
            suggestion_parts.append("Check for missing brackets, quotes, or indentation issues")
            if error_details.get("error_message"):
                suggestion_parts.append(f"Error: {error_details['error_message'][:100]}")
        
        elif "FileNotFoundError" in error_type or "OSError" in error_type:
            if error_details.get("file_path"):
                suggestion_parts.append(f"File not found: {error_details['file_path']}")
            suggestion_parts.append("Verify file exists and path is correct (check relative vs absolute paths)")
        
        elif "PermissionError" in error_type:
            if error_details.get("file_path"):
                suggestion_parts.append(f"Permission denied accessing: {error_details['file_path']}")
            suggestion_parts.append("Check file permissions or API token access rights")
        
        elif "KeyError" in error_type:
            if error_details.get("error_message"):
                key_match = re.search(r"['\"]([^'\"]+)['\"]", error_details["error_message"])
                if key_match:
                    suggestion_parts.append(f"Missing key '{key_match.group(1)}' in configuration or dictionary")
            suggestion_parts.append("Ensure all required keys are defined in your config/environment")
        
        elif "AttributeError" in error_type:
            if error_details.get("error_message"):
                attr_match = re.search(r"['\"]?([a-zA-Z_][a-zA-Z0-9_]*)['\"]?\s+has no attribute", error_details["error_message"])
                if attr_match:
                    suggestion_parts.append(f"Object '{attr_match.group(1)}' doesn't have the expected attribute")
            suggestion_parts.append("Check object type and available methods/attributes")
        
        elif "TypeError" in error_type:
            if error_details.get("error_message"):
                suggestion_parts.append(f"Type mismatch: {error_details['error_message'][:150]}")
            suggestion_parts.append("Verify variable types match expected function parameters")
        
        elif "AssertionError" in error_type or "TestFailure" in category:
            suggestion_parts.append("Test assertion failed")
            if error_details.get("file_path"):
                suggestion_parts.append(f"Check test file: {error_details['file_path']}")
            suggestion_parts.append("Review test logic and expected vs actual values")
        
        elif "TimeoutError" in error_type or "timeout" in text_lower:
            suggestion_parts.append("Operation timed out")
            suggestion_parts.append("Increase timeout in workflow file (timeout-minutes) or check network connectivity")
        
        elif "MemoryError" in error_type or "out of memory" in text_lower:
            suggestion_parts.append("Out of memory error")
            suggestion_parts.append("Reduce batch size, use larger CI runner, or optimize memory usage")
        
        elif "ConnectionError" in error_type or "connection refused" in text_lower:
            suggestion_parts.append("Connection error")
            suggestion_parts.append("Check network connectivity, API endpoints, or firewall settings")
    
    # 2. Language/framework-specific suggestions
    if not suggestion_parts:
        if re.search(r"pytest|unittest|test.*failed", text_lower):
            failed_test_match = re.search(r"(FAILED|failed)\s+([^\s]+\.py)", text)
            if failed_test_match:
                suggestion_parts.append(f"Test failed: {failed_test_match.group(2)}")
            suggestion_parts.append("Run tests locally with: pytest -v to see detailed failure")
        
        elif re.search(r"npm.*error|node.*error|yarn.*error", text_lower):
            if "package.json" in text_lower:
                suggestion_parts.append("Node.js dependency issue")
                suggestion_parts.append("Run: npm install or yarn install to update dependencies")
            elif "cannot find module" in text_lower:
                module_match = re.search(r"cannot find module ['\"]?([^'\"]+)", text_lower)
                if module_match:
                    suggestion_parts.append(f"Missing npm package: {module_match.group(1)}")
                    suggestion_parts.append(f"Install with: npm install {module_match.group(1)}")
        
        elif re.search(r"gradle|maven|javac", text_lower):
            suggestion_parts.append("Java build error")
            if "dependency" in text_lower:
                suggestion_parts.append("Check build.gradle or pom.xml for dependency issues")
            suggestion_parts.append("Verify Java version and build configuration")
        
        elif re.search(r"docker.*error|image.*not found|container.*failed", text_lower):
            image_match = re.search(r"image ['\"]?([^'\"]+)['\"]? not found", text_lower)
            if image_match:
                suggestion_parts.append(f"Docker image not found: {image_match.group(1)}")
                suggestion_parts.append(f"Pull image with: docker pull {image_match.group(1)}")
            else:
                suggestion_parts.append("Docker build failed - check Dockerfile syntax and base image")
        
        elif re.search(r"sql.*error|database.*error|psycopg2|sqlite", text_lower):
            suggestion_parts.append("Database error")
            if "connection" in text_lower:
                suggestion_parts.append("Check database connection string and credentials")
            elif "syntax" in text_lower or "query" in text_lower:
                suggestion_parts.append("Review SQL query syntax")
    
    # 3. Use RCA suggestion if no specific suggestions found
    if not suggestion_parts and rca_suggestion:
        suggestion_parts.append(rca_suggestion)
    
    # 4. Generic fallback based on label
    if not suggestion_parts:
        if label == "failed":
            # Try to extract any error line for context
            error_lines = [line for line in text.split('\n') if any(word in line.lower() for word in ['error', 'failed', 'exception', 'fatal'])]
            if error_lines:
                last_error = error_lines[-1].strip()[:150]
                suggestion_parts.append(f"Build failed. Last error: {last_error}")
            else:
                suggestion_parts.append("Build failed - review logs for specific error messages")
        elif label == "success":
            suggestion_parts.append("Build completed successfully")
        elif label == "skipped":
            suggestion_parts.append("Job was skipped - check workflow conditions")
        else:
            suggestion_parts.append("Review logs for configuration or runtime issues")
    
    # Combine suggestion parts
    suggestion_text = ". ".join(suggestion_parts) if suggestion_parts else "Inspect logs for potential issues."
    
    # Clean up suggestion
    suggestion_text = re.sub(r'\s+', ' ', suggestion_text).strip()
    
    result = {"suggestion": suggestion_text}
    
    # Generate fix command if requested
    if include_fix_command:
        fix_cmd = generate_fix_command(log_text, category)
        if fix_cmd:
            result["fix_command"] = fix_cmd
    
    return result


# ------------------------------------------------------------------
# Main Inference Logic
# ------------------------------------------------------------------
def predict_log_status(log_text: str):
    """Predict build status, confidence, and smart suggestion."""
    model, tokenizer = load_model_and_tokenizer()
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(
        log_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        conf, pred_class = torch.max(probs, dim=1)

    # Dynamic label mapping
    labels = ["success", "failed", "skipped", "unknown"]
    if model.config.num_labels <= len(labels):
        pred_label = labels[pred_class.item()]
    else:
        pred_label = "unknown"

    suggestion_data = generate_suggestion(log_text, pred_label, include_fix_command=True)
    suggestion = suggestion_data.get("suggestion", "Inspect logs for potential issues.")
    fix_command = suggestion_data.get("fix_command")
    
    logger.info(f"Prediction: {pred_label} ({conf.item():.2f}) — {suggestion}")
    if fix_command:
        logger.info(f"Fix command: {fix_command}")

    result = {
        "predicted_label": pred_label,
        "confidence": float(conf),
        "suggestion": suggestion
    }
    
    if fix_command:
        result["fix_command"] = fix_command
    
    return [result]
