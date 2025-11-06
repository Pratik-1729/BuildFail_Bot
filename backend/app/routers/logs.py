from fastapi import APIRouter, Request, BackgroundTasks, HTTPException
import os, io, zipfile, datetime, logging, requests, pandas as pd, csv, re
from app.ml.inference import predict_log_status
from app.config.settings import INGESTED_LOGS_FILE, FEEDBACK_DIR, GITHUB_TOKEN

router = APIRouter()
logger = logging.getLogger(__name__)

DATA_DIR = str(FEEDBACK_DIR)
INGESTED_FILE = str(INGESTED_LOGS_FILE)

# ---------------------------------------------------------
# CSV Append (Safe and Schema Consistent)
# ---------------------------------------------------------
def save_ingested(record: dict):
    expected_cols = [
        "timestamp", "repo", "run_id", "status", "label",
        "confidence", "failed_step", "suggestion", "clean_log_excerpt"
    ]
    df = pd.DataFrame([record])
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "-"
    df = df[expected_cols]

    header = not os.path.exists(INGESTED_FILE)
    try:
        df.to_csv(
            INGESTED_FILE,
            mode="a",
            index=False,
            header=header,
            quoting=csv.QUOTE_ALL,
            escapechar="\\",
            encoding="utf-8",
        )
        logger.info("✅ Log record saved successfully.")
    except Exception:
        logger.exception("❌ Error saving ingested record")


# ---------------------------------------------------------
# Log Cleaners
# ---------------------------------------------------------
def clean_log_text(text: str) -> str:
    """Remove binary, timestamps, runner metadata, and color codes."""
    if not text:
        return ""
    text = re.sub(r"\x1B\[[0-9;]*[A-Za-z]", "", text)  # color codes
    text = text.replace("\x00", "").replace("\r", "")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z", "", text)
    text = re.sub(r"##\[[^\]]+\]", "", text)
    text = re.sub(r"::group::|::endgroup::", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------
# Smart Contextual Extractor
# ---------------------------------------------------------
def extract_meaningful_part(log_text: str) -> str:
    """Extract context-rich segments (error, test, build, warning)."""
    if not log_text:
        return ""

    lines = log_text.splitlines()
    meaningful, keep_next = [], 0

    noise_patterns = [
        r"Ubuntu|Runner Image|Operating System|GITHUB_TOKEN|Hosted Compute Agent",
        r"Cache hit|Downloading|Installing|Requirement already satisfied",
        r"Triggering workflow|Uploading artifacts|Job succeeded|##\[section\]",
        r"All tests passed|Workflow complete|Image Release|Version:",
    ]
    signal_patterns = [
        r"error|exception|fail|traceback|warning|pytest|unittest|build|compile|run|stage|step|executing|ci|workflow"
    ]

    noise_re = re.compile("|".join(noise_patterns), re.IGNORECASE)
    signal_re = re.compile("|".join(signal_patterns), re.IGNORECASE)

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or noise_re.search(line):
            continue

        if signal_re.search(line):
            meaningful.append(line)
            keep_next = 5
        elif keep_next > 0:
            meaningful.append(line)
            keep_next -= 1
        elif i % 50 == 0:
            meaningful.append(line)

    excerpt = "\n".join(meaningful)
    
    # If no meaningful content found, take first 500 chars
    if not excerpt.strip() or len(excerpt.strip()) < 20:
        excerpt = log_text[:500].strip()
    
    # Clean up whitespace but preserve line breaks for readability
    # Only collapse multiple spaces/tabs, keep newlines
    lines = excerpt.split('\n')
    cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
    excerpt = "\n".join([line for line in cleaned_lines if line])
    
    return excerpt.strip()


# ---------------------------------------------------------
# Fetch GitHub Logs (Raw or ZIP)
# ---------------------------------------------------------
def fetch_github_logs(payload):
    try:
        workflow = payload.get("workflow_run", {})
        logs_url = workflow.get("logs_url")
        if not logs_url:
            logger.warning("⚠️ No logs_url in payload.")
            return None

        token = GITHUB_TOKEN or os.getenv("GITHUB_TOKEN")
        if not token:
            logger.warning("⚠️ Missing GITHUB_TOKEN.")
            return None

        # Try raw text logs
        headers = {"Authorization": f"token {token}", "Accept": "text/plain"}
        resp = requests.get(logs_url, headers=headers, timeout=30)
        if resp.ok and len(resp.text.strip()) > 300:
            logger.info("📄 Raw logs fetched successfully.")
            return {"__combined__": clean_log_text(resp.text)}

        # Fallback to ZIP
        logger.warning("⚠️ Raw logs unavailable — trying ZIP.")
        resp = requests.get(logs_url, headers={"Authorization": f"token {token}"}, timeout=30)
        if not resp.ok:
            logger.warning(f"⚠️ ZIP fetch failed ({resp.status_code}).")
            return None

        logs, combined = {}, ""
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for name in z.namelist():
                if name.endswith(".txt"):
                    with z.open(name) as f:
                        text = f.read().decode("utf-8", errors="ignore")
                        text = clean_log_text(text)
                        logs[name] = text
                        if "build" in name.lower() or "error" in name.lower():
                            combined += "\n" + text
        logs["__combined__"] = combined or "\n".join(logs.values())
        logger.info(f"📦 Extracted {len(logs)} logs from ZIP.")
        return logs
    except Exception:
        logger.exception("❌ Error fetching logs")
        return None


# ---------------------------------------------------------
# Main Log Processor
# ---------------------------------------------------------
def process_build_log(payload: dict):
    """Main processing pipeline for CI/CD logs."""
    try:
        # Extract repository information
        repo_data = payload.get("repository", {})
        repo = repo_data.get("full_name") or repo_data.get("name") or "-"
        
        # Extract workflow run information
        workflow_run = payload.get("workflow_run", {})
        run_id = str(workflow_run.get("id", "-"))
        status = workflow_run.get("conclusion") or workflow_run.get("status") or "-"
        
        # Generate ISO timestamp
        ts = datetime.datetime.utcnow().isoformat()

        logs = fetch_github_logs(payload)
        if not logs:
            logger.warning("⚠️ No logs fetched — skipping.")
            return

        log_text = logs.get("__combined__", "")
        if not log_text.strip():
            logger.warning("⚠️ Empty log text — skipping.")
            return

        clean_text = clean_log_text(log_text)
        meaningful_excerpt = extract_meaningful_part(clean_text)
        
        # Ensure excerpt is not empty and has meaningful content
        if not meaningful_excerpt or len(meaningful_excerpt.strip()) < 10:
            # Fallback to first 500 chars of cleaned text
            meaningful_excerpt = clean_text[:500].strip()
        
        # Limit excerpt to 2000 chars
        meaningful_excerpt = meaningful_excerpt[:2000]

        # ML Inference for Label + Confidence + Suggestion
        try:
            result = predict_log_status(clean_text)
            result = result[0] if isinstance(result, list) else result
            label = result.get("predicted_label", "unknown")
            confidence = result.get("confidence", 0.0)
            suggestion = result.get("suggestion", "Inspect error trace for issues.")
        except Exception:
            logger.exception("Model inference failed.")
            label, confidence, suggestion = "error", 0.0, "Failed during model prediction."

        record = {
            "timestamp": ts,
            "repo": repo,
            "run_id": run_id,
            "status": status,
            "label": label,
            "confidence": confidence,
            "failed_step": "-",
            "suggestion": suggestion,
            "clean_log_excerpt": meaningful_excerpt
        }

        save_ingested(record)
        logger.info(f"✅ Processed log: {repo} | {label.upper()} | {suggestion}")
        
        # Check if we should trigger automatic retraining (every 50 new logs)
        try:
            if os.path.exists(INGESTED_FILE):
                df = pd.read_csv(INGESTED_FILE, on_bad_lines="skip", engine="python")
                new_log_count = len(df)
                # Trigger auto-retrain every 50 new logs
                if new_log_count > 0 and new_log_count % 50 == 0:
                    logger.info(f"🔄 {new_log_count} logs accumulated. Consider retraining model.")
        except Exception:
            pass  # Don't fail if auto-retrain check fails

    except Exception:
        logger.exception("❌ Unexpected error in process_build_log")


@router.post("/manual")
async def manual_log_analysis(request: Request):
    """Handle log input from Chrome Extension. Analyzes and optionally saves log."""
    try:
        data = await request.json()
        log_text = data.get("log_text", "")
        save_log = data.get("save_log", True)  # Default to saving for retraining
        
        if not log_text:
            raise HTTPException(status_code=400, detail="Missing log text.")

        # Clean and extract meaningful parts
        clean_text = clean_log_text(log_text)
        meaningful_excerpt = extract_meaningful_part(clean_text)

        # ML Inference (includes fix_command)
        result = predict_log_status(clean_text)[0]
        label = result.get("predicted_label", "unknown")
        confidence = result.get("confidence", 0.0)
        suggestion = result.get("suggestion", "Inspect error trace for issues.")
        fix_command = result.get("fix_command")  # New: actionable fix command

        # Save to ingested_logs.csv if requested (default: True)
        if save_log:
            try:
                ts = datetime.datetime.utcnow().isoformat()
                
                # Ensure excerpt is meaningful
                if not meaningful_excerpt or len(meaningful_excerpt.strip()) < 10:
                    meaningful_excerpt = clean_text[:500].strip()
                
                # Extract repo from request data first (from extension URL)
                repo = data.get("repo")
                
                # If not provided, try to extract from log text
                if not repo or repo == "-" or repo == "manual_analysis":
                    # Try to extract from URL if log_text contains GitHub URL or path
                    # Pattern 1: github.com/owner/repo
                    url_match = re.search(r"github\.com/([^/\s]+/[^/\s]+)", log_text)
                    if url_match:
                        repo = url_match.group(1)
                    else:
                        # Pattern 2: Extract from common GitHub Actions paths
                        # /home/runner/work/owner/repo-name/repo-name
                        work_path_match = re.search(r"/home/runner/work/([^/\s]+)/([^/\s]+)", log_text)
                        if work_path_match:
                            owner = work_path_match.group(1)
                            repo_name = work_path_match.group(2)
                            repo = f"{owner}/{repo_name}"
                        else:
                            # Pattern 3: Look for repository name in file paths
                            file_path_match = re.search(r"/([a-zA-Z0-9_-]+)/_test\.py|/([a-zA-Z0-9_-]+)/.*\.py", log_text)
                            if file_path_match:
                                repo_name = file_path_match.group(1) or file_path_match.group(2)
                                # Try to find owner from URL or use default
                                owner_match = re.search(r"github\.com/([^/\s]+)/", log_text)
                                owner = owner_match.group(1) if owner_match else "unknown"
                                repo = f"{owner}/{repo_name}"
                            else:
                                # Pattern 4: Try simple path match
                                path_match = re.search(r"/home/runner/work/([^/\s]+)", log_text)
                                if path_match:
                                    repo = path_match.group(1).replace("-", "/")
                                else:
                                    repo = "manual_analysis"
                
                # Use run_id from request if provided (from extension URL)
                run_id = data.get("run_id") or "-"
                status = data.get("status") or "-"
                
                record = {
                    "timestamp": ts,
                    "repo": repo,
                    "run_id": run_id,
                    "status": status,
                    "label": label,
                    "confidence": confidence,
                    "failed_step": "-",
                    "suggestion": suggestion,
                    "clean_log_excerpt": meaningful_excerpt[:2000]
                }
                save_ingested(record)
                logger.info(f"✅ Manual log saved: {repo} | {label.upper()} | {suggestion}")
            except Exception as e:
                logger.warning(f"⚠️ Could not save manual log: {e}")

        response = {
            "predicted_label": label,
            "confidence": round(confidence, 4),
            "suggestion": suggestion,
            "saved": save_log
        }
        
        # Include fix command if available
        if fix_command:
            response["fix_command"] = fix_command
        
        return response
    except Exception as e:
        logger.exception("❌ Error analyzing log manually.")
        raise HTTPException(status_code=500, detail=str(e))
# ---------------------------------------------------------
# Webhook Endpoint
# ---------------------------------------------------------
@router.post("/webhook")
async def receive_ci_event(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        background_tasks.add_task(process_build_log, data)
        return {"status": "accepted", "message": "CI/CD logs processing in background."}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    