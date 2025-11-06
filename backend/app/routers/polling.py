"""
API endpoints for GitHub polling service
"""
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from app.services.github_poller import GitHubActionsPoller

router = APIRouter()
logger = logging.getLogger(__name__)

class PollingConfig(BaseModel):
    orgs_or_users: List[str]
    poll_interval: int = 300  # seconds

# Store active pollers
active_pollers = {}

@router.post("/start")
async def start_polling(config: PollingConfig, background_tasks: BackgroundTasks):
    """Start polling GitHub Actions for specified orgs/users."""
    try:
        poller = GitHubActionsPoller()
        
        # Start polling in background
        background_tasks.add_task(
            poller.poll_repositories,
            config.orgs_or_users,
            config.poll_interval
        )
        
        active_pollers[",".join(config.orgs_or_users)] = poller
        
        return {
            "status": "started",
            "message": f"Polling started for: {', '.join(config.orgs_or_users)}",
            "interval": config.poll_interval
        }
    except Exception as e:
        logger.exception("Error starting polling")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
def get_polling_status():
    """Get status of active polling services."""
    return {
        "active_pollers": len(active_pollers),
        "monitored_orgs": list(active_pollers.keys())
    }

@router.post("/stop")
def stop_polling(orgs: List[str]):
    """Stop polling for specified orgs/users."""
    key = ",".join(orgs)
    if key in active_pollers:
        del active_pollers[key]
        return {"status": "stopped", "message": f"Polling stopped for: {', '.join(orgs)}"}
    else:
        raise HTTPException(status_code=404, detail="No active poller found")

