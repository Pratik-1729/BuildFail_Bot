"""
GitHub API Polling Service
Automatically fetches logs from GitHub Actions without webhooks
"""
import os
import time
import requests
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from app.config.settings import GITHUB_TOKEN, INGESTED_LOGS_FILE
from app.routers.logs import process_build_log, clean_log_text, extract_meaningful_part
import pandas as pd

logger = logging.getLogger(__name__)

class GitHubActionsPoller:
    """Poll GitHub Actions API for new workflow runs."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or GITHUB_TOKEN or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("GITHUB_TOKEN not found in environment")
        
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.base_url = "https://api.github.com"
        self.last_poll_time = {}
        self.processed_runs = set()
    
    def get_repositories(self, org_or_user: str) -> List[Dict]:
        """Get all repositories for an organization or user."""
        url = f"{self.base_url}/users/{org_or_user}/repos"
        params = {"per_page": 100, "sort": "updated"}
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            repos = response.json()
            logger.info(f"✅ Found {len(repos)} repositories for {org_or_user}")
            return repos
        except Exception as e:
            logger.error(f"❌ Error fetching repositories: {e}")
            return []
    
    def get_workflow_runs(self, owner: str, repo: str, since: Optional[datetime] = None) -> List[Dict]:
        """Get workflow runs for a repository."""
        url = f"{self.base_url}/repos/{owner}/{repo}/actions/runs"
        params = {
            "per_page": 100,
            "status": "completed"
        }
        
        if since:
            params["created"] = since.isoformat()
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            runs = data.get("workflow_runs", [])
            logger.info(f"✅ Found {len(runs)} workflow runs for {owner}/{repo}")
            return runs
        except Exception as e:
            logger.error(f"❌ Error fetching workflow runs for {owner}/{repo}: {e}")
            return []
    
    def get_run_logs(self, owner: str, repo: str, run_id: int) -> Optional[str]:
        """Fetch logs for a specific workflow run."""
        url = f"{self.base_url}/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # GitHub returns logs as ZIP
            import zipfile
            import io
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                logs = []
                for name in z.namelist():
                    if name.endswith(".txt"):
                        with z.open(name) as f:
                            logs.append(f.read().decode("utf-8", errors="ignore"))
                
                return "\n".join(logs)
        except Exception as e:
            logger.error(f"❌ Error fetching logs for run {run_id}: {e}")
            return None
    
    def process_run(self, owner: str, repo: str, run: Dict):
        """Process a workflow run and ingest logs."""
        run_id = run.get("id")
        run_url = run.get("html_url", "")
        
        # Skip if already processed
        run_key = f"{owner}/{repo}/{run_id}"
        if run_key in self.processed_runs:
            return
        
        conclusion = run.get("conclusion")
        status = run.get("status")
        
        # Only process completed runs (success or failure)
        if status != "completed":
            return
        
        logger.info(f"📥 Processing run: {owner}/{repo} #{run_id} ({conclusion})")
        
        # Fetch logs
        logs = self.get_run_logs(owner, repo, run_id)
        if not logs or len(logs.strip()) < 50:
            logger.warning(f"⚠️ No logs found for run {run_id}")
            return
        
        # Create webhook-like payload
        payload = {
            "repository": {
                "full_name": f"{owner}/{repo}",
                "name": repo
            },
            "workflow_run": {
                "id": run_id,
                "conclusion": conclusion,
                "status": status,
                "html_url": run_url,
                "logs_url": f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/logs"
            }
        }
        
        # Process using existing pipeline
        try:
            process_build_log(payload)
            self.processed_runs.add(run_key)
            logger.info(f"✅ Processed run {run_id}")
        except Exception as e:
            logger.error(f"❌ Error processing run {run_id}: {e}")
    
    def poll_repositories(self, orgs_or_users: List[str], poll_interval: int = 300):
        """
        Continuously poll repositories for new workflow runs.
        
        Args:
            orgs_or_users: List of GitHub orgs/users to monitor
            poll_interval: Seconds between polls (default: 5 minutes)
        """
        logger.info(f"🔄 Starting polling service for: {orgs_or_users}")
        logger.info(f"⏰ Poll interval: {poll_interval} seconds")
        
        while True:
            try:
                for org_user in orgs_or_users:
                    repos = self.get_repositories(org_user)
                    
                    for repo in repos:
                        owner = repo.get("owner", {}).get("login", org_user)
                        repo_name = repo.get("name")
                        
                        # Get last poll time for this repo
                        repo_key = f"{owner}/{repo_name}"
                        last_poll = self.last_poll_time.get(repo_key)
                        since = last_poll if last_poll else datetime.utcnow() - timedelta(hours=24)
                        
                        # Get recent runs
                        runs = self.get_workflow_runs(owner, repo_name, since=since)
                        
                        # Process each run
                        for run in runs:
                            self.process_run(owner, repo_name, run)
                        
                        # Update last poll time
                        self.last_poll_time[repo_key] = datetime.utcnow()
                
                # Wait before next poll
                logger.info(f"⏳ Waiting {poll_interval} seconds before next poll...")
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Polling stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in polling loop: {e}")
                time.sleep(60)  # Wait 1 minute on error

def start_polling_service(orgs_or_users: List[str], poll_interval: int = 300):
    """Start the polling service."""
    poller = GitHubActionsPoller()
    poller.poll_repositories(orgs_or_users, poll_interval)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Actions Polling Service")
    parser.add_argument("--orgs", nargs="+", required=True, help="GitHub orgs/users to monitor")
    parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds (default: 300)")
    
    args = parser.parse_args()
    
    start_polling_service(args.orgs, args.interval)

