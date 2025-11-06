import os
import requests
import logging

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"

def get_github_headers():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logger.warning("GITHUB_TOKEN not set — skipping GitHub API actions.")
        return None
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

def fetch_github_log(repo_full_name: str, run_id: int) -> str:
    """
    Fetch the full workflow run log from GitHub Actions.
    Returns raw log text or empty string on failure.
    """
    headers = get_github_headers()
    if not headers:
        return ""

    url = f"{GITHUB_API_URL}/repos/{repo_full_name}/actions/runs/{run_id}/logs"
    logger.info(f"Fetching logs from {url}")
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        logger.error(f"Failed to fetch logs: {resp.status_code} {resp.text}")
        return ""

    return resp.text

def post_pr_comment(repo_full_name: str, pr_number: int, comment: str):
    """
    Post a comment to a PR.
    """
    headers = get_github_headers()
    if not headers:
        return

    url = f"{GITHUB_API_URL}/repos/{repo_full_name}/issues/{pr_number}/comments"
    payload = {"body": comment}
    resp = requests.post(url, headers=headers, json=payload)

    if resp.status_code == 201:
        logger.info(f"Comment posted to PR #{pr_number}")
    else:
        logger.error(f"Failed to post comment: {resp.status_code} {resp.text}")
