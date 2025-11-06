import os
import yaml
import re
from pathlib import Path

# Get config path relative to this file
CONFIG_PATH = Path(__file__).parent.parent / "config" / "rca_rules.yaml"

def load_rca_rules():
    """Load RCA rules from YAML"""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load RCA rules: {e}")
        return []

def detect_root_cause(log_text: str):
    """Detect likely root cause and suggestion"""
    rules = load_rca_rules()

    for rule in rules:
        for kw in rule.get("keywords", []):
            if kw.lower() in log_text.lower():
                return rule["category"], rule["suggestion"]
    
    # Default case if nothing matches
    for rule in rules:
        if rule["category"] == "UnknownError":
            return rule["category"], rule["suggestion"]
    
    return "UnknownError", "Inspect error stack trace manually."

def generate_fix_command(log_text: str, category: str) -> str:
    """Generate actionable fix command based on error category"""
    text_lower = log_text.lower()
    
    # Dependency errors
    if category == "DependencyError":
        # Python
        if "module not found" in text_lower or "importerror" in text_lower:
            # Extract module name
            match = re.search(r"no module named ['\"]?([a-zA-Z0-9_-]+)", text_lower, re.IGNORECASE)
            if match:
                module = match.group(1)
                return f"pip install {module}"
            return "pip install <missing-package>"
        
        # Node.js
        if "cannot find module" in text_lower or "module not found" in text_lower:
            match = re.search(r"cannot find module ['\"]?([a-zA-Z0-9_-]+)", text_lower, re.IGNORECASE)
            if match:
                module = match.group(1)
                return f"npm install {module}"
            return "npm install <missing-package>"
        
        return "pip install <missing-package>  # or npm install for Node.js"
    
    # Permission errors
    if category == "PermissionError":
        return "chmod +x <file>  # or check API token permissions"
    
    # Docker errors
    if category == "DockerError":
        if "image not found" in text_lower:
            match = re.search(r"image ['\"]?([^'\"]+)['\"]? not found", text_lower, re.IGNORECASE)
            if match:
                image = match.group(1)
                return f"docker pull {image}"
            return "docker pull <image-name>"
        return "docker build -t <image-name> ."
    
    # Git errors
    if category == "GitError":
        if "merge conflict" in text_lower:
            return "git status  # then resolve conflicts manually"
        return "git pull origin <branch-name>"
    
    # Timeout errors
    if category == "TimeoutError":
        return "# Increase timeout in workflow file: timeout-minutes: 30"
    
    # Memory errors
    if category == "MemoryError":
        return "# Increase memory in workflow: runs-on: ubuntu-latest (or larger runner)"
    
    # Network errors
    if category == "NetworkError":
        return "# Check network connectivity or proxy settings"
    
    # Test failures
    if category == "TestFailure":
        return "pytest -v  # Run tests locally to debug"
    
    return None  # No specific command for this category