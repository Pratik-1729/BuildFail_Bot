"""
Download models from Hugging Face Hub
Usage: python scripts/download_from_hf_hub.py
"""
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
import argparse
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)

def download_model(repo_id: str, local_dir: str = None):
    """Download a model from Hugging Face Hub."""
    if not local_dir:
        # Default to models/ directory
        project_root = Path(__file__).parent.parent
        model_name = repo_id.split("/")[-1]
        local_dir = project_root / "models" / model_name
    
    print(f"📥 Downloading {repo_id} to {local_dir}...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=os.getenv("HF_TOKEN"),
        )
        print(f"✅ Successfully downloaded to: {local_dir}")
        return True
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download models from Hugging Face Hub")
    parser.add_argument("repo_id", help="Hugging Face repo ID (e.g., username/model-name)")
    parser.add_argument("--local-dir", help="Local directory to save model")
    parser.add_argument("--token", help="Hugging Face token (or set HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Check token
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("⚠️ No token provided. Public repos can be downloaded without token.")
        print("   For private repos, set HF_TOKEN or use --token")
    
    download_model(args.repo_id, args.local_dir)

if __name__ == "__main__":
    main()

