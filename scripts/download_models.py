"""
Download models from Hugging Face Hub to local models/ directory
Run this script before starting the application if models don't exist locally.

Usage:
    python scripts/download_models.py --username YOUR_HF_USERNAME
"""
import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env file
project_root = Path(__file__).parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded environment variables from: {env_file}")
else:
    print(f"⚠️  .env file not found at: {env_file}")
    print("   Using environment variables from system or command line")

from backend.app.config.settings import (
    DISTILBERT_BEST_DIR,
    DISTILBERT_TRAINED_DIR,
    DISTILBERT_RETRAINED_DIR
)

def download_model(repo_id: str, local_dir: Path, token: str = None):
    """Download a model from Hugging Face Hub."""
    print(f"📥 Downloading {repo_id}...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=token,
        )
        print(f"✅ Successfully downloaded to: {local_dir}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {repo_id}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download BuildFail Bot models from Hugging Face Hub"
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Your Hugging Face username"
    )
    parser.add_argument(
        "--token",
        help="Hugging Face token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--model",
        choices=["best", "trained", "retrained", "all"],
        default="all",
        help="Which model to download (default: all)"
    )
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("⚠️  Warning: No token provided. Public repos can be downloaded without token.")
        print("   For private repos, set HF_TOKEN or use --token")
    
    username = args.username
    success_count = 0
    total_count = 0
    
    # Download models based on selection
    models_to_download = []
    
    if args.model == "all" or args.model == "best":
        models_to_download.append((
            f"{username}/buildfail-bot-distilbert_best",
            DISTILBERT_BEST_DIR,
            "Best Model"
        ))
    
    if args.model == "all" or args.model == "trained":
        models_to_download.append((
            f"{username}/buildfail-bot-distilbert_trained",
            DISTILBERT_TRAINED_DIR,
            "Trained Model"
        ))
    
    if args.model == "all" or args.model == "retrained":
        models_to_download.append((
            f"{username}/buildfail-bot-distilbert_retrained",
            DISTILBERT_RETRAINED_DIR,
            "Retrained Model"
        ))
    
    print(f"\n🚀 Downloading {len(models_to_download)} model(s) from Hugging Face Hub...")
    print(f"   Username: {username}\n")
    
    for repo_id, local_dir, model_name in models_to_download:
        total_count += 1
        print(f"\n[{total_count}/{len(models_to_download)}] {model_name}")
        print(f"   Repo: {repo_id}")
        print(f"   Local: {local_dir}")
        
        # Create directory if it doesn't exist
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        
        if download_model(repo_id, local_dir, token):
            success_count += 1
        else:
            print(f"   ⚠️  Skipping {model_name}")
    
    print(f"\n{'='*60}")
    print(f"✅ Download complete: {success_count}/{total_count} models downloaded")
    
    if success_count == total_count:
        print("🎉 All models downloaded successfully!")
        print("   You can now start the application.")
    elif success_count > 0:
        print("⚠️  Some models failed to download.")
        print("   The application will use base DistilBERT for missing models.")
    else:
        print("❌ No models downloaded.")
        print("   The application will use base DistilBERT model.")

if __name__ == "__main__":
    main()

