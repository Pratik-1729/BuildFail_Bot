"""
Upload models and results to Hugging Face Hub
Usage: python scripts/upload_to_hf_hub.py
"""
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, login
from huggingface_hub import upload_folder
import argparse
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

def upload_model_to_hf(model_path: str, repo_id: str, private: bool = False):
    """Upload a model directory to Hugging Face Hub."""
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        return False
    
    print(f"📤 Uploading {model_path} to {repo_id}...")
    
    try:
        # Create repo if it doesn't exist (with private setting)
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="model",
                private=private,
                exist_ok=True
            )
        except Exception as e:
            # Repo might already exist, that's okay
            print(f"   ℹ️  Repo creation check: {str(e)[:100]}")
        
        # Upload the entire folder
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=os.getenv("HF_TOKEN"),
        )
        print(f"✅ Successfully uploaded to: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        return False

def upload_all_models(hf_username: str, private: bool = False):
    """Upload all models in the models/ directory."""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    
    if not models_dir.exists():
        print("❌ Models directory not found")
        return
    
    # Find all model directories
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("⚠️ No model directories found")
        return
    
    print(f"📦 Found {len(model_dirs)} model directories:")
    for model_dir in model_dirs:
        print(f"  - {model_dir.name}")
    
    for model_dir in model_dirs:
        repo_id = f"{hf_username}/buildfail-bot-{model_dir.name}"
        upload_model_to_hf(str(model_dir), repo_id, private)

def upload_results(hf_username: str, private: bool = False):
    """Upload results directory as a dataset."""
    api = HfApi(token=os.getenv("HF_TOKEN"))
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    
    if not results_dir.exists():
        print("❌ Results directory not found")
        return
    
    repo_id = f"{hf_username}/buildfail-bot-results"
    print(f"📤 Uploading results to {repo_id}...")
    
    try:
        # Create repo if it doesn't exist (with private setting)
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=True
            )
        except Exception as e:
            # Repo might already exist, that's okay
            print(f"   ℹ️  Repo creation check: {str(e)[:100]}")
        
        # Upload the entire folder
        upload_folder(
            folder_path=str(results_dir),
            repo_id=repo_id,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
        )
        print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"❌ Error uploading: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload models to Hugging Face Hub")
    parser.add_argument("--username", required=True, help="Hugging Face username")
    parser.add_argument("--token", help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make repos private")
    parser.add_argument("--models-only", action="store_true", help="Upload only models")
    parser.add_argument("--results-only", action="store_true", help="Upload only results")
    parser.add_argument("--model-path", help="Upload specific model directory")
    parser.add_argument("--repo-id", help="Specific repo ID for model-path")
    
    args = parser.parse_args()
    
    # Login to HF
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found. Please set it as environment variable or use --token")
        print("   Get your token from: https://huggingface.co/settings/tokens")
        return
    
    login(token=token)
    print("✅ Logged in to Hugging Face")
    
    # Upload specific model
    if args.model_path:
        if not args.repo_id:
            print("❌ --repo-id required when using --model-path")
            return
        upload_model_to_hf(args.model_path, args.repo_id, args.private)
        return
    
    # Upload all models
    if not args.results_only:
        upload_all_models(args.username, args.private)
    
    # Upload results
    if not args.models_only:
        upload_results(args.username, args.private)

if __name__ == "__main__":
    main()

