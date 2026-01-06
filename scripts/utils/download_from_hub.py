import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def download_from_hub(repo_id, local_dir, repo_type="model", token=None):
    """
    Downloads a repository from Hugging Face Hub to a local directory.
    """
    if token is None:
        token = os.getenv("HF_TOKEN")

    base_path = Path(__file__).resolve().parent.parent.parent
    local_path = base_path / local_dir

    print(f"Downloading {repo_id} ({repo_type}) to {local_path}...")

    try:
        snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_path, token=token, resume_download=True)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Download checkpoints or data from Hugging Face Hub")
    parser.add_argument(
        "--repo-id", type=str, required=True, help="Hugging Face repository ID (e.g., username/repo_name)"
    )
    parser.add_argument(
        "--folder", type=str, required=True, help="Local directory to download to (e.g., 'data' or 'checkpoints')"
    )
    parser.add_argument(
        "--type", type=str, default="model", choices=["model", "dataset"], help="Repository type: 'model' or 'dataset'"
    )

    args = parser.parse_args()

    download_from_hub(args.repo_id, args.folder, args.type)
