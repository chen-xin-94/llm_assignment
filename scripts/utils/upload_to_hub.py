import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub import create_repo


def upload_to_hub(local_dir, repo_id, repo_type="model", token=None):
    """
    Uploads a local directory to Hugging Face Hub.
    """
    if token is None:
        token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN not found. Please set it in .env or pass as argument.")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type=repo_type, token=token, exist_ok=True)
        print(f"Repository {repo_id} ({repo_type}) ready.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        return

    base_path = Path(__file__).resolve().parent.parent.parent
    path_to_upload = base_path / local_dir

    if not path_to_upload.exists():
        print(f"Error: Local directory '{local_dir}' not found at {path_to_upload}")
        return

    allow_patterns = None
    if local_dir == "checkpoints":
        allow_patterns = [
            "dropped/final/**",
            "dropped_lora/final/**",
            "original/final/**",
            "original_lora/final/**",
            "pruned/final/**",
            "pruned_lora/final/**",
        ]
        print(f"Applying allow_patterns for checkpoints: {allow_patterns}")

    print(f"Uploading '{local_dir}' to {repo_id} ({repo_type})...")
    try:
        api.upload_folder(
            folder_path=str(path_to_upload),
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=".",  # Upload content directly to root of repo
            token=token,
            allow_patterns=allow_patterns,
        )
        print(f"Successfully uploaded {local_dir} to {repo_id}")
    except Exception as e:
        print(f"Failed to upload {local_dir}: {e}")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Upload checkpoints or data to Hugging Face Hub")
    parser.add_argument(
        "--folder", type=str, required=True, help="Local folder to upload (e.g., 'data' or 'checkpoints')"
    )
    parser.add_argument(
        "--repo-id", type=str, required=True, help="Hugging Face repository ID (e.g., username/repo_name)"
    )
    parser.add_argument(
        "--type", type=str, default="model", choices=["model", "dataset"], help="Repository type: 'model' or 'dataset'"
    )

    args = parser.parse_args()

    upload_to_hub(args.folder, args.repo_id, args.type)
