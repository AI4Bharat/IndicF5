#!/usr/bin/env python3
"""Utility script to pre-download model weights for offline or initial setup."""

import os
from pathlib import Path

def download():
    print("=== Downloading IndicF5 Models ===")
    
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("Error: 'huggingface_hub' is not installed. Please run 'pip install huggingface_hub' first.")
        return

    # 1. Main IndicF5 Model
    model_repo = "ai4bharat/IndicF5"
    local_dir = Path("models/IndicF5")
    
    print(f"Downloading main IndicF5 model from '{model_repo}' to '{local_dir}'...")
    try:
        snapshot_download(
            repo_id=model_repo,
            local_dir=local_dir,
            ignore_patterns=["*.git*", "*.gitattributes"]
        )
        print(f"✓ IndicF5 model saved to {local_dir.absolute()}")
    except Exception as e:
        print(f"Error downloading main model: {str(e)}")
        return

    # 2. Vocoder Model
    vocoder_repo = "charactr/vocos-mel-24khz"
    print(f"Downloading Vocos vocoder from '{vocoder_repo}'...")
    try:
        hf_hub_download(repo_id=vocoder_repo, filename="config.yaml")
        hf_hub_download(repo_id=vocoder_repo, filename="pytorch_model.bin")
        print("✓ Vocos vocoder downloaded and cached successfully.")
    except Exception as e:
        print(f"Error downloading vocoder: {str(e)}")
        return
    
    print("\nAll models downloaded successfully and ready to use!")

if __name__ == "__main__":
    download()
