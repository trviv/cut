#!/usr/bin/env python3
"""Download GGUF models from Hugging Face to the models/ directory.

Usage:
    # Download a specific GGUF file from a repo:
    python scripts/download_model.py HuggingFaceTB/SmolLM2-135M-Instruct-GGUF smollm2-135m-instruct-q8_0.gguf

    # Download with a custom output name:
    python scripts/download_model.py HuggingFaceTB/SmolLM2-135M-Instruct-GGUF smollm2-135m-instruct-q8_0.gguf -o my_model.gguf

    # List available GGUF files in a repo:
    python scripts/download_model.py HuggingFaceTB/SmolLM2-135M-Instruct-GGUF --list

    # Search for SmolLM2 models:
    python scripts/download_model.py --search "SmolLM2 GGUF"
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, list_models


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def list_gguf_files(repo_id: str):
    """List all GGUF files in a Hugging Face repo."""
    files = list_repo_files(repo_id)
    gguf_files = [f for f in files if f.endswith(".gguf")]
    if not gguf_files:
        print(f"No .gguf files found in {repo_id}")
        print("All files:", ", ".join(sorted(files)))
        return
    print(f"GGUF files in {repo_id}:")
    for f in sorted(gguf_files):
        print(f"  {f}")


def search_models(query: str):
    """Search Hugging Face for models matching a query."""
    results = list_models(search=query, limit=20, sort="downloads")
    for model in results:
        print(f"  {model.id}  (downloads: {model.downloads})")


def download_model(repo_id: str, filename: str, output_name: str = None):
    """Download a GGUF file from Hugging Face to models/."""
    MODELS_DIR.mkdir(exist_ok=True)
    dest_name = output_name or filename
    dest_path = MODELS_DIR / dest_name

    if dest_path.exists():
        print(f"Already exists: {dest_path}")
        return str(dest_path)

    print(f"Downloading {repo_id}/{filename} ...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(MODELS_DIR),
    )
    # hf_hub_download saves with original filename in local_dir
    actual = MODELS_DIR / filename
    if output_name and actual.exists() and not dest_path.exists():
        actual.rename(dest_path)
        print(f"Saved to {dest_path}")
    else:
        print(f"Saved to {actual}")
    return str(dest_path if dest_path.exists() else actual)


def main():
    parser = argparse.ArgumentParser(description="Download GGUF models from Hugging Face")
    parser.add_argument("repo_id", nargs="?", help="HF repo (e.g. HuggingFaceTB/SmolLM2-135M-Instruct-GGUF)")
    parser.add_argument("filename", nargs="?", help="GGUF filename to download")
    parser.add_argument("-o", "--output", help="Custom output filename")
    parser.add_argument("--list", action="store_true", help="List GGUF files in the repo")
    parser.add_argument("--search", help="Search HF for models")
    args = parser.parse_args()

    if args.search:
        print(f"Searching for '{args.search}'...")
        search_models(args.search)
        return

    if not args.repo_id:
        parser.print_help()
        sys.exit(1)

    if args.list:
        list_gguf_files(args.repo_id)
        return

    if not args.filename:
        print("No filename specified. Available GGUF files:")
        list_gguf_files(args.repo_id)
        sys.exit(1)

    download_model(args.repo_id, args.filename, args.output)


if __name__ == "__main__":
    main()
