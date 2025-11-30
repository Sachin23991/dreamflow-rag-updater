#!/usr/bin/env python3
"""
Upload rag_storage/ to HuggingFace
(no max_retries bug, fully compatible)
"""

import os
from pathlib import Path
import time
from huggingface_hub import HfApi, login, upload_file

HF_REPO = os.environ.get("HF_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")
RAG_DIR = Path("rag_storage")

if not HF_REPO or not HF_TOKEN:
    raise SystemExit("HF_REPO and HF_TOKEN must be set")

def list_local_files():
    return sorted([p for p in RAG_DIR.iterdir() if p.is_file()])

def main():
    print("[HF-UPLOAD] Login...")
    login(HF_TOKEN)
    api = HfApi()

    print("[HF-UPLOAD] Listing remote...")
    try:
        remote_files = api.list_repo_files(repo_id=HF_REPO, repo_type="model")
    except:
        remote_files = []

    remote_set = set(remote_files)

    local_files = list_local_files()
    if not local_files:
        print("[HF-UPLOAD] No files in rag_storage/")
        return

    for p in local_files:
        path_in_repo = f"rag_storage/{p.name}"

        if path_in_repo in remote_set:
            print(f"[HF-UPLOAD] {p.name} exists → skipping")
            continue

        print(f"[HF-UPLOAD] Uploading {p.name}")
        try:
            upload_file(
                path_or_fileobj=str(p),
                path_in_repo=path_in_repo,
                repo_id=HF_REPO,
                repo_type="model",
                token=HF_TOKEN
            )
            print("[HF-UPLOAD] Uploaded:", p.name)
            time.sleep(0.2)
        except Exception as e:
            print("[HF-UPLOAD] Failed upload:", e)

    # Upload manifest (replace existing)
    manifest = RAG_DIR / "manifest.json"
    if manifest.exists():
        print("[HF-UPLOAD] Uploading manifest.json")
        try:
            upload_file(
                path_or_fileobj=str(manifest),
                path_in_repo=f"rag_storage/manifest.json",
                repo_id=HF_REPO,
                repo_type="model",
                token=HF_TOKEN
            )
            print("[HF-UPLOAD] Manifest uploaded.")
        except Exception as e:
            print("[HF-UPLOAD] Manifest upload FAILED:", e)

    print("[HF-UPLOAD] Done.")

if __name__ == "__main__":
    main()
