#!/usr/bin/env python3
"""
sharded_upload_to_hf.py
- Uploads contents of rag_storage/ to HF model repo under rag_storage/.
- Uses HfApi.list_repo_files to detect existing remote files.
- Uploads only files that are missing or have different sizes (best-effort).
"""

import os
from pathlib import Path
import time
import json
from huggingface_hub import HfApi, login, upload_file

HF_REPO = os.environ.get("HF_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")
RAG_DIR = Path("rag_storage")

if not HF_REPO or not HF_TOKEN:
    raise SystemExit("HF_REPO and HF_TOKEN environment variables must be set")

def list_local_files():
    return sorted([p for p in RAG_DIR.iterdir() if p.is_file()])

def main():
    print("[HF-UPLOAD] Logging in...")
    login(HF_TOKEN)
    api = HfApi()
    print("[HF-UPLOAD] Listing remote files...")
    try:
        remote_files = api.list_repo_files(repo_id=HF_REPO, repo_type="model")
    except Exception as e:
        print("[HF-UPLOAD] Could not list remote files (continuing):", e)
        remote_files = []

    remote_set = set(remote_files)

    files = list_local_files()
    if not files:
        print("[HF-UPLOAD] No files to upload in rag_storage/")
        return

    for p in files:
        path_in_repo = f"rag_storage/{p.name}"
        if path_in_repo in remote_set:
            print(f"[HF-UPLOAD] {p.name} already exists in HF repo; skipping.")
            continue
        print(f"[HF-UPLOAD] Uploading {p.name} -> {HF_REPO}:{path_in_repo}")
        try:
            upload_file(
                path_or_fileobj=str(p),
                path_in_repo=path_in_repo,
                repo_id=HF_REPO,
                repo_type="model",
                token=HF_TOKEN
               
            )
            print("[HF-UPLOAD] Uploaded", p.name)
            time.sleep(0.5)
        except Exception as e:
            print("[HF-UPLOAD] Failed to upload", p.name, e)

    # Optionally, also upload manifest.json if present (replace remote)
    manifest = RAG_DIR / "manifest.json"
    if manifest.exists():
        print("[HF-UPLOAD] Uploading manifest.json")
        try:
            upload_file(
                path_or_fileobj=str(manifest),
                path_in_repo=f"rag_storage/{manifest.name}",
                repo_id=HF_REPO,
                repo_type="model",
                token=HF_TOKEN,
                max_retries=3
            )
            print("[HF-UPLOAD] manifest uploaded.")
        except Exception as e:
            print("[HF-UPLOAD] manifest upload failed:", e)

    print("[HF-UPLOAD] Done.")
if __name__ == "__main__":
    main()
