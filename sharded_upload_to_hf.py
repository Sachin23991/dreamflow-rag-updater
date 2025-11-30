#!/usr/bin/env python3
"""
Upload rag_storage/ to HuggingFace
- Compares local file size with remote file size
- Uploads if file is new OR updated
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

def list_local():
    return sorted([p for p in RAG_DIR.iterdir() if p.is_file()])

def main():
    print("[HF-UPLOAD] Logging in...")
    login(HF_TOKEN)
    api = HfApi()

    print("[HF-UPLOAD] Getting remote file info...")
    try:
        remote_info = api.list_repo_files(repo_id=HF_REPO, repo_type="model", expand=True)
    except:
        remote_info = []

    # Map: path_in_repo -> size
    remote_sizes = {
        item.rfilename: item.size
        for item in remote_info
    }

    local_files = list_local()

    for p in local_files:
        path_in_repo = f"rag_storage/{p.name}"
        local_size = p.stat().st_size

        remote_size = remote_sizes.get(path_in_repo)

        # ----------------------------
        # FIX: Upload if NEW or CHANGED
        # ----------------------------
        if remote_size == local_size:
            print(f"[HF-UPLOAD] {p.name} unchanged → skipping")
            continue

        print(f"[HF-UPLOAD] Uploading {p.name} (local={local_size}, remote={remote_size})")
        try:
            upload_file(
                path_or_fileobj=str(p),
                path_in_repo=path_in_repo,
                repo_id=HF_REPO,
                repo_type="model",
                token=HF_TOKEN
            )
            print(f"[HF-UPLOAD] Uploaded {p.name}")
            time.sleep(0.2)
        except Exception as e:
            print("[HF-UPLOAD] Upload FAILED:", e)

    print("[HF-UPLOAD] Done.")

if __name__ == "__main__":
    main()
