#!/usr/bin/env python3
"""
sharded_rag_update.py
- Downloads existing shards/manifest from HF
- Loads latest shard
- Checks real file size of shard (MB)
- If shard >= 90 MB → create new shard
- Otherwise append to the same shard
- Updates manifest
"""

import os
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import hf_hub_download, HfApi, login as hf_login

# ------- CONFIG -------
TRAIN_JSON = "train.jsonl"
RAG_DIR = Path("rag_storage")
MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
DIM = int(os.environ.get("EMBED_DIM", "384"))
MAX_SHARD_MB = 90                     # REAL rollover limit (90MB)
HF_REPO = os.environ.get("HF_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")
# ----------------------

RAG_DIR.mkdir(exist_ok=True)
MANIFEST = RAG_DIR / "manifest.json"

model = SentenceTransformer(MODEL_NAME)


# --------------------- UTILS ---------------------

def get_shard_size_mb(path: Path):
    """Return actual file size in MB (0 if not exists)."""
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def download_manifest_and_shards_from_hf():
    """Download manifest + shards from HF if available."""
    if not HF_REPO or not HF_TOKEN:
        print("[HF] HF_REPO or HF_TOKEN not set — skipping HF download.")
        return

    print("[HF] Logging in...")
    try:
        hf_login(HF_TOKEN)
    except Exception as e:
        print("[HF] login failed:", e)
        return

    api = HfApi()

    # Try download manifest
    manifest_file = None
    try:
        manifest_file = hf_hub_download(
            repo_id=HF_REPO,
            filename="rag_storage/manifest.json",
            local_dir=str(RAG_DIR),
            token=HF_TOKEN,
            local_dir_use_symlinks=False
        )
        print("[HF] Manifest downloaded:", manifest_file)
    except Exception:
        print("[HF] No manifest found (first run?)")

    # If manifest exists → download shards
    if manifest_file:
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for entry in manifest.get("shards", []):
            for fname in (entry.get("shard_file"), entry.get("meta_file")):
                if not fname:
                    continue
                out = RAG_DIR / fname
                if out.exists():
                    continue
                try:
                    hf_hub_download(
                        repo_id=HF_REPO,
                        filename=f"rag_storage/{fname}",
                        local_dir=str(RAG_DIR),
                        token=HF_TOKEN,
                        local_dir_use_symlinks=False
                    )
                    print(f"[HF] Downloaded {fname}")
                except Exception as e:
                    print(f"[HF] Failed to download {fname}:", e)


def load_local_manifest():
    if MANIFEST.exists():
        with open(MANIFEST, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"shards": [], "total_vectors": 0}


def save_local_manifest(m):
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)


def read_train_json(path=TRAIN_JSON):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "text" not in obj:
                    obj["text"] = obj.get("content") or obj.get("body") or ""
                items.append(obj)
            except:
                pass
    return items


def load_last_shard(manifest):
    if not manifest["shards"]:
        return None, [], None

    last = manifest["shards"][-1]
    sid = last["id"]
    shard_path = RAG_DIR / last["shard_file"]
    meta_path = RAG_DIR / last["meta_file"]

    if shard_path.exists():
        idx = faiss.read_index(str(shard_path))
        metas = []
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metas = json.load(f)
        return idx, metas, sid

    return None, [], None


def write_shard(index_obj, meta_list, shard_id):
    fname = f"shard_{shard_id:04d}.faiss"
    meta_fname = f"metadata_{shard_id:04d}.json"

    out_index = RAG_DIR / fname
    out_meta = RAG_DIR / meta_fname

    faiss.write_index(index_obj, str(out_index))
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)

    print(f"[RAG] Saved shard {fname} ({len(meta_list)} vectors)")
    return fname, meta_fname


# --------------------- MAIN ---------------------

def main():
    download_manifest_and_shards_from_hf()

    manifest = load_local_manifest()
    print(f"[RAG] Manifest shards={len(manifest['shards'])}, total_vectors={manifest['total_vectors']}")

    last_index, last_meta, last_id = load_last_shard(manifest)

    # ------------------------------
    #   SIZE-BASED SHARD ROLLOVER
    # ------------------------------

    if last_index is not None:
        shard_path = RAG_DIR / f"shard_{last_id:04d}.faiss"
        size_mb = get_shard_size_mb(shard_path)

        if size_mb >= MAX_SHARD_MB:
            # Create a NEW shard
            current_shard_id = last_id + 1
            current_index = faiss.IndexFlatL2(DIM)
            current_meta = []
            current_vectors = 0
            print(f"[RAG] Last shard is {size_mb:.2f}MB ≥ {MAX_SHARD_MB}MB → NEW SHARD {current_shard_id}")

        else:
            # Append to existing shard
            current_shard_id = last_id
            current_index = last_index
            current_meta = last_meta
            current_vectors = current_index.ntotal
            print(f"[RAG] Appending to shard {current_shard_id} (size={size_mb:.2f}MB)")

    else:
        # No shards at all → start #1
        current_shard_id = 1
        current_index = faiss.IndexFlatL2(DIM)
        current_meta = []
        current_vectors = 0
        print(f"[RAG] No previous shards → starting shard 1")

    # -------- Encode training items --------

    items = read_train_json()
    if not items:
        print("[RAG] No new items")
        return

    texts = [x["text"] for x in items]
    print(f"[RAG] Encoding {len(texts)} items...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    added = 0

    for emb_vec, item in zip(embeddings, items):
        current_index.add(emb_vec.reshape(1, -1))
        meta = {
            "text": item.get("text", ""),
            "source": item.get("source"),
            "id": item.get("id"),
            "created_at": item.get("created_at") or datetime.utcnow().isoformat() + "Z",
            "extra": {k: v for k, v in item.items() if k not in ("text", "source", "id", "created_at")}
        }
        current_meta.append(meta)
        current_vectors += 1
        added += 1

    # -------- Save updated shard --------

    fname, meta_fname = write_shard(current_index, current_meta, current_shard_id)

    # -------- Update manifest --------

    now = datetime.utcnow().isoformat() + "Z"
    shard_rec = {
        "id": current_shard_id,
        "shard_file": fname,
        "meta_file": meta_fname,
        "vectors": len(current_meta),
        "updated_at": now
    }

    if manifest["shards"] and manifest["shards"][-1]["id"] == current_shard_id:
        manifest["shards"][-1] = shard_rec
    else:
        manifest["shards"].append(shard_rec)

    manifest["total_vectors"] = sum(x["vectors"] for x in manifest["shards"])
    manifest["last_updated"] = now

    save_local_manifest(manifest)

    print(f"[RAG] Added {added} vectors → shard {current_shard_id}")
    print(f"[RAG] Manifest shards={len(manifest['shards'])}, total={manifest['total_vectors']}")
    print("[RAG] Ready for upload.")

if __name__ == "__main__":
    main()
