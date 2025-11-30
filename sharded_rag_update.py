#!/usr/bin/env python3
"""
sharded_rag_update.py (patched)
- Downloads existing manifest + shards from HF (correct filenames)
- Loads latest shard (if any)
- Checks real file size of the last shard (MB)
- If shard >= MAX_SHARD_MB -> create new shard
- Otherwise append to the same shard
- Writes updated shard files and manifest into rag_storage/
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
MAX_SHARD_MB = int(os.environ.get("MAX_SHARD_MB", "90"))   # rollover limit in MB
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
    """
    Download manifest.json and shard/meta files from HF repo into rag_storage/.
    IMPORTANT: We pass simple filenames (manifest.json, shard_0001.faiss, ...)
    because files on the repo are stored under rag_storage/ on HF - hf_hub_download
    returns them into the local_dir given (RAG_DIR).
    """
    if not HF_REPO or not HF_TOKEN:
        print("[HF] HF_REPO or HF_TOKEN not set — skipping HF download.")
        return

    print("[HF] Logging in to HuggingFace hub...")
    try:
        hf_login(HF_TOKEN)
    except Exception as e:
        print("[HF] login failed:", e)
        return

    # Try to download manifest.json (filename on the remote: 'manifest.json' inside rag_storage folder)
    manifest_file = None
    try:
        manifest_file = hf_hub_download(
            repo_id=HF_REPO,
            filename="manifest.json",        # CORRECT: only filename
            local_dir=str(RAG_DIR),
            token=HF_TOKEN
        )
        print("[HF] Manifest downloaded to:", manifest_file)
    except Exception as e:
        print("[HF] No manifest found in remote repo or download failed (first run?):", e)
        manifest_file = None

    # If manifest exists locally now, parse it and download files listed in it.
    if manifest_file:
        try:
            with open(RAG_DIR / "manifest.json", "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            print("[HF] Failed to read downloaded manifest.json:", e)
            return

        for entry in manifest.get("shards", []):
            for fname in (entry.get("shard_file"), entry.get("meta_file")):
                if not fname:
                    continue
                out = RAG_DIR / fname
                if out.exists():
                    print(f"[HF] Local {out.name} already exists; skipping download.")
                    continue
                try:
                    # IMPORTANT: pass only the filename so hf_hub_download finds it under rag_storage/ in the repo
                    downloaded = hf_hub_download(
                        repo_id=HF_REPO,
                        filename=fname,      # CORRECT: only filename
                        local_dir=str(RAG_DIR),
                        token=HF_TOKEN
                    )
                    print(f"[HF] Downloaded {fname} -> {downloaded}")
                except Exception as e:
                    print(f"[HF] Could not download {fname}:", e)


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
        print(f"[RAG] No train file found at {path}")
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
            except Exception as e:
                print("[RAG] Skipping invalid train.jsonl line:", e)
    return items


def load_last_shard(manifest):
    """
    Return (index_obj or None, metadata_list or [], shard_id or None)
    """
    if not manifest.get("shards"):
        return None, [], None

    last = manifest["shards"][-1]
    sid = last.get("id")
    shard_fname = last.get("shard_file")
    meta_fname = last.get("meta_file")
    if not shard_fname:
        return None, [], None

    shard_path = RAG_DIR / shard_fname
    meta_path = RAG_DIR / meta_fname if meta_fname else None

    if shard_path.exists():
        try:
            idx = faiss.read_index(str(shard_path))
            metas = []
            if meta_path and meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metas = json.load(f)
            return idx, metas, sid
        except Exception as e:
            print("[RAG] Failed to load last shard:", e)
            return None, [], None
    return None, [], None


def write_shard(index_obj, metadata_list, shard_id):
    fname = f"shard_{shard_id:04d}.faiss"
    meta_fname = f"metadata_{shard_id:04d}.json"
    out_idx = RAG_DIR / fname
    out_meta = RAG_DIR / meta_fname
    faiss.write_index(index_obj, str(out_idx))
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"[RAG] Wrote shard: {out_idx.name} ({len(metadata_list)} vectors) and {out_meta.name}")
    return out_idx.name, out_meta.name


# --------------------- MAIN ---------------------

def main():
    # Download manifest and shard files from HF into rag_storage/
    download_manifest_and_shards_from_hf()

    manifest = load_local_manifest()
    print("[RAG] Manifest shards=", len(manifest.get("shards", [])), "total_vectors=", manifest.get("total_vectors", 0))

    # Load last shard (if manifest points to any)
    last_index, last_meta, last_id = load_last_shard(manifest)

    if last_index is not None:
        shard_path = RAG_DIR / f"shard_{last_id:04d}.faiss"
        size_mb = get_shard_size_mb(shard_path)

        if size_mb >= MAX_SHARD_MB:
            # rollover: create new shard id
            current_shard_id = last_id + 1
            current_index = faiss.IndexFlatL2(DIM)
            current_meta = []
            current_vectors = 0
            print(f"[RAG] Last shard size {size_mb:.2f}MB >= {MAX_SHARD_MB}MB -> creating shard {current_shard_id}")
        else:
            # append to existing last shard
            current_shard_id = last_id
            current_index = last_index
            current_meta = last_meta
            current_vectors = current_index.ntotal
            print(f"[RAG] Appending to existing shard {current_shard_id} (size={size_mb:.2f}MB, vectors={current_vectors})")
    else:
        # No valid last shard found -> create shard 1
        current_shard_id = 1
        current_index = faiss.IndexFlatL2(DIM)
        current_meta = []
        current_vectors = 0
        print("[RAG] No previous shards -> starting shard 1")

    # Read new training items
    new_items = read_train_json()
    if not new_items:
        print("[RAG] No new items. Exiting.")
        return

    texts = [it.get("text", "") for it in new_items]
    print(f"[RAG] Encoding {len(texts)} new items with model {MODEL_NAME} ...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype("float32")

    added = 0
    for emb_vec, item in zip(embeddings, new_items):
        # Add vector + metadata
        current_index.add(emb_vec.reshape(1, -1))
        metadata_entry = {
            "text": item.get("text", ""),
            "source": item.get("source"),
            "id": item.get("id"),
            "created_at": item.get("created_at") or datetime.utcnow().isoformat() + "Z",
            "extra": {k: v for k, v in item.items() if k not in ("text", "source", "id", "created_at")}
        }
        current_meta.append(metadata_entry)
        current_vectors += 1
        added += 1

    # Write the updated (or new) shard to disk
    fname, meta_fname = write_shard(current_index, current_meta, current_shard_id)

    # Update manifest: either replace last entry or append a new one
    now = datetime.utcnow().isoformat() + "Z"
    shard_rec = {
        "id": current_shard_id,
        "shard_file": fname,
        "meta_file": meta_fname,
        "vectors": len(current_meta),
        "updated_at": now
    }

    if manifest.get("shards") and manifest["shards"][-1].get("id") == current_shard_id:
        manifest["shards"][-1] = shard_rec
    else:
        manifest.setdefault("shards", []).append(shard_rec)

    manifest["total_vectors"] = sum(x.get("vectors", 0) for x in manifest.get("shards", []))
    manifest["last_updated"] = now

    save_local_manifest(manifest)

    print(f"[RAG] Finished. Added {added} vectors. Manifest shards={len(manifest['shards'])}, total_vectors={manifest['total_vectors']}.")
    print("[RAG] To upload results to HF, run sharded_upload_to_hf.py or let the workflow do it.")


if __name__ == "__main__":
    main()
