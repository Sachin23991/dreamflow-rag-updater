#!/usr/bin/env python3
"""
sharded_rag_update.py
- Reads local TRAIN_JSON (train.jsonl) containing new items (scraper output, downloaded earlier).
- Optionally downloads manifest + shards from HF (if HF_REPO & HF_TOKEN provided).
- Appends new vectors to the last shard until rollover, otherwise creates new shards.
- Writes shard_NNNN.faiss and metadata_NNNN.json and updates manifest.json in rag_storage/.
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
TRAIN_JSON = "train.jsonl"            # Created by your scraper pipeline (downloaded by workflow)
RAG_DIR = Path("rag_storage")
MODEL_NAME = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
DIM = int(os.environ.get("EMBED_DIM", "384"))   # 384 for MiniLM
ROLLOVER_MB = int(os.environ.get("ROLLOVER_MB", "300"))
HF_REPO = os.environ.get("HF_REPO")   # e.g. "username/model-repo"
HF_TOKEN = os.environ.get("HF_TOKEN")
# -----------------------

RAG_DIR.mkdir(exist_ok=True)
MANIFEST = RAG_DIR / "manifest.json"

model = SentenceTransformer(MODEL_NAME)


def download_manifest_and_shards_from_hf():
    """Attempt to download manifest.json and shards from HF repo into rag_storage/"""
    if not HF_REPO or not HF_TOKEN:
        print("[HF] HF_REPO or HF_TOKEN not set — skipping HF download.")
        return
    print("[HF] Logging in and trying to download manifest + shards...")
    try:
        hf_login(HF_TOKEN)
    except Exception as e:
        print("[HF] login failed:", e)
        return
    api = HfApi()
    try:
        # download manifest.json if present
        manifest_file = None
        try:
            manifest_file = hf_hub_download(repo_id=HF_REPO, filename="rag_storage/manifest.json",
                                            local_dir=str(RAG_DIR), token=HF_TOKEN, local_dir_use_symlinks=False)
            print("[HF] Manifest downloaded:", manifest_file)
        except Exception:
            print("[HF] No manifest found in remote repo (first run?) - continuing local only")

        # if manifest downloaded, read shards list and download each file listed
        if manifest_file:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for entry in manifest.get("shards", []):
                for fname in (entry.get("shard_file"), entry.get("meta_file")):
                    if not fname:
                        continue
                    out = RAG_DIR / Path(fname).name
                    if out.exists():
                        print(f"[HF] {out.name} already exists locally; skipping.")
                        continue
                    try:
                        hf_hub_download(repo_id=HF_REPO, filename=f"rag_storage/{fname}",
                                        local_dir=str(RAG_DIR), token=HF_TOKEN, local_dir_use_symlinks=False)
                        print(f"[HF] Downloaded {fname}")
                    except Exception as e:
                        print(f"[HF] Could not download {fname}:", e)
    except Exception as e:
        print("[HF] Error downloading from HF:", e)


def load_local_manifest():
    if MANIFEST.exists():
        with open(MANIFEST, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"shards": [], "total_vectors": 0}


def save_local_manifest(manifest: dict):
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def read_train_json(path=TRAIN_JSON):
    items = []
    if not os.path.exists(path):
        print("[RAG] No train file found at", path)
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
                print("[RAG] Skipping invalid line:", e)
    return items


def estimate_vectors_per_shard(rollover_mb, dim):
    bytes_per_vector = dim * 4
    return max(1, int((rollover_mb * 1024 * 1024) / bytes_per_vector))


def next_shard_id(manifest):
    if not manifest["shards"]:
        return 1
    return manifest["shards"][-1]["id"] + 1


def load_last_shard(manifest):
    """Return (index_obj or None, metadata_list or [] , shard_id or None)"""
    if not manifest["shards"]:
        return None, [], None
    last = manifest["shards"][-1]
    fname = Path(last["shard_file"]).name
    meta_fname = Path(last.get("meta_file", "")).name
    shard_path = RAG_DIR / fname
    meta_path = RAG_DIR / meta_fname
    if shard_path.exists():
        try:
            idx = faiss.read_index(str(shard_path))
            metas = []
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    metas = json.load(f)
            return idx, metas, last["id"]
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


def main():
    # Optionally get existing shards/manifest from HF
    download_manifest_and_shards_from_hf()

    manifest = load_local_manifest()
    print("[RAG] Manifest has", len(manifest.get("shards", [])), "shards, total_vectors:",
          manifest.get("total_vectors", 0))

    max_vectors = estimate_vectors_per_shard(ROLLOVER_MB, DIM)
    print(f"[RAG] Rollover approx vectors per shard: {max_vectors}")

    # Try load last shard
    last_index, last_meta, last_id = load_last_shard(manifest)
    if last_index is not None:
        current_index = last_index
        current_meta = last_meta
        current_vectors = current_index.ntotal
        current_shard_id = last_id
        print(f"[RAG] Loaded last shard id {current_shard_id} with {current_vectors} vectors")
    else:
        current_index = faiss.IndexFlatL2(DIM)
        current_meta = []
        current_vectors = 0
        current_shard_id = next_shard_id(manifest) if manifest["shards"] else 1
        print(f"[RAG] Starting new shard id {current_shard_id}")

    # Read new items (train.jsonl must be present)
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
        # Roll shard if would overflow
        if current_vectors + 1 > max_vectors:
            # write current shard and update manifest
            fname, meta_fname = write_shard(current_index, current_meta, current_shard_id)
            now = datetime.utcnow().isoformat() + "Z"
            # if last manifest shard has same id, update, else append
            if manifest["shards"] and manifest["shards"][-1]["id"] == current_shard_id:
                manifest["total_vectors"] -= manifest["shards"][-1]["vectors"]
                manifest["shards"][-1].update({
                    "shard_file": fname,
                    "meta_file": meta_fname,
                    "vectors": len(current_meta),
                    "updated_at": now
                })
                manifest["total_vectors"] += len(current_meta)
            else:
                manifest["shards"].append({
                    "id": current_shard_id,
                    "shard_file": fname,
                    "meta_file": meta_fname,
                    "vectors": len(current_meta),
                    "created_at": now
                })
                manifest["total_vectors"] = manifest.get("total_vectors", 0) + len(current_meta)

            # start new shard
            current_shard_id += 1
            current_index = faiss.IndexFlatL2(DIM)
            current_meta = []
            current_vectors = 0
            print(f"[RAG] Rolled over to new shard id {current_shard_id}")

        # add vector + metadata
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

    # Write the last (current) shard to disk and update manifest
    if current_meta:
        fname, meta_fname = write_shard(current_index, current_meta, current_shard_id)
        now = datetime.utcnow().isoformat() + "Z"
        if manifest["shards"] and manifest["shards"][-1]["id"] == current_shard_id:
            manifest["total_vectors"] -= manifest["shards"][-1]["vectors"]
            manifest["shards"][-1].update({
                "shard_file": fname,
                "meta_file": meta_fname,
                "vectors": len(current_meta),
                "updated_at": now
            })
            manifest["total_vectors"] += len(current_meta)
        else:
            manifest["shards"].append({
                "id": current_shard_id,
                "shard_file": fname,
                "meta_file": meta_fname,
                "vectors": len(current_meta),
                "created_at": now
            })
            manifest["total_vectors"] = manifest.get("total_vectors", 0) + len(current_meta)

    manifest["last_updated"] = datetime.utcnow().isoformat() + "Z"
    save_local_manifest(manifest)
    print(f"[RAG] Finished. Added {added} vectors. Manifest now has {len(manifest['shards'])} shards, total_vectors={manifest['total_vectors']}.")
    print("[RAG] To upload results to HF, run sharded_upload_to_hf.py or let the workflow do it.")

if __name__ == "__main__":
    main()
