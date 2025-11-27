#!/usr/bin/env python3
"""
inference_search.py
- Load all shard_*.faiss and matching metadata_*.json from rag_storage/
- Query each shard, collect top_k, merge and return final top_k by distance (L2 small-is-better)
"""

import argparse
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

RAG_DIR = Path("rag_storage")
MODEL_NAME = "all-MiniLM-L6-v2"
DIM = 384

model = SentenceTransformer(MODEL_NAME)

def load_shards():
    shards = []
    for p in sorted(RAG_DIR.glob("shard_*.faiss")):
        try:
            idx = faiss.read_index(str(p))
            # derive metadata filename (match same number)
            num = p.stem.split("_")[1]
            meta_file = RAG_DIR / f"metadata_{num}.json"
            metas = []
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as f:
                    metas = json.load(f)
            print(f"[INFER] Loaded {p.name} (vectors={idx.ntotal})")
            shards.append((p.name, idx, metas))
        except Exception as e:
            print("[INFER] Failed to load", p, e)
    return shards

def search(query, top_k=10):
    emb = model.encode([query], convert_to_numpy=True).astype("float32")
    shards = load_shards()
    results = []
    for name, idx, metas in shards:
        if idx.ntotal == 0:
            continue
        D, I = idx.search(emb, top_k)
        for dist, i in zip(D[0], I[0]):
            if i < 0:
                continue
            meta = metas[i] if i < len(metas) else {}
            results.append({
                "shard": name,
                "distance": float(dist),
                "meta": meta
            })
    # smaller L2 distance is better
    results = sorted(results, key=lambda x: x["distance"])[:top_k]
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    out = search(args.query, top_k=args.top_k)
    print(json.dumps(out, indent=2, ensure_ascii=False))
