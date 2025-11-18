import json
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

# Make sure folder exists
os.makedirs("rag_storage", exist_ok=True)

print("Reading train.jsonl...")
records = []
with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

if not records:
    raise SystemExit("❌ train.jsonl is empty or missing")

df = pd.DataFrame(records)

# Extract every piece of text from your dataset
df["chunk"] = df["prompt"].fillna("") + "\n\n" + df["completion"].fillna("")

print(f"✔ Extracted {len(df)} knowledge chunks")

print("Loading MiniLM embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding chunks...")
embeddings = embedder.encode(
    df["chunk"].tolist(),
    convert_to_numpy=True,
    batch_size=32,
    show_progress_bar=True
)

print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings.astype("float32"))

faiss.write_index(index, "rag_storage/index.faiss")
df[["chunk"]].to_json("rag_storage/metadata.json", indent=2)

print("🎉 RAG knowledge built and saved in rag_storage/")
