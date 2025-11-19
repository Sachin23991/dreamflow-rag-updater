import json
import pandas as pd
import os
import faiss
from sentence_transformers import SentenceTransformer
import hashlib

os.makedirs("rag_storage", exist_ok=True)

# ---------------------------------------------------
# Helper: Hash chunks so we don't re-add duplicates
# ---------------------------------------------------
def hash_text(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

# ---------------------------------------------------
# 1. Load NEW records from train.jsonl
# ---------------------------------------------------
records = []
with open("train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df_new = pd.DataFrame(records)
df_new["chunk"] = df_new["prompt"].fillna("") + "\n\n" + df_new["completion"].fillna("")
df_new["hash"] = df_new["chunk"].apply(hash_text)

print(f"📥 Loaded {len(df_new)} new rows from train.jsonl")

# ---------------------------------------------------
# 2. Load old metadata if exists
# ---------------------------------------------------
meta_path = "rag_storage/metadata.json"

if os.path.exists(meta_path):
    df_old = pd.read_json(meta_path)
    print(f"📦 Loaded existing metadata with {len(df_old)} entries")
else:
    df_old = pd.DataFrame(columns=["chunk", "hash"])
    print("📦 No previous metadata found. Starting fresh.")

# ---------------------------------------------------
# 3. Filter only new unique chunks
# ---------------------------------------------------
existing_hashes = set(df_old["hash"].tolist())
df_new_unique = df_new[~df_new["hash"].isin(existing_hashes)]

print(f"✨ Unique new chunks to embed: {len(df_new_unique)}")

if len(df_new_unique) == 0:
    print("🚀 Nothing new to add. RAG is already up to date!")
    exit(0)

# ---------------------------------------------------
# 4. Embed ONLY new unique chunks
# ---------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
new_embeddings = embedder.encode(
    df_new_unique["chunk"].tolist(),
    convert_to_numpy=True,
    batch_size=32,
    show_progress_bar=True
)

# ---------------------------------------------------
# 5. Load or create FAISS index
# ---------------------------------------------------
index_path = "rag_storage/index.faiss"

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    print(f"📚 Loaded FAISS index with {index.ntotal} vectors")
else:
    dim = new_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    print("📚 Creating new FAISS index")

# ---------------------------------------------------
# 6. Add new embeddings to FAISS
# ---------------------------------------------------
index.add(new_embeddings.astype("float32"))
print(f"➕ Added {len(new_embeddings)} new vectors")
print(f"🔢 Total vectors now: {index.ntotal}")

# ---------------------------------------------------
# 7. Save updated metadata (append mode)
# ---------------------------------------------------
df_all = pd.concat([df_old, df_new_unique], ignore_index=True)
df_all.to_json(meta_path, indent=2)

# ---------------------------------------------------
# 8. Save updated FAISS index
# ---------------------------------------------------
faiss.write_index(index, index_path)

print("\n🎉 RAG updated successfully — APPEND MODE enabled!")
