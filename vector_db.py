from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def ingest_into_vector_db(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        documents = f.read().split("\n\n")

    embeddings = MODEL.encode(documents)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    faiss.write_index(index, "rag_index.faiss")

    print("✅ RAG Vector Database Updated")
