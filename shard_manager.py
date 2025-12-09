import os

MAX_MB = 90

def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def get_active_rag_shard():
    files = sorted([f for f in os.listdir("rag_components") if "rag_final" in f])

    if not files:
        path = "rag_components/rag_final_1.jsonl"
        open(path, "w").close()
        return path

    latest = f"rag_components/{files[-1]}"
    if file_size_mb(latest) < MAX_MB:
        return latest

    new_id = len(files) + 1
    path = f"rag_components/rag_final_{new_id}.jsonl"
    open(path, "w").close()
    return path
