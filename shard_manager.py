import os

MAX_MB = 90
RAG_FOLDER = "rag"

def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def get_active_rag_version():
    files = sorted([f for f in os.listdir(RAG_FOLDER) if f.startswith("rag_version_")])

    # If no version exists, create version 1
    if not files:
        path = f"{RAG_FOLDER}/rag_version_1.jsonl"
        open(path, "w").close()
        return path

    latest = f"{RAG_FOLDER}/{files[-1]}"

    # If the latest is under limit, use it
    if file_size_mb(latest) < MAX_MB:
        return latest

    # Else create new version file
    new_version = len(files) + 1
    path = f"{RAG_FOLDER}/rag_version_{new_version}.jsonl"
    open(path, "w").close()
    return path
