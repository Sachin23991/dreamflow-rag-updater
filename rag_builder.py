import json
import uuid
import os

TRAIN_FILE = "datasets/train.jsonl"
RAG_FOLDER = "rag"
TEMP_RAG_FILE = "rag/rag_component_temp.jsonl"


def ensure_rag_folder_exists():
    """Ensure the rag folder exists before writing."""
    os.makedirs(RAG_FOLDER, exist_ok=True)


def build_rag_component():
    # ✅ Make sure rag/ folder exists
    ensure_rag_folder_exists()

    with open(TRAIN_FILE, "r", encoding="utf-8") as src, \
         open(TEMP_RAG_FILE, "w", encoding="utf-8") as dst:

        for line in src:
            data = json.loads(line)

            rag_entry = {
                "id": str(uuid.uuid4()),
                "text": f"Q: {data['question']} A: {data['answer']}",
                "tags": data["tags"]
            }

            dst.write(json.dumps(rag_entry) + "\n")

    return TEMP_RAG_FILE
