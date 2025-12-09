import json
import os
import uuid

RAG_TEMP_FILE = "rag_components/rag_update_temp.jsonl"
TRAIN_FILE = "datasets/train.jsonl"

def build_rag_component():
    with open(TRAIN_FILE, "r", encoding="utf-8") as src, \
         open(RAG_TEMP_FILE, "w", encoding="utf-8") as out:

        for line in src:
            data = json.loads(line)
            rag_entry = {
                "id": str(uuid.uuid4()),
                "text": f"Q: {data['question']} A: {data['answer']}",
                "tags": data["tags"]
            }
            out.write(json.dumps(rag_entry) + "\n")

    return RAG_TEMP_FILE
