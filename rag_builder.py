import json
import uuid

TRAIN_FILE = "datasets/train.jsonl"
TEMP_RAG_FILE = "rag/rag_component_temp.jsonl"

def build_rag_component():
    with open(TRAIN_FILE, "r") as src, open(TEMP_RAG_FILE, "w") as dst:
        for line in src:
            data = json.loads(line)
            rag_entry = {
                "id": str(uuid.uuid4()),
                "text": f"Q: {data['question']} A: {data['answer']}",
                "tags": data["tags"]
            }
            dst.write(json.dumps(rag_entry) + "\n")

    return TEMP_RAG_FILE
