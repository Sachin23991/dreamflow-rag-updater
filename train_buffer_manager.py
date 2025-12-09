import json

TRAIN_FILE = "datasets/train.jsonl"

def append_to_train_buffer(data):
    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def clear_train_buffer():
    open(TRAIN_FILE, "w").close()
