import json
import os

TRAIN_FILE = "datasets/train.jsonl"

def ensure_train_file_exists():
    """Ensure both the folder and the file exist."""
    os.makedirs("datasets", exist_ok=True)  # Create folder if missing
    if not os.path.exists(TRAIN_FILE):
        open(TRAIN_FILE, "w").close()       # Create empty file


def append_to_train_buffer(data):
    ensure_train_file_exists()

    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def clear_train_buffer():
    ensure_train_file_exists()

    # Overwrite with an empty file
    open(TRAIN_FILE, "w").close()
