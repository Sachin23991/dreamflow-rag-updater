#!/usr/bin/env python3
import json, os
HF_FILE = "train.jsonl"
NEW_FILE = "pipeline/train.jsonl"
OUT_FILE = "merged_train.jsonl"

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

old = read_jsonl(HF_FILE)
new = read_jsonl(NEW_FILE)
print("old:", len(old), "new:", len(new))
# avoid exact duplicates by text
seen = {json.dumps(item, sort_keys=True) for item in old}
combined = old[:]
for it in new:
    key = json.dumps(it, sort_keys=True)
    if key in seen:
        continue
    seen.add(key)
    combined.append(it)
print("final:", len(combined))
write_jsonl(OUT_FILE, combined)
print("Wrote", OUT_FILE)
