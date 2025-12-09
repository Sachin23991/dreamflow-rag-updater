import json
import os

CLEAN_DATA_PATH = "rag_ready_data.txt"

def convert_jsonl_to_text():
    all_text = []

    for file in os.listdir("datasets"):
        if file.endswith(".jsonl"):
            with open(f"datasets/{file}", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    text = f"""
Title: {data['title']}
Tags: {', '.join(data['tags'])}
Link: {data['link']}
"""
                    all_text.append(text.strip())

    with open(CLEAN_DATA_PATH, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

    return CLEAN_DATA_PATH
