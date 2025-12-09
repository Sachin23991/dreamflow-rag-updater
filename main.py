from stackoverflow_scraper import fetch_stackoverflow_data
from dataset_manager import append_to_jsonl
from hf_uploader import upload_dataset
from rag_converter import convert_jsonl_to_text
from vector_db import ingest_into_vector_db
from git_auto_commit import git_commit_push

def run_pipeline():
    print("🔄 Fetching Stack Overflow data...")
    data = fetch_stackoverflow_data()

    print("📦 Updating JSONL dataset...")
    append_to_jsonl(data)

    print("☁ Uploading dataset to Hugging Face...")
    upload_dataset()

    print("🔄 Converting dataset for RAG...")
    clean_text = convert_jsonl_to_text()

    print("🧠 Updating Vector Database...")
    ingest_into_vector_db(clean_text)

    print("📈 Auto committing to GitHub...")
    git_commit_push()

    print("✅ FULL RAG PIPELINE EXECUTED SUCCESSFULLY")

if __name__ == "__main__":
    run_pipeline()
