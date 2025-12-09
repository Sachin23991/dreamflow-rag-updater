from stackoverflow_scraper import fetch_stackoverflow_qa
from train_buffer_manager import append_to_train_buffer, clear_train_buffer
from rag_builder import build_rag_component
from hf_rag_uploader import merge_and_upload_rag
from git_auto_commit import git_commit_push

def run():
    print("🔎 Scraping StackOverflow Q&A...")
    qa = fetch_stackoverflow_qa()

    print("✍ Writing to train.jsonl...")
    append_to_train_buffer(qa)

    print("🧠 Converting train.jsonl → RAG...")
    temp_rag = build_rag_component()

    print("☁ Uploading RAG to HuggingFace (Versioned)...")
    merge_and_upload_rag(temp_rag)

    print("🧹 Clearing train.jsonl...")
    clear_train_buffer()

    print("📈 GitHub auto-commit...")
    git_commit_push()

    print("✅ Completed hourly RAG update.")

if __name__ == "__main__":
    run()
