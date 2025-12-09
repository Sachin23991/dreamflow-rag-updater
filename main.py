from stackoverflow_scraper import fetch_stackoverflow_qa
from train_buffer_manager import append_to_train_buffer, clear_train_buffer
from rag_component_builder import build_rag_component
from hf_rag_merger import merge_and_upload_rag
from git_auto_commit import git_commit_push

def run_pipeline():
    print("🔄 Fetching Stack Overflow Q&A...")
    qa_data = fetch_stackoverflow_qa()

    print("🧾 Writing to train.jsonl...")
    append_to_train_buffer(qa_data)

    print("🧠 Building RAG component...")
    rag_temp = build_rag_component()

    print("☁ Uploading & appending to Hugging Face RAG...")
    merge_and_upload_rag(rag_temp)

    print("🧹 Clearing train.jsonl buffer...")
    clear_train_buffer()

    print("📈 Committing to GitHub...")
    git_commit_push()

    print("✅ HOURLY RAG PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    run_pipeline()
