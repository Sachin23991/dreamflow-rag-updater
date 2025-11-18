from huggingface_hub import HfApi
import os

def upload_rag():
    print("🚀 Starting upload to HuggingFace...")
    
    api = HfApi()

    api.upload_folder(
        repo_id="Sachin21112004/carrerflow-ai",
        folder_path="rag_storage",
        path_in_repo="rag_storage",
        token=os.environ["HF_TOKEN"],
        commit_message="Daily updated RAG knowledge"
    )

    print("✔ Upload complete")

if __name__ == "__main__":
    upload_rag()
