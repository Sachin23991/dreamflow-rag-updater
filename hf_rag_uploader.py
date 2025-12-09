import shutil
from huggingface_hub import HfApi, hf_hub_download
from shard_manager import get_active_rag_version

HF_REPO = "Sachin21112004/distilbart-news-summarizer"
HF_FOLDER = "rag"   # the folder you specified

def merge_and_upload_rag(temp_rag):
    api = HfApi()
    active_file = get_active_rag_version()
    filename = active_file.split("/")[-1]

    # Try loading old version from HF to local
    try:
        old_file = hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{HF_FOLDER}/{filename}",
            repo_type="model"
        )
        shutil.copy(old_file, active_file)
    except:
        pass

    # Append new data
    with open(temp_rag, "r") as src, open(active_file, "a") as dst:
        dst.write(src.read())

    # Upload updated rag folder to HF
    api.upload_folder(
        folder_path="rag",
        repo_id=HF_REPO,
        repo_type="model",
        path_in_repo="rag"
    )
