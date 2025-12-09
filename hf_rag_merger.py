import shutil
from huggingface_hub import hf_hub_download, HfApi
from shard_manager import get_active_rag_shard

HF_REPO = "Sachin21112004/final-rag-dataset"

def merge_and_upload_rag(temp_rag_file):
    api = HfApi()
    active_shard = get_active_rag_shard()

    if active_shard:
        try:
            hf_file = hf_hub_download(
                repo_id=HF_REPO,
                filename=active_shard.split("/")[-1],
                repo_type="dataset"
            )
            shutil.copy(hf_file, active_shard)
        except:
            pass

    with open(temp_rag_file, "r") as src, open(active_shard, "a") as dst:
        dst.write(src.read())

    api.upload_folder(
        folder_path="rag_components",
        repo_id=HF_REPO,
        repo_type="dataset"
    )
