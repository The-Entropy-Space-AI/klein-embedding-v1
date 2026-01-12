from huggingface_hub import HfApi

api = HfApi()

# Change these
repo_id = "theentropyspaceai/klein-embedding"
folder_path = "data/tokenizer"

api.create_repo(repo_id=repo_id, exist_ok=True)

api.upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    path_in_repo="tokenizer"
)

print("Uploaded successfully!")