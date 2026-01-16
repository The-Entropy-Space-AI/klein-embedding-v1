from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="data/processed/samanantar_train.jsonl",
    path_in_repo="samanantar_train.jsonl",  # Name inside the repo
    repo_id="the-entropy-space-ai/klein-embedding-data",
    repo_type="dataset"
)