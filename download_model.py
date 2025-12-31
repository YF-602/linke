from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    cache_dir="./models"
)
