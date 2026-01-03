from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="Qwen/Qwen3-4B-Instruct-2507",
#     cache_dir="./models"
# )

local_path = snapshot_download(
    repo_id="routellm/bert_gpt4_augmented",
    cache_dir="./models"  # 你希望存放本地模型的目录
)

print("Router 模型本地路径：", local_path)
