from huggingface_hub import snapshot_download

# This will download everything under repo "meta-llama/Llama-3.2-1B"
# into your local "models/Llama-3.2-1B" folder.
snapshot_download(
    repo_id="meta-llama/Llama-3.2-1B",
    repo_type="model",
    local_dir="models/Llama-3.2-1B",
    local_dir_use_symlinks=False  # ensure full copy
)
print("Download complete.")
