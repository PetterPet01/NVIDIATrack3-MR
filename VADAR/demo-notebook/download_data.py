from huggingface_hub import snapshot_download

snapshot_download(
  repo_id="nvidia/PhysicalAI-Spatial-Intelligence-Warehouse",
  repo_type="dataset",
  allow_patterns=["train_sample/**"],  # Only download this folder
  local_dir="PhysicalAI_Dataset"       # Or any path you prefer
)
