from huggingface_hub import snapshot_download
import os

# Save model in the medrag directory (same level as src)
medrag_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
model_dir = os.path.join(medrag_dir, 'deepseek_8b')
snapshot_download(
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    revision="main",
    local_dir=model_dir,
    local_dir_use_symlinks=False
)