from huggingface_hub import hf_hub_download
from pathlib import Path

hf_hub_download(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        local_dir=Path(__file__).parent.resolve(),
    )