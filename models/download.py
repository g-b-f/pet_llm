import huggingface_hub
from pathlib import Path
import json
from pathlib import Path

model_path = token_path = Path(__file__).parent
token_path = model_path / "hf_token.json"

with open(token_path) as f:
    token = json.load(f)["download"]

def download(repo_id:str, filename:str):
    huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=repo_id,
        local_dir=model_path.resolve(),
        token=token
    )

download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
)

download(
    repo_id="NikolayKozloff/SmolLM2-1.7B-Q8_0-GGUF",
    filename="smollm2-1.7b-q8_0.gguf",
)
