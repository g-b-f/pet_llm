import huggingface_hub
from pathlib import Path
import json
from pathlib import Path

model_path = token_path = Path(__file__).parent
token_path = model_path / "hf_token.json"

mapping = {
 "smollm2-1.7b-q8_0": "NikolayKozloff/SmolLM2-1.7B-Q8_0-GGUF",
 "qwen2.5-1.5b-instruct-q4_k_m": "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
}

def get_model(model:str):
    filename = model + ".gguf"
    filepath = Path(__file__).parent / filename
    if filepath.exists():
        return filepath
        
    with open(token_path) as f:
        token = json.load(f)["download"]
    
    huggingface_hub.hf_hub_download(
        repo_id=mapping[model],
        filename=filename,
        local_dir=model_path.resolve(),
        token=token
    )
    if not filepath.exists():
        raise RuntimeError("error downloading: maybe in the wrong place?")
    return filepath