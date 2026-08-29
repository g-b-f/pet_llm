from pathlib import Path
import json
from enum import Enum
import huggingface_hub

model_path = token_path = Path(__file__).parent
token_path = model_path / "hf_token.json"

class Model(Enum):
    """Model to load into the brain. Options:
    - smollm = smollm2-1.7b-q8_0
    - qwen = qwen2.5-1.5b-instruct-q4_k_m
    - gemma = gemma-4-E2B.i1-Q4_K_M
    - llama = llama-3.2-3b-q4_0
    """
    smollm = "smollm2-1.7b-q8_0"
    qwen = "qwen2.5-1.5b-instruct-q4_k_m"
    gemma = "gemma-4-E2B.i1-Q4_K_M"
    llama = "llama-3.2-3b-q4_0"

mapping = {
 "smollm2-1.7b-q8_0": "NikolayKozloff/SmolLM2-1.7B-Q8_0-GGUF",
 "qwen2.5-1.5b-instruct-q4_k_m": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
 "gemma-4-E2B.i1-Q4_K_M":"mradermacher/gemma-4-E2B-i1-GGUF",
 "llama-3.2-3b-q4_0":"kaetemi/Llama-3.2-3B-Q4_0-GGUF"
}

def get_model(model:Model):
    filename = model.value + ".gguf"
    filepath = Path(__file__).parent / filename
    if filepath.exists():
        return filepath

    if token_path.exists():
        token = json.loads(token_path.read_text())
    else:
        json_example = '{"download": "hf_xyz"}'
        print(f"please add a token of the form {json_example} to {token_path}")
        token = None
    
    huggingface_hub.hf_hub_download(
        repo_id=mapping[model.value],
        filename=filename,
        local_dir=model_path.resolve(),
        token=token
    )
    if not filepath.exists():
        raise RuntimeError(f"error downloading: maybe it went to the wrong place? Check {token_path}")
    return filepath