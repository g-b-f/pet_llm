import json
from enum import StrEnum
from pathlib import Path

import huggingface_hub

model_path = token_path = Path(__file__).parent
token_path = model_path / "hf_token.json"

class Model(StrEnum):
    """Model to load into the brain. Options:
    - smollm2 = smollm2-1.7b-q8_0
    - smollm3 = SmolLM3-3B-128K-Q4_K_M
    - qwen = qwen2.5-1.5b-instruct-q4_k_m
    - gemma = gemma-4-E2B.i1-Q4_K_M
    - llama = llama-3.2-3b-q4_0
    - LFM = LFM2.5-8B-A1B-UD-Q4_K_M
    - miniCPM = "minicpm5-1b-Q8_0"
    - deepseek = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
    - granite = "granite-4.2-3b-Q4_K_M"
    """
    smollm2 = "smollm2-1.7b-q8_0"
    smollm3 = "SmolLM3-3B-128K-Q4_K_M"
    qwen2_5 = "qwen2.5-1.5b-instruct-q4_k_m"
    qwen3 = "Qwen3-4B-Q4_K_M"
    gemma = "gemma-4-E2B.i1-Q4_K_M"
    llama = "llama-3.2-3b-q4_0"
    LFM = "LFM2.5-2.6B-Q4_K_M"
    miniCPM = "minicpm5-1b-Q8_0"
    deepseek = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M"
    granite = "granite-4.2-3b-Q4_K_M"


mapping: dict[Model, str] = {
    Model.smollm2 : "NikolayKozloff/SmolLM2-1.7B-Q8_0-GGUF",
    Model.smollm3 : "unsloth/SmolLM3-3B-128K-GGUF",
    Model.qwen2_5 : "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    Model.qwen3 : "Qwen/Qwen3-4B-GGUF",
    Model.gemma : "mradermacher/gemma-4-E2B-i1-GGUF",
    Model.llama : "kaetemi/Llama-3.2-3B-Q4_0-GGUF",
    Model.LFM : "LiquidAI/LFM2.5-2.6B-GGUF",
    Model.miniCPM : "Abiray/MiniCPM5-1B-GGUF",
    Model.deepseek : "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
    Model.granite : "ibm-granite/granite-4.2-3b-GGUF"

}

def get_model(model:Model):
    filename = model.value + ".gguf"
    filepath = Path(__file__).parent / filename
    if filepath.exists():
        return filepath

    if token_path.exists():
        token = json.loads(token_path.read_text())["download"]
    else:
        json_example = '{"download": "hf_xyz"}'
        print(f"please add a token of the form {json_example} to {token_path}")
        token = None
    
    huggingface_hub.hf_hub_download(
        repo_id=mapping[model],
        filename=filename,
        local_dir=model_path.resolve(),
        token=token
    )
    if not filepath.exists():
        raise RuntimeError(
            f"error downloading: maybe it went to the wrong place? Check {token_path}"
        )
    return filepath