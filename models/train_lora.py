"""Standalone LoRA trainer invoked by BackgroundTrainer as a subprocess.

Pipeline:
1. Read a JSONL chat dataset (one ``{"messages": [...]}`` object per line).
2. Fine-tune a LoRA adapter with HuggingFace ``peft`` (CPU).
3. Convert the peft adapter to a GGUF LoRA ``.gguf`` via llama.cpp's
   ``convert_lora_to_gguf.py`` (downloaded once from the llama.cpp repo), so the
   result can be hot-swapped into ``DynamicAdapterLLM``.

Run directly for a smoke test::

    uv run python models/train_lora.py --model <hf-id> --dataset data.jsonl --output adapter.gguf
"""

import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from lib.utils import get_logger

logger = get_logger(__name__)

CONVERT_SCRIPT_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_lora_to_gguf.py"
CONVERT_SCRIPT_CACHE = Path(__file__).parent / "_convert_lora_to_gguf.py"


def _load_dataset(dataset_path: Path) -> list[str]:
    """Flatten each chat example into a single training text."""
    texts: list[str] = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            messages = example["messages"]
            texts.append("\n".join(f"{m['role']}: {m['content']}" for m in messages))
    return texts


def _train_peft(
    model_name: str, dataset_path: Path, peft_dir: Path, epochs: int
) -> Path:
    """Train a LoRA adapter and save it in HuggingFace peft format."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    texts = _load_dataset(dataset_path)
    if not texts:
        raise ValueError(f"no training examples in {dataset_path}")
    logger.info("loaded %s examples from %s", len(texts), dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    labels = input_ids.clone()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        logger.info("epoch %s/%s loss=%.4f", epoch + 1, epochs, loss.item())

    peft_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(peft_dir)
    tokenizer.save_pretrained(peft_dir)
    logger.info("saved peft adapter to %s", peft_dir)
    return peft_dir


def _get_convert_script() -> Path:
    """Return llama.cpp's convert_lora_to_gguf.py, downloading it once."""
    if not CONVERT_SCRIPT_CACHE.exists():
        logger.info("downloading convert_lora_to_gguf.py from llama.cpp")
        with urllib.request.urlopen(CONVERT_SCRIPT_URL, timeout=60) as resp:
            CONVERT_SCRIPT_CACHE.write_bytes(resp.read())
    return CONVERT_SCRIPT_CACHE


def _convert_to_gguf(peft_dir: Path, output_path: Path, base_model: str) -> Path:
    """Convert a peft adapter directory into a GGUF LoRA .gguf file."""
    script = _get_convert_script()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--outfile",
        str(output_path),
        "--outtype",
        "f32",
        "--base",
        base_model,
        str(peft_dir),
    ]
    logger.info("converting to GGUF: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603
    if not output_path.exists():
        raise RuntimeError(f"GGUF conversion produced no file at {output_path}")
    return output_path


def train(model_name: str, dataset_path: Path, output_path: Path, epochs: int) -> Path:
    with tempfile.TemporaryDirectory(prefix="pet_peft_") as tmp:
        peft_dir = _train_peft(model_name, dataset_path, Path(tmp) / "adapter", epochs)
        return _convert_to_gguf(peft_dir, output_path, model_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter from a chat dataset and export it as GGUF."
    )
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument(
        "--dataset", required=True, type=Path, help="JSONL chat dataset"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="GGUF adapter output path"
    )
    parser.add_argument("--epochs", type=int, default=1, help="training epochs")
    args = parser.parse_args(argv)

    try:
        train(args.model, args.dataset, args.output, args.epochs)
    except Exception:
        logger.exception("training failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
