from pathlib import Path
MODEL_FILE_PATH = (Path().parent/"models/qwen2.5-1.5b-instruct-q4_k_m.gguf").resolve()

print(MODEL_FILE_PATH)