from pathlib import Path

import modal

# --- Cost controls -----------------------------------------------------------
# CPU-only (no GPU). 4 vCPU gives llama.cpp enough threads for the 1.7B Q8 model
# without paying for idle cores; this is the single biggest cost lever.
CPU_CORES = 4.0
MEMORY_MB = 8192  # ~1.7 GB model + ctx + overhead
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW = 5  # shut the container down ASAP after the run

# Wall-clock budget for the *optimization loop itself* (model load excluded).
# The study stops starting new trials once this elapses -> a predictable cost ceiling.
# TIME_BUDGET = 12 * 60 * 60
TIME_BUDGET = 1 * 60 * 60
TIMEOUT = TIME_BUDGET + (60 * 60)

# RUNTIME = 300  # seconds per simulation
RUNTIME = 120  # seconds per simulation
# N_TRIALS = 20
N_TRIALS = 2

APP_NAME = "testing"
VOLUME_NAME = "test-output"
VOLUME_MOUNT = "/vol"

app = modal.App(APP_NAME)
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

PROJECT_ROOT = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    # CPU-only llama.cpp wheel (same custom index as pyproject.toml).
    .pip_install(
        "llama-cpp-python",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cpu",
    )
    .pip_install("pygame", "pydantic", "optuna")
    .env(
        {
            # Headless pygame: run the real Tank/sim loop with no window or audio.
            "SDL_VIDEODRIVER": "dummy",
            "SDL_AUDIODRIVER": "dummy",
            # Mounted packages live at /root/lib, /root/models (namespace packages).
            "PYTHONPATH": "/root",
            # Signal to the sim loop that it may skip rendering entirely.
            "PET_LLM_HEADLESS": "1",
            # Keep llama.cpp threads matched to the container's CPUs.
            "OMP_NUM_THREADS": str(int(CPU_CORES)),
        }
    )
    .add_local_dir(PROJECT_ROOT / "lib", remote_path="/root/lib")
    .add_local_dir(PROJECT_ROOT / "models", remote_path="/root/models")
    .add_local_file(
        PROJECT_ROOT / "models" / "smollm2-1.7b-q8_0.gguf",
        remote_path="/root/models/smollm2-1.7b-q8_0.gguf",
    )
    # tank.py reads the project version from pyproject.toml.
    .add_local_file(PROJECT_ROOT / "pyproject.toml", remote_path="/root/pyproject.toml")
)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: output_volume},
    cpu=CPU_CORES,
    memory=MEMORY_MB,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
)
def run_study():
    volume = Path(VOLUME_MOUNT)
    for file_name in ("report.json", "log.txt"):
        file_path = Path("/root") / file_name
        if file_path.exists():
            shutil.copy(file_path, volume / file_name)
    (volume / "best_params.json").write_text(
        json.dumps({
            "best_params": study.best_params,
            "best_loss": study.best_value,
            "trials_completed": len(study.trials),
            "elapsed_hours": round(elapsed_h, 3),
            }, indent=2
        )
    )
    output_volume.commit()


if __name__ == "__main__":
    print(f"approx time = {RUNTIME*N_TRIALS}")
    with app.run(detach=True):
        call = run_study.spawn()
        print(f"Spawned remote study. Function call ID: {call.object_id}")
        print(f"Stream logs:   modal app logs {APP_NAME}")
        print(f"Fetch results: modal volume get {VOLUME_NAME} report.json .")
