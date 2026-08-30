"""Fully-remote Bayesian hyperparameter optimization on Modal.

The entire Optuna study runs inside a single detached Modal Function. This local
process only builds the image, spawns the remote call, and exits -- it does NOT
wait for results, so you can shut your computer off after launching.

Outputs (all remote, nothing local):
  - report.json / log.txt / best_params.json  -> Modal Volume `pet-llm-bayes-output`
  - full stdout/stderr + ETA                  -> `modal app logs pet-llm-bayes-optimise`
"""

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
TIME_BUDGET = 12 * 60 * 60
TIMEOUT = TIME_BUDGET + (60 * 60)

RUNTIME = 300  # seconds per simulation
N_TRIALS = 20

APP_NAME = "pet-llm-bayes-optimise"
VOLUME_NAME = "pet-llm-bayes-output"
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
def run_study() -> None:
    import json
    import os
    import shutil
    import time

    import optuna

    from lib.brain import Brain
    from lib.extra_types import SimulationConfig
    from lib.tank import Tank
    from lib.utils import get_logger, loss_function
    from models.download import Model, get_model

    logger = get_logger(__name__)
    model_path = get_model(Model.smollm2)

    headless = os.environ.get("PET_LLM_HEADLESS") == "1"

    def evaluate_simulation(trial: optuna.Trial) -> float:
        """Sample one parameter set, run the sim, and return its scalar loss."""
        temperature = trial.suggest_float("temperature", 1.5, 2.5)
        frequency_penalty = trial.suggest_float("frequency_penalty", 1.5, 2.5)
        presence_penalty = trial.suggest_float("presence_penalty", 1.5, 2.5)
        repeat_penalty = trial.suggest_float("repeat_penalty", 1.5, 2.5)

        config = SimulationConfig.model_construct()
        config.tank.runtime = RUNTIME
        config.brain.params.temperature = temperature
        config.brain.params.frequency_penalty = frequency_penalty
        config.brain.params.presence_penalty = presence_penalty
        config.brain.params.repeat_penalty = repeat_penalty

        brain = Brain(model_path, config.brain)
        tank = Tank(brain, config.tank)
        if headless:
            tank._render_scene = lambda: None  # type: ignore[attr-defined, method-assign]
        logger.info(
            f"{temperature=}, {frequency_penalty=}, "
            f"{presence_penalty=}, {repeat_penalty=}"
        )
        result = tank.run()
        loss = loss_function(result.report)
        logger.info(f"{loss=}")
        return loss

    study = optuna.create_study(direction="minimize")

    start = time.monotonic()
    deadline = start + TIME_BUDGET
    first_trial_duration: float | None = None

    for trial_num in range(N_TRIALS):
        if time.monotonic() >= deadline:
            logger.info(f"Optimization budget reached after {trial_num} trials; stopping early.")
            break

        trial_start = time.monotonic()
        study.optimize(evaluate_simulation, n_trials=1)
        trial_duration = time.monotonic() - trial_start

        if first_trial_duration is None:
            first_trial_duration = trial_duration
            eta = first_trial_duration * N_TRIALS
            logger.info(
                f"First trial took {first_trial_duration / 60:.1f} min -> "
                f"rough ETA ~{eta / 3600:.1f} h for all {N_TRIALS} trials "
                f"(budget-capped at {TIME_BUDGET / 3600:.0f} h)."
            )  # noqa: G004
        else:
            done = trial_num + 1
            remaining = max(N_TRIALS - done, 0)
            avg = (time.monotonic() - start) / done
            logger.info(
                f"Trial {done}/{N_TRIALS} took {trial_duration / 60:.1f} min; "
                f"~{remaining * avg / 3600:.1f} h remaining."
            )

    elapsed_h = (time.monotonic() - start) / 3600
    logger.info(f"Best parameters: {study.best_params}") 
    logger.info(f"Best loss: {study.best_value}")

    # Persist outputs to the Modal Volume (report.json/log.txt are written by
    # tank.py / utils.py into /root; copy them out and commit).
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
    with app.run(detach=True):
        call = run_study.spawn()
        print(f"Spawned remote study. Function call ID: {call.object_id}")
        print(f"Stream logs:   modal app logs {APP_NAME}")
        print(f"Fetch results: modal volume get {VOLUME_NAME} report.json .")
