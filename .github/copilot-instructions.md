# Pet LLM — Agent Guide

Desktop "virtual pet" simulation: a Pygame-rendered pet swims in a tank, steered by a local LLM (via llama-cpp-python) that runs in a background thread and emits structured JSON decisions.

## Commands

This project uses **uv** (see `uv.lock`). Always prefix commands with `uv run`:

```bash
uv sync                 # install deps (uses custom CPU wheel index for llama-cpp-python)
uv run pytest           # run tests (-v --tb=short, testpaths=["tests"])
uv run pytest --cov=lib # with coverage
uv run ruff check .     # lint (extended rules: BLE, FBT, A, C4, LOG, G, PT, SIM, TID, I, C901)
uv run ruff format .    # format
uv run main.py          # run the app
```

Python >= 3.11 (uses `StrEnum`). Pydantic v2 (`model_dump_json`).

## Architecture

```
main.py → Tank (orchestrator) → owns → Brain (llama-cpp wrapper)
          ↓ drives via DriverBase     ↓ spawns daemon thread
        PyGameDriver /              LLM inference → PetAction → result_queue
        DummyDriver (headless)        ↕
                                    Memory (bounded deque of RoleContent)
```

| Module | Responsibility |
|---|---|
| [main.py](main.py) | Bootstrap only: resolve GGUF model, build `SimulationConfig`, `Brain` → driver → `Tank(brain, config.tank, driver)` → `run()` |
| [lib/tank.py](lib/tank.py) | Orchestration only: wakes the brain, per frame builds `RenderInfo.from_brain(brain)` → `brain.update(get_info())` → `driver.loop(render_info)`. No rendering, no pygame import. |
| [lib/drivers/base.py](lib/drivers/base.py) | `DriverBase` ABC: `render(info)`, `loop(info)`, owns the `running` flag |
| [lib/drivers/pygame_driver.py](lib/drivers/pygame_driver.py) | All pygame rendering + event handling + runtime clock. Owns screen layout constants (padding, colors, fonts, FPS) |
| [lib/drivers/dummy.py](lib/drivers/dummy.py) | Headless driver for batch runs; only implements `loop` |
| [lib/brain.py](lib/brain.py) | Simulation state + LLM orchestration: position, movement, background inference thread, memory |
| [lib/memory.py](lib/memory.py) | Conversation history as bounded `deque[RoleContent]`; builds LLM messages; thought-loop detection |
| [lib/types/other.py](lib/types/other.py) | Pydantic models: `PetAction` (also the LLM JSON schema), `RoleContent`, `EnvironmentalInfo`, `RenderInfo`, `ChatCompletionResponse` |
| [lib/types/config.py](lib/types/config.py) | Config models: `SimulationConfig` → `TankConfig`/`BrainConfig` (`thoughts`, `params`, `memory`), `TunerConfig` for bayes optimisation |
| [lib/types/report.py](lib/types/report.py) | Report models: `BrainReport`, `OutputReport(config, report)`, `StudyReport`/`Trial` for tuning runs |
| [lib/background_trainer.py](lib/background_trainer.py), [lib/dynamic_adapter_llm.py](lib/dynamic_adapter_llm.py), [lib/experience_buffer.py](lib/experience_buffer.py) | Experimental LoRA fine-tuning pipeline (not wired into `Brain`) |
| [lib/utils.py](lib/utils.py) | `get_logger` factory → rotating `log.txt` in project root (rotations named `log_1.txt`) |
| [models/download.py](models/download.py) | `Model` enum of selectable GGUFs; `get_model()` downloads from HuggingFace if missing (optional token in `models/hf_token.json`) |

Model selection is hardcoded in [main.py](main.py): `get_model(Model.smollm2)`, called at module import time (tests rely on this).

## Critical Invariants — read before editing

1. **Two coordinate systems.** `Brain` works in *tank-local* coordinates: (0,0) = top-left of swimmable area, bounded by `(x_bounds, y_bounds)` passed to `wake_up`. `PyGameDriver` adds `bounds_offset = (TANK_PADDING_X, TEXT_BOX_HEIGHT // 2)` only when drawing. **Never** apply screen offsets inside `Brain` or `Tank`. `Tank.TEXT_BOX_HEIGHT`/`TANK_PADDING_X` still exist (marked TODO) only to compute tank-local bounds for `wake_up` — keep them in sync with the driver's layout constants.

2. **No blocking work in `Brain.update()`** — it runs on the render thread at 60 FPS. It only: drains `result_queue` → integrates movement → maybe kicks off inference. LLM calls belong in the daemon thread (`request_decision_async`), guarded by `is_thinking`.

3. **Threading:** `result_queue: queue.Queue[PetAction]` is the only cross-thread channel. `is_thinking` must be reset in a `finally` block. `wake_up()` must complete before `request_decision_async` (enforced by `assert self.awake`).

4. **Two-phase init:** `Brain.__init__` only stores the model path; `wake_up(bounds)` (called by `Tank`) does the expensive `Llama` load and creates `memory`/`result_queue`.

5. **Drivers are display-only.** The driver never mutates simulation state: `Tank` hands it a `RenderInfo` snapshot built fresh from the brain each frame, and there is no channel for the driver to write back. The loop ends when the driver sets `self.running = False` (on QUIT, or when `pygame.time.get_ticks()` passes `end_time` computed from `TankConfig.runtime`).

6. **Memory is lossy:** `maxlen` comes from `MemoryConfig.max_length` (default 5); old messages silently drop. The system prompt is **not** stored — `get_messages(system_prompt)` prepends it fresh per call. `Memory.__add__` mutates and returns `self` (both `+` and `+=` mutate).

7. **Decision triggering is arrival-based:** new decision requested only when the pet is within `ARRIVAL_THRESHOLD` (3.0 px) of its target. Movement = normalized step of `PET_SPEED` (2.5 px/frame). `PetAction.action` is currently commented out of the schema — only `thought`, `target_x`, `target_y` exist.

## Testing Conventions

- [tests/conftest.py](tests/conftest.py) only inserts the project root into `sys.path` — no shared fixtures; each test file defines its own.
- Mock at the import site: `patch("lib.brain.Llama")`, `patch("lib.drivers.pygame_driver.pygame")`, `patch("lib.brain.threading.Thread")`.
- Fake LLM responses with `brain.llm = MagicMock()` and `create_chat_completion.return_value` = an OpenAI-style chat-completion dict whose content is `PetAction.model_dump_json()`.
- Tank tests inject a plain driver double exposing `running` + `loop(info)` (see `ScriptedDriver` in [tests/test_tank_driver_loop.py](tests/test_tank_driver_loop.py)); `mock_brain` is a `MagicMock` whose attributes mimic a post-`wake_up` brain.
- Config models in tests: `BrainConfig.model_construct()` / `TankConfig.model_construct()` (skips building nested defaults).
- Tests use `unittest.mock` directly (though `pytest-mock` is installed).
- `Brain("fake/model/path.gguf")` constructs fine without a real model — loading is deferred to `wake_up`.
- Mock surgically, to avoid expensive calls to `llama.cpp` et al, while still ensuring that the tests are representative of common usage


## Known Quirks (don't "fix" without checking)

- `Tank.get_info()` currently returns a hardcoded `EnvironmentalInfo(mouse=(0,0))` — it is vestigial but will be re-implemented later; don't remove it.
- [lib/tank.py](lib/tank.py) has unused module-level `version`/`report_path` globals and an unused `json` import — leftovers from a removed `generate_report()`.
- The planned roadmap is in [TODO.md](TODO.md); setup/usage in [README.md](README.md).

## Logging

Use `get_logger(name, level)` from [lib/utils.py](lib/utils.py) — never `print()`. It clears existing handlers on each call and writes to the rotating project-root `log.txt` (format: `%(asctime)s %(levelname)s - %(message)s`).
