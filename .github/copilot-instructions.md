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
main.py → Tank (pygame UI) → owns → Brain (llama-cpp wrapper)
                                      ↓ spawns daemon thread
                                    LLM inference → PetAction → result_queue
                                      ↕
                                    Memory (bounded deque of RoleContent)
```

| Module | Responsibility |
|---|---|
| [main.py](main.py) | Bootstrap only: resolve GGUF model, `Brain(model_path)` → `Tank(brain)` → `run()` |
| [lib/tank.py](lib/tank.py) | Presentation: pygame window, game loop (`run()` at 60 FPS), rendering, screen offsets |
| [lib/brain.py](lib/brain.py) | Simulation state + LLM orchestration: position, movement, background inference thread, memory |
| [lib/memory.py](lib/memory.py) | Conversation history as bounded `deque[RoleContent]`; builds LLM messages; thought-loop detection |
| [lib/extra_types.py](lib/extra_types.py) | Pydantic models: `PetAction` (also the LLM JSON schema), `RoleContent`, `EnvironmentalInfo`, `Action` enum |
| [lib/utils.py](lib/utils.py) | `get_logger` factory → rotating `log.txt` in project root (rotations named `log_1.txt`) |
| [models/download.py](models/download.py) | `Model` enum of selectable GGUFs; `get_model()` downloads from HuggingFace if missing (optional token in `models/hf_token.json`) |

Model selection is hardcoded in [main.py](main.py): `get_model(Model.smollm)`, called at module import time (tests rely on this).

## Critical Invariants — read before editing

1. **Two coordinate systems.** `Brain` works in *tank-local* coordinates: (0,0) = top-left of swimmable area, bounded by `(x_bounds, y_bounds)` passed to `wake_up`. `Tank` adds `bounds_offset = (TANK_PADDING_X, TEXT_BOX_HEIGHT // 2)` only when drawing. **Never** apply screen offsets inside `Brain`.

2. **No blocking work in `Brain.update()`** — it runs on the render thread at 60 FPS. It only: drains `result_queue` → integrates movement → maybe kicks off inference. LLM calls belong in the daemon thread (`request_decision_async`), guarded by `is_thinking`.

3. **Threading:** `result_queue: queue.Queue[PetAction]` is the only cross-thread channel. `is_thinking` must be reset in a `finally` block. `wake_up()` must complete before `request_decision_async` (enforced by `assert self.awake`).

4. **Two-phase init:** `Brain.__init__` only stores the model path; `wake_up(bounds)` (called by `Tank`) does the expensive `Llama` load and creates `memory`/`result_queue`.

5. **`use_enum_values=True` on Pydantic models** — `PetAction.action` is a `str` at runtime, not an `Action` enum member. Note: the movement code ignores `action` entirely; only `target_x/y` are used.

6. **Memory is lossy:** `maxlen=5` (`MEMORY_LENGTH`); old messages silently drop. The system prompt is **not** stored — `get_messages(system_prompt)` prepends it fresh per call. `Memory.__add__` mutates and returns `self` (both `+` and `+=` mutate).

7. **Decision triggering is arrival-based:** new decision requested only when the pet is within `ARRIVAL_THRESHOLD` (3.0 px) of its target. Movement = normalized step of `PET_SPEED` (2.5 px/frame).

## Configuration

Tuning knobs are **class attributes**, not config files:
- `Brain`: `CONTEXT_SIZE=2048`, `TEMPERATURE=2` (deliberately high for erratic behavior), penalties, `MEMORY_LENGTH=5`, `PET_SPEED`, `ARRIVAL_THRESHOLD`, `MAX_OOB_COUNT=3` ([lib/brain.py](lib/brain.py))
- `Tank`: 800×600 screen, layout/colors, `FPS=60`; module-level `DEBUG = True` toggles the on-screen HUD ([lib/tank.py](lib/tank.py))
- `Tank` reads `pyproject.toml` at import time (via `tomllib`) to display the version — the file must parse even in tests.

## Testing Conventions

- [tests/conftest.py](tests/conftest.py) only inserts the project root into `sys.path` — no shared fixtures; each test file defines its own.
- Mock at the import site: `patch("lib.brain.Llama")`, `patch("lib.tank.pygame")`, `patch("lib.brain.threading.Thread")`.
- Fake LLM responses with `brain.llm = MagicMock()` and `create_chat_completion.return_value` = an OpenAI-style chat-completion dict whose content is `PetAction.model_dump_json()`.
- `mock_brain` for tank tests: `MagicMock(spec=Brain)`.
- Tests use `unittest.mock` directly (though `pytest-mock` is installed).
- `Brain("fake/model/path.gguf")` constructs fine without a real model — loading is deferred to `wake_up`.

## Known Quirks (don't "fix" without checking)

- No `try/except` around `json.loads(content)` in the worker thread: malformed LLM JSON can leave `is_thinking=True` forever (latent deadlock).
- The planned roadmap is in [TODO.md](TODO.md); setup/usage in [README.md](README.md).

## Logging

Use `get_logger(name, level)` from [lib/utils.py](lib/utils.py) — never `print()`. It clears existing handlers on each call and writes to the rotating project-root `log.txt` (format: `%(asctime)s %(levelname)s - %(message)s`).
