import ast
import contextlib
import re
from pathlib import Path

from lib.types.log import (
    LogTrial,
    MalformedJSONEvent,
    OOBResetEvent,
    OptimisationLog,
    Params,
    SeedRun,
    ThoughtEvent,
    ThoughtLoopEvent,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

INPUT_PATH = Path(__file__).parent / "reports" / "llama_log.txt"
OUTPUT_PATH = INPUT_PATH.with_suffix(".json")

# --------------------------------------------------------------------------- #
# Log-line helpers
# --------------------------------------------------------------------------- #

# "2026-09-05 19:40:42 INFO - starting optimisation"  ->  "starting optimisation"
_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+ - (.*)$")


def _message(line: str) -> str | None:
    """Return the message part of a log line.

    Args:
        line: A raw log line, e.g. ``"2026-09-05 19:40:42 INFO - thought 'hi'"``.

    Returns:
        The message after the ``"LEVEL - "`` prefix, or ``None`` if the line
        does not match the expected log format.
    """
    match = _LINE_RE.match(line)
    return match.group(1) if match else None


def _parse_numbers(message: str) -> dict[str, int | float]:
    """Parse a ``k=v, k=v, ...`` message into a dict of numbers.

    Args:
        message: A comma-separated list of ``key=value`` pairs, e.g.
            ``"seed=0, context_size=2048, temperature=1.45"``.

    Returns:
        A mapping of each key to its value as an ``int`` where possible,
        otherwise a ``float``. Pairs that are not numeric are skipped.
    """
    out: dict[str, int | float] = {}
    for part in message.split(", "):
        key, sep, raw = part.partition("=")
        if not sep:
            continue
        raw = raw.strip()
        try:
            out[key] = int(raw)
        except ValueError:
            with contextlib.suppress(ValueError):
                out[key] = float(raw)
    return out


def _parse_repr(message: str, prefix: str) -> str | None:
    """Parse ``<prefix><python-repr-string>`` into the underlying string.

    The brain logs thoughts and raw LLM output with ``!r``, so the payload is
    always a valid Python string literal (quotes chosen to avoid clashing with
    the content, internal quotes/newlines escaped). ``ast.literal_eval``
    reverses that safely.

    Args:
        message: The full log message.
        prefix: The literal prefix that must precede the repr, e.g.
            ``"thought "`` or ``"malformed JSON: "``.

    Returns:
        The decoded string, or ``None`` if the message does not start with
        ``prefix`` or the payload is not a valid string literal.
    """
    if not message.startswith(prefix):
        return None
    try:
        value = ast.literal_eval(message[len(prefix) :].strip())
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


class _ParseState:
    """Mutable state carried across log lines while parsing."""

    def __init__(self) -> None:
        self.trial: LogTrial | None = None
        self.run: SeedRun | None = None
        self.thought: ThoughtEvent | None = None

    def close_run(self, log: OptimisationLog) -> None:
        """Attach the in-progress run to its trial and reset the run state."""
        if self.run is not None:
            if self.trial is None:
                self.trial = LogTrial()
                log.trials.append(self.trial)
            self.trial.seeds.append(self.run)
        self.run = None
        self.thought = None


def _parse_params(numbers: dict[str, int | float]) -> Params | None:
    """Build a :class:`Params` from a ``seed=`` line's numeric fields.

    Args:
        numbers: The parsed ``key=value`` pairs from a ``seed=`` line.

    Returns:
        The parameters, or ``None`` if the line carried no parameters or they
        could not be validated.
    """
    fields = {k: v for k, v in numbers.items() if k != "seed"}
    if not fields:
        return None
    try:
        return Params.model_validate(fields)
    except ValueError:
        return None


def _parse_target(message: str) -> tuple[int, int] | None:
    """Parse the coordinates from a ``tried to go to (x, y)`` message.

    Args:
        message: The full log message.

    Returns:
        The ``(x, y)`` coordinates, or ``None`` if they cannot be parsed.
    """
    try:
        target = ast.literal_eval(message[len("tried to go to ") :].strip())
    except (ValueError, SyntaxError):
        return None
    if isinstance(target, tuple) and len(target) == 2:
        return int(target[0]), int(target[1])
    return None


def _handle_event(
    message: str, run: SeedRun, current_thought: ThoughtEvent | None
) -> ThoughtEvent | None:
    """Append the event described by ``message`` to ``run``.

    Args:
        message: The full log message.
        run: The seed run the event belongs to.
        current_thought: The most recent thought event, if any.

    Returns:
        The updated current thought (only a ``thought`` event changes it).
    """
    if message.startswith("thought "):
        thought = _parse_repr(message, "thought ")
        if thought is not None:
            event = ThoughtEvent(thought=thought)
            run.events.append(event)
            return event
    elif message.startswith("tried to go to "):
        target = _parse_target(message)
        if current_thought is not None and target is not None:
            current_thought.target = target
    elif message.startswith("attempted out-of-bounds too much"):
        run.events.append(OOBResetEvent())
    elif message.startswith("thought loop detected"):
        run.events.append(ThoughtLoopEvent())
    elif message.startswith("malformed JSON:"):
        content = _parse_repr(message, "malformed JSON: ")
        if content is not None:
            run.events.append(MalformedJSONEvent(content=content))
    # "thought was:" and any other message are dropped
    return current_thought


def _handle_seed(message: str, log: OptimisationLog, state: _ParseState) -> None:
    """Start a new seed run, opening a new trial if the params changed.

    Args:
        message: The ``seed=`` log message.
        log: The log being built.
        state: The mutable parse state.
    """
    numbers = _parse_numbers(message)
    state.close_run(log)
    params = _parse_params(numbers)
    # A change in params (None <-> Params, or different values) marks the
    # start of a new trial.
    if state.trial is None or state.trial.params != params:
        state.trial = LogTrial(params=params)
        log.trials.append(state.trial)
    state.run = SeedRun(seed=int(numbers.get("seed", -1)))


def _handle_loss(message: str, log: OptimisationLog, state: _ParseState) -> None:
    """Record the loss for the current run and close it.

    Args:
        message: The ``loss=`` log message.
        log: The log being built.
        state: The mutable parse state.
    """
    numbers = _parse_numbers(message)
    if state.run is not None and "loss" in numbers:
        state.run.loss = float(numbers["loss"])
    state.close_run(log)


def _dispatch(message: str, log: OptimisationLog, state: _ParseState) -> None:
    """Route a single log message to the appropriate handler.

    Args:
        message: The message part of a log line.
        log: The log being built.
        state: The mutable parse state.
    """
    if message.startswith("RUNTIME="):
        numbers = _parse_numbers(message)
        if "RUNTIME" in numbers:
            log.runtime = int(numbers["RUNTIME"])
        return
    if message.startswith(("starting optimisation", "eta:", "system prompt hash:")):
        return
    if message.startswith("seed="):
        _handle_seed(message, log, state)
        return
    if message.startswith("loss="):
        _handle_loss(message, log, state)
        return
    # An event belongs to the current run; open one defensively if needed.
    if state.run is None:
        state.run = SeedRun(seed=-1)
    if state.trial is None:
        state.trial = LogTrial()
        log.trials.append(state.trial)
    state.thought = _handle_event(message, state.run, state.thought)


def parse_log(path: Path) -> OptimisationLog:
    """Stream ``path`` line-by-line and build an :class:`OptimisationLog`.

    Args:
        path: Path to the optimisation log file.

    Returns:
        The parsed log, with trials grouped by their shared parameters and
        each seed run's events in the order they were logged.
    """
    log = OptimisationLog()
    state = _ParseState()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            message = _message(line)
            if message is not None:
                _dispatch(message, log, state)
    state.close_run(log)
    return log


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    """Parse the log at ``INPUT_PATH`` and write JSON to ``OUTPUT_PATH``."""
    log = parse_log(INPUT_PATH)
    OUTPUT_PATH.write_text(log.model_dump_json(indent=2) + "\n", encoding="utf-8")

    n_seeds = sum(len(trial.seeds) for trial in log.trials)
    n_events = sum(len(run.events) for trial in log.trials for run in trial.seeds)
    print(f"wrote {OUTPUT_PATH}")
    print(f"runtime={log.runtime} trials={len(log.trials)} seeds={n_seeds} events={n_events}")


if __name__ == "__main__":
    main()
