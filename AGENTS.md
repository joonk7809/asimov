# AGENTS.md

Project conventions, the working standard and decisions live in the asiimov shared folder
(`AGENTS.md`, `decisions.md`, `plan/`). This file only covers the code in this repo.

## What this is
The world core for Prototype 0 (see `plan/prototype-scope.md` and `plan/agent-individuality.md`
in the shared folder): a deterministic, event-sourced village of agents with individual
parameters, per-agent markdown memory, and minds behind a narrow interface.

## Rules for changes
- Every state change happens inside an emitted event with an `origin` (rule, chance, mind, player)
  and its causal `parents`. No silent state edits.
- Simulation arithmetic is integer-only; randomness goes through `Rng.draw(stream, ...)` so it is
  recomputable from the log.
- Mind outputs are recorded verbatim. `replay()` must reproduce the log and final hash without
  calling any model. `tests/test_world.py` enforces this; keep it green.
- `StubMind` and `NoisyMind` are stand-ins for the GPU-served model, not the product's minds.
  Don't tune game behavior to them.

## Run
    pip install pytest hypothesis
    python -m pytest -q
    python -m asiimov.run --agents 100 --days 7 --seed 1 --fire 3@1 --out runs/demo
