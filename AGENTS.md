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
- A tick collects mind requests, then commits replies in `Sim._order` (the one place that decides who
  acts first; `--order id|shuffle`). Never apply a reply where it is requested: concurrency must not
  change the log.
- A complete log starts with `genesis` (naming `ENGINE_VERSION`) and ends with `end`. Bump
  `ENGINE_VERSION` and the golden hash in `tests/test_world.py` whenever a change alters the log
  for the same inputs.
- Runs persist as they go (`asiimov/persist.py`): resuming a crashed run must write the same bytes
  an uninterrupted run writes. `tests/test_persist.py` enforces this.
- Mind outputs are recorded verbatim. `replay()` must reproduce the log and final hash without
  calling any model. `tests/test_world.py` enforces this; keep it green.
- `StubMind` and `NoisyMind` are stand-ins for the GPU-served model, not the product's minds.
  Don't tune game behavior to them.

## Run
    pip install pytest hypothesis
    python -m pytest -q
    python -m asiimov.run --agents 100 --days 7 --seed 1 --fire 3@1 --out runs/demo
    python -m asiimov.run --resume --days 10 --out runs/demo   # continue after a crash, or run longer
    python -m asiimov.scorecard --mind stub --out runs/scorecard-stub   # variety tests, plan/variety-scorecard.md
    python -m asiimov.diagnostics runs/demo/events.jsonl   # plan copy rate, value-word core share (any engine)
    python -m asiimov.authorship --log runs/demo/events.jsonl --out runs/authorship   # who authors each waking tick (test 7)
    python -m asiimov.fork --base runs/demo/events.jsonl --out runs/twin --fire 2@1 --pod POD_ID   # twin: served where prompts match, live elsewhere (tests 8, 9, 15)
