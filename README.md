# asiimov

World engine prototype: a persistent village of individually different, minded agents whose
actions have traceable consequences. See AGENTS.md.

Status (2026-11-03): world core with stand-in minds, plus an LLM mind (`asiimov/llm.py`) for an
open model served by vLLM on a RunPod GPU. Not yet run against a real model: the cloud session's
network policy blocks RunPod.

## Real minds on RunPod
    export RUNPOD_API_KEY=...
    python scripts/runpod_pod.py up --model Qwen/Qwen3-4B-Instruct-2507   # prints POD_ID
    python scripts/runpod_pod.py wait POD_ID
    python -m asiimov.bench --pod POD_ID --out runs/bench-4b.json         # throughput, $/1k calls
    python -m asiimov.run --mind llm --pod POD_ID --agents 100 --days 7 --seed 1 --fire 3@1 --out runs/llm-4b
    python scripts/runpod_pod.py down POD_ID                               # always terminate
