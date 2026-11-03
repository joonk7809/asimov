# asiimov

World engine prototype: a persistent village of individually different, minded agents whose
actions have traceable consequences. See AGENTS.md.

Status (2026-11-03): world core with stand-in minds, plus an LLM mind (`asiimov/llm.py`) for an
open model served by vLLM on a RunPod GPU. Qwen3-4B and Qwen3-8B have run the 7-day,
100-agent world on a 48GB card at $0.59/hr (about $0.006 per 1k mind calls at 64 concurrent).

## Real minds on RunPod
    export RUNPOD_API_KEY=...
    python scripts/runpod_pod.py up --model Qwen/Qwen3-4B-Instruct-2507 --gpu "NVIDIA RTX A6000" "NVIDIA A40"  # prints POD_ID; self-terminates after --ttl-hours (6)
    python scripts/runpod_pod.py wait POD_ID
    python -m asiimov.bench --pod POD_ID --out runs/bench-4b.json         # throughput, $/1k calls
    python -m asiimov.run --mind llm --pod POD_ID --agents 100 --days 7 --seed 1 --fire 3@1 --out runs/llm-4b
    python -m asiimov.run --resume --mind llm --pod NEW_POD_ID --days 7 --out runs/llm-4b  # if the pod died
    python -m asiimov.scorecard --mind llm --pod POD_ID --arms full --out runs/sc-4b  # one arm per process; then --grade-only
    python scripts/runpod_pod.py down POD_ID                               # always terminate
