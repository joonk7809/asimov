#!/usr/bin/env bash
# B3 cost-at-scale protocol, run on a pod by scripts/podjob.py (LLM_URL/LLM_MODEL/LLM_KEY come from the pod):
#   python scripts/podjob.py run POD cost-4b 'bash scripts/cost_at_scale.sh 0.59 full'
# Every bench uses its own world seed, so no prompt was cached by an earlier bench on the same server.
# full: throughput curve, cache A/B, 1,000-agent prompts, then real-mind worlds of 100, 300 and 1,000 agents
# for one world day each (the tick-synchronous engine as it is). bench: the throughput curve and cache A/B only.
set -u
COST=$1; WHAT=${2:-full}
M="--llm-url $LLM_URL --llm-model $LLM_MODEL --llm-key $LLM_KEY"
mkdir -p out
python3 -m asiimov.bench $M --cost-per-hr "$COST" --concurrency 1 16 64 128 256 --n 256 --agents 100 --days 4 \
    --world-seed 1 --out out/bench-100.json
python3 -m asiimov.bench $M --cost-per-hr "$COST" --concurrency 256 --n 512 --agents 100 --days 4 \
    --world-seed 2 --cache-bust --out out/bench-100-bust.json
python3 -m asiimov.bench $M --cost-per-hr "$COST" --concurrency 256 --n 512 --agents 100 --days 4 \
    --world-seed 3 --out out/bench-100-cached.json   # same size as the bust arm, so the A/B is like for like
[ "$WHAT" = full ] || exit 0
python3 -m asiimov.bench $M --cost-per-hr "$COST" --concurrency 256 --n 512 --agents 1000 --days 1 \
    --world-seed 4 --out out/bench-1000.json
for n in 100 300 1000; do
    python3 -m asiimov.run --mind llm $M --agents $n --days 1 --seed 1 --workers 256 --out out/world-$n \
        > out/world-$n.log 2>&1
    tail -c 3000 out/world-$n.log
done
