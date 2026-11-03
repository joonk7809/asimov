"""Throughput and cost benchmark for an LLM mind server.

Uses real world prompts: runs the stub world to collect the exact prompts the sim sends
(plan, deliberate, reflect in their natural mix), then replays them against the server at
several concurrency levels.

    python -m asiimov.bench --pod POD_ID --concurrency 1 8 32 64 --n 256 --out runs/bench-4b.json
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from .llm import LLMMind
from .minds import StubMind
from .run import llm_from_args
from .sim import Sim


class Recorder(StubMind):
    def __init__(self):
        self.reqs = []

    def decide(self, req):
        self.reqs.append(req)
        return super().decide(req)


def collect_prompts(agents=100, days=2, seed=1):
    rec = Recorder()
    Sim(seed, agents, rec).run(days)
    return rec.reqs


def measure(mind: LLMMind, reqs, concurrency: int) -> dict:
    mind.calls = []
    parsed = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(concurrency) as ex:
        replies = list(ex.map(mind.decide, reqs))
    wall = time.monotonic() - t0
    for r in replies:
        try:
            json.loads(r.raw_output)
            parsed += 1
        except json.JSONDecodeError:
            pass
    s = mind.stats()
    return {"concurrency": concurrency, "requests": len(reqs), "wall_s": round(wall, 2),
            "req_per_s": round(len(reqs) / wall, 2),
            "out_tok_per_s": round(s["completion_tokens"] / wall, 1),
            "in_tok_per_s": round(s["prompt_tokens"] / wall, 1),
            "mean_prompt_tokens": s["prompt_tokens"] // len(reqs),
            "mean_completion_tokens": s["completion_tokens"] // len(reqs),
            "latency_p50_s": s["latency_p50_s"], "latency_p95_s": s["latency_p95_s"],
            "valid_json_share": round(parsed / len(reqs), 3), "truncated": s["truncated"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod")
    ap.add_argument("--llm-url")
    ap.add_argument("--llm-model")
    ap.add_argument("--llm-key")
    ap.add_argument("--cost-per-hr", type=float, help="defaults to the pod record's price")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mind, rec = llm_from_args(args)
    cost = args.cost_per_hr or (rec or {}).get("cost_per_hr")
    reqs = collect_prompts()
    kinds = {}
    for r in reqs:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    mind.decide(reqs[0])  # warm-up
    results = []
    for c in args.concurrency:
        n = min(args.n, len(reqs)) if c > 1 else min(64, len(reqs))
        # Spread the sample across the run so the kind mix matches the world's.
        sample = reqs[:: max(1, len(reqs) // n)][:n]
        r = measure(mind, sample, c)
        if cost:
            r["usd_per_1k_calls"] = round(float(cost) / (r["req_per_s"] * 3600) * 1000, 4)
        results.append(r)
        print(json.dumps(r), flush=True)
    out = {"model": mind.model, "cost_per_hr": cost, "gpu": (rec or {}).get("gpu"),
           "prompt_mix": kinds, "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
