"""Throughput and cost benchmark for an LLM mind server.

Uses real world prompts: runs the stub world to collect the exact prompts the sim sends
(plan, deliberate, reflect in their natural mix), then replays them against the server at
several concurrency levels.

    python -m asiimov.bench --pod POD_ID --concurrency 1 8 32 64 --n 256 --out runs/bench-4b.json

On the pod itself (scripts/podjob.py), --llm-url $LLM_URL skips the public proxy, which caps throughput
well below the server's. --agents sets the world the prompts come from (prompts grow with the village);
--cache-bust puts a unique first line on every prompt, so the A/B against a normal run measures what
vLLM's prefix cache saves. Prefix-cache counters are read from the server's /metrics when it serves them.
"""
import argparse
import json
import random
import re
import time
import urllib.request
from dataclasses import replace
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


def cache_counters(base_url: str) -> dict:
    """vLLM's prefix-cache query and hit token counters, or {} if /metrics isn't reachable."""
    try:
        with urllib.request.urlopen(urllib.request.Request(
                base_url.rstrip("/") + "/metrics", headers={"User-Agent": "asiimov-bench/0.1"}), timeout=10) as r:
            text = r.read().decode()
    except Exception:  # noqa: BLE001 - optional; the proxy or an older server may not serve it
        return {}
    out = {}
    for name in ("queries", "hits"):
        m = re.findall(rf"^vllm:(?:gpu_)?prefix_cache_{name}_total(?:{{[^}}]*}})? ([0-9.e+]+)$", text, re.M)
        if m:
            out[name] = sum(float(x) for x in m)
    return out


def measure(mind: LLMMind, reqs, concurrency: int) -> dict:
    mind.calls = []
    before = cache_counters(mind.base_url)
    parsed = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(concurrency) as ex:
        replies = list(ex.map(mind.decide, reqs))
    wall = time.monotonic() - t0
    after = cache_counters(mind.base_url)
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
            "valid_json_share": round(parsed / len(reqs), 3), "truncated": s["truncated"],
            "prefix_cache_hit_share": round((after["hits"] - before["hits"]) / max(1.0, after["queries"] - before["queries"]), 3)
            if "hits" in after and "hits" in before else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod")
    ap.add_argument("--llm-url")
    ap.add_argument("--llm-model")
    ap.add_argument("--llm-key")
    ap.add_argument("--cost-per-hr", type=float, help="defaults to the pod record's price")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--agents", type=int, default=100, help="village size the prompts come from")
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--world-seed", type=int, default=1,
                    help="another seed gives prompts this server hasn't cached from an earlier bench")
    ap.add_argument("--cache-bust", action="store_true", help="unique first line per prompt: no shared prefixes")
    ap.add_argument("--no-structured", action="store_true",
                    help="no JSON schema per request (diagnostic: grammar compilation is per request on the server)")
    ap.add_argument("--label")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mind, rec = llm_from_args(args)
    mind.structured = not args.no_structured
    cost = args.cost_per_hr or (rec or {}).get("cost_per_hr")
    reqs = collect_prompts(args.agents, args.days, args.world_seed)
    if args.cache_bust:
        reqs = [replace(r, prompt=f"[request {i}]\n" + r.prompt) for i, r in enumerate(reqs)]
    kinds = {}
    for r in reqs:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    # Each level gets prompts no earlier level sent: vLLM caches whole prompts, and a repeat would read as
    # a prefix-cache win the world never gets. The shuffle keeps each sample's kind mix the world's.
    pool = reqs[1:]
    random.Random(0).shuffle(pool)
    sizes = [min(args.n, 64) if c == 1 else args.n for c in args.concurrency]
    if sum(sizes) > len(pool):
        raise SystemExit(f"{sum(sizes)} prompts needed, the world made {len(pool)}; raise --days or lower --n")
    mind.decide(reqs[0])  # warm-up
    results = []
    for c, n in zip(args.concurrency, sizes):
        sample, pool = pool[:n], pool[n:]
        r = measure(mind, sample, c)
        if cost:
            r["usd_per_1k_calls"] = round(float(cost) / (r["req_per_s"] * 3600) * 1000, 4)
        results.append(r)
        print(json.dumps(r), flush=True)
    out = {"model": mind.model, "cost_per_hr": cost, "gpu": (rec or {}).get("gpu"), "label": args.label,
           "agents": args.agents, "world_seed": args.world_seed, "cache_bust": args.cache_bust, "structured": mind.structured, "prompt_mix": kinds, "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
