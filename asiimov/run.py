"""Run a world: python -m asiimov.run --agents 100 --days 3 --seed 1 --out runs/demo"""
import argparse
import json
import os
import time

from .events import write_jsonl
from .memory import render_files
from .metrics import summarize
from .llm import LLMMind
from .minds import NoisyMind, StubMind
from .sim import TICKS_PER_DAY, Sim


POD_DIR = os.path.join(os.path.dirname(__file__), "..", "runs", "pods")


def llm_from_args(args) -> tuple[LLMMind, dict | None]:
    """Build an LLMMind from --pod (a record written by scripts/runpod_pod.py) or explicit flags."""
    rec = None
    if getattr(args, "pod", None):
        with open(os.path.join(POD_DIR, f"{args.pod}.json")) as f:
            rec = json.load(f)
    url = args.llm_url or (rec or {}).get("url")
    model = args.llm_model or (rec or {}).get("model")
    key = args.llm_key or (rec or {}).get("api_key")
    if not url or not model:
        raise SystemExit("LLM mind needs --pod, or --llm-url and --llm-model")
    return LLMMind(url, model, api_key=key, temperature=getattr(args, "temperature", 0.7)), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mind", choices=["stub", "noisy", "llm"], default="stub")
    ap.add_argument("--fire", type=str, default=None, help="player fire as FAMILY@DAY, e.g. 3@1")
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--pod", help="pod id from scripts/runpod_pod.py (url, model and key)")
    ap.add_argument("--llm-url")
    ap.add_argument("--llm-model")
    ap.add_argument("--llm-key")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()
    rec = None
    if args.mind == "llm":
        mind, rec = llm_from_args(args)
    else:
        mind = StubMind() if args.mind == "stub" else NoisyMind()
    sim = Sim(args.seed, args.agents, mind)
    t0 = time.monotonic()
    player = {}
    if args.fire:
        fam, day = (int(x) for x in args.fire.split("@"))
        player[day * TICKS_PER_DAY + 60] = [{"kind": "fire", "family": fam}]
    os.makedirs(args.out, exist_ok=True)
    try:
        for day in range(args.days):
            sim.run(1, player)
            extra = f", model wall {mind.stats().get('wall_s_in_model', 0)}s" if args.mind == "llm" else ""
            print(f"day {day} done at {time.monotonic() - t0:.0f}s, {len(sim.events)} events{extra}", flush=True)
    except BaseException:
        # A paid LLM run that dies mid-way (server gone, Ctrl-C) keeps every recorded mind output.
        # The last tick may be incomplete, so this file is for inspection, not for replay().
        write_jsonl(sim.events, os.path.join(args.out, "events.partial.jsonl"))
        print(f"run aborted at tick {sim.state['tick']}; wrote events.partial.jsonl", flush=True)
        raise
    write_jsonl(sim.events, os.path.join(args.out, "events.jsonl"))
    for a in sim.state["agents"]:
        d = os.path.join(args.out, "agents", f"{a['id']:03d}")
        os.makedirs(d, exist_ok=True)
        for name, text in render_files(a, sim.state).items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text + "\n")
    summary = summarize(sim.events, sim.metrics["activity"], args.agents)
    summary["final_hash"] = sim.state_hash()
    summary["wall_s"] = round(time.monotonic() - t0, 1)
    if args.mind == "llm":
        summary["llm"] = mind.stats()
        cost = (rec or {}).get("cost_per_hr")
        if cost:
            summary["llm"]["cost_per_hr"] = cost
            summary["llm"]["usd_this_run"] = round(float(cost) * summary["wall_s"] / 3600, 3)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
