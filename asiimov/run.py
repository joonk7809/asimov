"""Run a world: python -m asiimov.run --agents 100 --days 3 --seed 1 --out runs/demo"""
import argparse
import json
import os

from .events import write_jsonl
from .memory import render_files
from .metrics import summarize
from .minds import NoisyMind, StubMind
from .sim import TICKS_PER_DAY, Sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mind", choices=["stub", "noisy"], default="stub")
    ap.add_argument("--fire", type=str, default=None, help="player fire as FAMILY@DAY, e.g. 3@1")
    ap.add_argument("--out", default="runs/demo")
    args = ap.parse_args()
    mind = StubMind() if args.mind == "stub" else NoisyMind()
    sim = Sim(args.seed, args.agents, mind)
    player = {}
    if args.fire:
        fam, day = (int(x) for x in args.fire.split("@"))
        player[day * TICKS_PER_DAY + 60] = [{"kind": "fire", "family": fam}]
    sim.run(args.days, player)
    os.makedirs(args.out, exist_ok=True)
    write_jsonl(sim.events, os.path.join(args.out, "events.jsonl"))
    for a in sim.state["agents"]:
        d = os.path.join(args.out, "agents", f"{a['id']:03d}")
        os.makedirs(d, exist_ok=True)
        for name, text in render_files(a, sim.state).items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text + "\n")
    summary = summarize(sim.events, sim.metrics["activity"], args.agents)
    summary["final_hash"] = sim.state_hash()
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
