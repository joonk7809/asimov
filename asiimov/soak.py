"""Long unattended run: a world run through the crash-safe path for weeks of sim time, with every
invariant and both ledgers checked after every tick, then replayed from disk without the mind.

Passes when the run finishes without an exception, no tick has a violation (asiimov/invariants.py),
and replaying events.jsonl from disk reproduces the live final hash. It keeps going past violations
so one run reports all of them (the first 50 are kept in full). Per sim-day it also records wall time,
log growth, snapshot size and peak memory, so a slowdown or a leak shows before it kills a long run,
and a few health numbers (hunger, money, stock) that say whether the world is still a world.

python -m asiimov.soak --agents 100 --days 24 --seed 1 --out runs/soak    # stub, CPU, $0
"""
import argparse
import json
import os
import resource
import sys
import time

from . import persist
from .events import read_jsonl
from .invariants import Checker
from .minds import NoisyMind, StubMind
from .replay import replay
from .sim import TICKS_PER_DAY

KEEP = 50


def health(sim) -> dict:
    ag = sim.state["agents"]
    return {"hunger_mean": sum(a["needs"]["hunger"] for a in ag) // len(ag),
            "hunger_max_agents": sum(1 for a in ag if a["needs"]["hunger"] >= 1000),
            "coin_total": sum(a["inv"]["coin"] for a in ag),
            "food_held": sum(a["inv"]["food"] for a in ag), "market_food": sim.state["market_food"],
            "broke_and_foodless": sum(1 for a in ag if a["inv"]["coin"] < 3 and a["inv"]["food"] == 0)}


def soak(out: str, seed: int, agents: int, days: int, mind_kind: str = "stub", player: dict | None = None) -> dict:
    mind = StubMind() if mind_kind == "stub" else NoisyMind()
    t0 = time.monotonic()
    run = persist.start(out, seed, agents, mind, player=player or {}, days=days, mind_kind=mind_kind)
    sim, checker = run.sim, Checker(run.sim)
    res = {"seed": seed, "agents": agents, "days": days, "mind": mind_kind, "violations": 0,
           "first_violations": [], "per_day": [], "error": None}
    try:
        while sim.state["tick"] < days * TICKS_PER_DAY:
            tick, day_t, n0 = sim.state["tick"], time.monotonic(), len(sim.events)
            for _ in range(TICKS_PER_DAY):
                checker.before(run.player.get(sim.state["tick"], []))
                run.advance(sim.state["tick"] + 1)
                bad = checker.after()
                res["violations"] += len(bad)
                res["first_violations"] += bad[:KEEP - len(res["first_violations"])]
            snap = os.path.join(out, "snapshots", f"{sim.state['tick']:08d}.json")
            res["per_day"].append({"day": tick // TICKS_PER_DAY, "wall_s": round(time.monotonic() - day_t, 2),
                                   "events": len(sim.events) - n0,
                                   "snapshot_kb": os.path.getsize(snap) // 1024 if os.path.exists(snap) else None,
                                   "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
                                   **health(sim)})
            print(json.dumps(res["per_day"][-1]), flush=True)
        run.finish()
    except Exception as e:  # an unattended run reports its crash rather than losing the report
        res["error"] = f"{type(e).__name__} at tick {sim.state['tick']}: {e}"
    res["live_hash"] = sim.state_hash()
    res["events"] = len(sim.events)
    res["wall_s"] = round(time.monotonic() - t0, 1)
    if res["error"] is None:
        r0 = time.monotonic()
        try:
            res["replay_hash"] = replay(read_jsonl(os.path.join(out, "events.jsonl"))).state_hash()
        except Exception as e:
            res["replay_hash"], res["error"] = None, f"replay: {type(e).__name__}: {e}"
        res["replay_wall_s"] = round(time.monotonic() - r0, 1)
    res["replay_matches"] = res.get("replay_hash") == res["live_hash"]
    res["ok"] = res["error"] is None and res["violations"] == 0 and res["replay_matches"]
    with open(os.path.join(out, "soak.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--days", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mind", choices=["stub", "noisy"], default="stub")
    ap.add_argument("--fire", action="append", default=[], help="a fire as FAMILY@DAY (repeatable)")
    args = ap.parse_args()
    player = {}
    for spec in args.fire:
        fam, day = (int(x) for x in spec.split("@"))
        player.setdefault(day * TICKS_PER_DAY + 60, []).append({"kind": "fire", "family": fam})
    res = soak(args.out, args.seed, args.agents, args.days, args.mind, player)
    print(json.dumps({k: v for k, v in res.items() if k != "per_day"}, indent=1))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
