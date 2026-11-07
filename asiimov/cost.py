"""What minds cost: call load read from recorded logs, and the price per agent-hour and player-hour it implies.

    python -m asiimov.cost load runs/a/events.jsonl[.gz] ... --out load.json
        Mind calls per agent-day by kind, and how bursty they are: calls per tick (p50/p95/p99/max),
        per agent, so worlds of any size compare. Any engine's log; nothing is replayed.
    python -m asiimov.cost price --load load.json --bench bench.json [--bench ...] --out cost.json
        Joins the load with measured server throughput (asiimov.bench) into $ per agent-hour of world
        time, how many agents one GPU keeps in real time at a given time scale, and $ per player-hour.

A player-hour depends on two game-design numbers that don't exist yet: how many agents one player's world
holds, and how fast world time runs against real time. They are inputs here (--agents-per-player,
--world-minutes-per-hour), and the report prints a grid over them rather than one invented figure.
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict

from .sim import TICKS_PER_DAY

MINUTES_PER_TICK = 24 * 60 // TICKS_PER_DAY


def _read(path: str):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        for line in f:
            yield json.loads(line)


def _pct(xs: list[int], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def load(path: str) -> dict:
    """Mind-call load of one log."""
    agents, last, model = 0, 0, None
    kinds, per_tick, per_hour = Counter(), Counter(), Counter()
    tokens = defaultdict(lambda: [0, 0])
    for e in _read(path):
        if e["kind"] == "genesis":
            agents = len(e["data"]["population"]["agents"])
        last = max(last, e["tick"])
        if e["origin"] != "mind":
            continue
        k = e["kind"].removeprefix("mind_")
        kinds[k] += 1
        per_tick[e["tick"]] += 1
        per_hour[(e["tick"] % TICKS_PER_DAY) * MINUTES_PER_TICK // 60] += 1
        tokens[k][0] += e["mind"].get("prompt_tokens") or 0
        tokens[k][1] += e["mind"].get("output_tokens") or 0
        model = model or e["mind"].get("model")
    ticks = last + 1
    days = ticks / TICKS_PER_DAY
    calls = sum(kinds.values())
    series = [per_tick.get(t, 0) for t in range(ticks)]
    per100 = 100 / agents
    return {
        "log": path, "model": model, "agents": agents, "days": round(days, 3), "calls": calls,
        "calls_per_agent_day": round(calls / agents / days, 3),
        "by_kind_per_agent_day": {k: round(n / agents / days, 3) for k, n in sorted(kinds.items())},
        # Engine-side estimates (memory.tokens, ~4 chars/token), for the kind mix only; the bench measures real ones.
        "est_tokens_per_call": {k: {"prompt": tokens[k][0] // kinds[k], "output": tokens[k][1] // kinds[k]}
                                for k in sorted(kinds)},
        "calls_per_tick_per_100_agents": {"mean": round(calls / ticks * per100, 3),
                                          **{k: round(_pct(series, q) * per100, 3)
                                             for k, q in (("p50", .5), ("p95", .95), ("p99", .99))},
                                          "max": round(max(series) * per100, 3)},
        "busy_tick_share": round(sum(1 for x in series if x) / ticks, 3),
        "share_by_hour": {h: round(per_hour[h] / calls, 3) for h in range(24)},
    }


def price(loads: list[dict], benches: list[dict], world_minutes_per_hour: list[float],
          agents_per_player: list[int]) -> dict:
    """$ per agent-hour of world time from each bench's saturated throughput, and what that makes a
    player-hour. Capacity assumes calls can be spread over time (a pipelined engine); the tick-synchronous
    engine today needs headroom for its peak tick, reported as peak_to_mean."""
    calls = sum(l["calls"] for l in loads)
    agent_days = sum(l["agents"] * l["days"] for l in loads)
    cpad = calls / agent_days
    peak = max(l["calls_per_tick_per_100_agents"]["p99"] / l["calls_per_tick_per_100_agents"]["mean"] for l in loads)
    out = {"calls_per_agent_day": round(cpad, 3), "peak_to_mean_p99": round(peak, 2), "servers": []}
    for b in benches:
        best = max(b["results"], key=lambda r: r["req_per_s"])
        usd_call = b["cost_per_hr"] / (best["req_per_s"] * 3600)
        usd_agent_world_hour = usd_call * cpad / 24
        row = {"model": b["model"], "gpu": b.get("gpu"), "cost_per_hr": b["cost_per_hr"],
               "label": b.get("label"), "req_per_s": best["req_per_s"], "concurrency": best["concurrency"],
               "usd_per_1k_calls": round(usd_call * 1000, 5),
               "usd_per_agent_world_hour": round(usd_agent_world_hour, 8), "grid": []}
        for wmph in world_minutes_per_hour:
            world_hours_per_real_hour = wmph / 60
            agent_calls_per_s = cpad / 24 * world_hours_per_real_hour / 3600
            for app in agents_per_player:
                row["grid"].append({
                    "world_minutes_per_real_hour": wmph, "agents_per_player": app,
                    "agents_per_gpu": int(best["req_per_s"] / agent_calls_per_s),
                    "usd_per_player_hour": round(usd_agent_world_hour * world_hours_per_real_hour * app, 5)})
        out["servers"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("load")
    p.add_argument("logs", nargs="+")
    p.add_argument("--out", required=True)
    p = sub.add_parser("price")
    p.add_argument("--load", required=True, help="output of `load`; every log in it counts")
    p.add_argument("--bench", nargs="+", required=True, help="asiimov.bench outputs")
    # 60 = real time; 1440 = a world day per real hour.
    p.add_argument("--world-minutes-per-hour", type=float, nargs="+", default=[60, 240, 1440])
    p.add_argument("--agents-per-player", type=int, nargs="+", default=[100, 300, 1000])
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "load":
        out = [load(p) for p in a.logs]
    else:
        with open(a.load) as f:
            loads = json.load(f)
        benches = []
        for path in a.bench:
            with open(path) as f:
                benches.append(json.load(f))
        out = price(loads, benches, a.world_minutes_per_hour, a.agents_per_player)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
