"""T13 twins on the canonical minds: the driver for test 15 (intervention reach), built on asiimov.fork.

The spec is the shared folder's plan/t13-twins-spec.md, fixed before any output; the bar is frozen v2
(plan/variety-thresholds-v2.json, t15_intervention_reach). This module runs the twins and reads them.
It changes no rule: every twin is a fork.run_twin.

Test 15, per placement:
  P1  the base's own fire (family 3, day 1). Its twin drops the fire at that tick; the reference is the base.
  P2, P3  a fire added at day 3, at P1's hour of day, on families drawn by random.Random("t13") from those
      with at least 3 members other than family 3, without replacement; the reference is the base.
  One intervention twin, then 8 noise twins (salts 1-8) with no intervention and the intervention twin's
  live set forced live from the placement tick. Every twin runs 2 sim-days from the placement tick.
  An agent diverges when, on at least one graded feature in either of the 2 days, its difference from the
  reference exceeds the largest difference among the 8 noise twins. Reach counts diverging agents outside
  the target family. The null reach is the median over the 8 noise twins of each one's reach against the
  other 7. Causal depth is the longest parent chain from the intervention event to an event of a reached
  agent, in the log that holds the intervention (the base for P1, the twin for P2 and P3).

Graded day features (per agent, per day from the placement tick): work, socialize, explore and shop
hours; talks started; receipts by action. Raw divergence (fork.summary) is reported, never counted as reach.

The stub runs every design first as full reruns (serve=False), at $0: its noise twins equal the base, so
its band is zero and every feature difference counts.

python -m asiimov.twins t15 --base runs/t13/stub/full/events.jsonl --arm full --mind stub --out runs/t13/t15-stub
python -m asiimov.twins t15 --base BASE.jsonl.gz --arm full --pod POD_ID --out runs/t13/t15-8b
python -m asiimov.twins t15 --out runs/t13/t15-8b --pod POD_ID   # a second call continues: finished twins are kept
"""
import argparse
import hashlib
import json
import os
import random
import statistics
from collections import Counter, defaultdict

from .events import write_jsonl
from .fork import ForkLeak, read_log, run_twin, summary
from .sim import ACTIVITIES, TICKS_PER_DAY

NOISE_SALTS = tuple(range(1, 9))
WINDOW_DAYS = 2
HOURS = ("work", "socialize", "explore", "shop")
BAR = {"reach_min": 10, "depth_min": 6, "null_inconclusive_at": 5}  # frozen v2; null floor from the T13 spec
assert set(HOURS) <= set(ACTIVITIES)


# ---------------------------------------------------------------------------- placements
def base_fire(base) -> tuple[int, int, dict]:
    """(tick, family, event id) of the base's own fire; the T1n protocol has exactly one."""
    fires = [e for e in base if e.origin == "player" and e.kind == "fire"]
    if len(fires) != 1:
        raise ValueError(f"the base has {len(fires)} fires; test 15's P1 needs exactly one")
    return fires[0].tick, fires[0].data["family"], fires[0].id


def placements(base) -> list[dict]:
    t1, fam1, _ = base_fire(base)
    fams = base[0].data["population"]["families"]
    pool = sorted(f["id"] for f in fams if len(f["members"]) >= 3 and f["id"] != fam1)
    drawn = random.Random("t13").sample(pool, 2)
    t3 = 3 * TICKS_PER_DAY + t1 % TICKS_PER_DAY
    return [{"name": "P1", "kind": "remove", "tick": t1, "family": fam1},
            {"name": "P2", "kind": "add", "tick": t3, "family": drawn[0]},
            {"name": "P3", "kind": "add", "tick": t3, "family": drawn[1]}]


# ---------------------------------------------------------------------------- features
def day_features(events, t0: int, days: int = WINDOW_DAYS) -> dict[int, Counter]:
    """{agent: Counter{(day, feature): value}} over days [t0 + d*TPD, t0 + (d+1)*TPD), d < days.
    Hours from the activity an agent holds each waking tick (wake sets home, sleep ends it), as
    scorecard.extract counts them; talks are the ones it started; receipts are counted by action."""
    end = t0 + days * TICKS_PER_DAY
    cur, out = {}, defaultdict(Counter)
    by_tick = defaultdict(list)
    for e in events:
        if e.tick < end:
            by_tick[e.tick].append(e)
    for tick in range(end):
        for e in by_tick.get(tick, ()):
            a = e.actor
            if a is None:
                continue
            if e.kind == "wake":
                cur[a] = "home"
            elif e.kind == "activity" and a in cur:
                cur[a] = e.data["activity"]
            elif e.kind == "sleep":
                cur.pop(a, None)
            elif tick >= t0 and e.kind == "talk":
                out[a][((tick - t0) // TICKS_PER_DAY, "talks")] += 1
            elif tick >= t0 and e.kind == "receipt":
                out[a][((tick - t0) // TICKS_PER_DAY, f"receipt:{e.data['action']}")] += 1
        if tick >= t0:
            d = (tick - t0) // TICKS_PER_DAY
            for a, act in cur.items():
                if act in HOURS:
                    out[a][(d, f"{act}_h")] += 1 / 6
    return out


def _diff(x: Counter, y: Counter) -> dict:
    return {k: abs(x.get(k, 0) - y.get(k, 0)) for k in set(x) | set(y)}


def diverging(ref: dict, twin: dict, band: list[dict], agents) -> set[int]:
    """Agents for which some cell's |twin - ref| exceeds the largest |noise - ref| among `band`."""
    out = set()
    for a in agents:
        r = ref.get(a, Counter())
        noise = [_diff(n.get(a, Counter()), r) for n in band]
        for k, d in _diff(twin.get(a, Counter()), r).items():
            if d > 1e-9 and d > max((n.get(k, 0) for n in noise), default=0) + 1e-9:
                out.add(a)
                break
    return out


def causal_depth(events, root: int, agents: set[int], until: int) -> int:
    """The longest parent chain (in edges) from event `root` to an event, before tick `until`, whose
    actor is in `agents`. 0 when none descends from the root."""
    depth = {root: 0}
    best = 0
    for e in events:
        if e.id <= root or e.tick >= until:
            continue
        ds = [depth[p] for p in e.parents if p in depth]
        if ds:
            depth[e.id] = max(ds) + 1
            if e.actor in agents:
                best = max(best, depth[e.id])
    return best


# ---------------------------------------------------------------------------- running
def _twin_dir(out, placement, salt):
    return os.path.join(out, placement["name"], "intervention" if salt is None else f"noise-{salt}")


def run_placement(base, live, out: str, p: dict, arm: str | None, serve: bool, workers: int) -> dict:
    """Run (or reload) the intervention twin and its 8 noise twins; returns {salt or None: events}."""
    t0, until = p["tick"], p["tick"] + WINDOW_DAYS * TICKS_PER_DAY
    act = {"kind": "fire", "family": p["family"]}
    logs = {}
    for salt in (None,) + NOISE_SALTS:
        d = _twin_dir(out, p, salt)
        path = os.path.join(d, "events.jsonl")
        if os.path.exists(path):
            logs[salt] = read_log(path)
            continue
        partial = os.path.join(d, "events.partial.jsonl")
        prior = read_log(partial) if os.path.exists(partial) else []
        if salt is None:
            kw = {"drop": [(t0, act)]} if p["kind"] == "remove" else {"player": {t0: [act]}}
            live_set, live_from = (), 0
        else:
            live_set = summary(base, logs[None])["agents_live"]
            live_from, kw = t0, {}
            if not live_set:  # nothing for the noise twin to sample: its world is the base's
                os.makedirs(d, exist_ok=True)
                logs[salt] = [e for e in base if e.tick < until]
                write_jsonl(logs[salt], path)
                continue
        sim, mind = run_twin(base, live, d, live_agents=live_set, live_from=live_from, salt=salt or 0, arm=arm,
                             until=until, workers=workers, prior=prior, serve=serve, **kw)
        write_jsonl(sim.events, path)
        if os.path.exists(partial):
            os.remove(partial)
        rep = summary(base, sim.events, live_set, live_from)
        rep["served"] = dict(mind.counts)
        if hasattr(live, "stats"):
            rep["llm"] = live.stats()
        with open(os.path.join(d, "summary.json"), "w") as f:
            json.dump(rep, f, indent=1)
        logs[salt] = sim.events
    return logs


def read_placement(base, p: dict, logs: dict) -> dict:
    t0, until = p["tick"], p["tick"] + WINDOW_DAYS * TICKS_PER_DAY
    pop = base[0].data["population"]
    family = set(pop["families"][p["family"]]["members"])
    outside = [a["id"] for a in pop["agents"] if a["id"] not in family]
    ref = day_features(base, t0)
    twin = day_features(logs[None], t0)
    noise = [day_features(logs[s], t0) for s in NOISE_SALTS]
    reached = diverging(ref, twin, noise, outside)
    null = [len(diverging(ref, noise[i], noise[:i] + noise[i + 1:], outside)) for i in range(len(noise))]
    if p["kind"] == "remove":
        holder, root = base, base_fire(base)[2]
    else:
        holder = logs[None]
        root = next(e.id for e in holder if e.origin == "player" and e.kind == "fire" and e.tick == t0
                    and e.data["family"] == p["family"])
    raw = summary([e for e in base if e.tick < until], logs[None])
    return {**p, "family_members": sorted(family), "reach": len(reached), "reached": sorted(reached),
            "family_diverging": len(diverging(ref, twin, noise, sorted(family))),
            "null_reach_by_noise_twin": null, "null_reach": statistics.median(null),
            "depth": causal_depth(holder, root, reached, until),
            "raw_agents_diverged": raw["agents_diverged"], "agents_live": raw["agents_live"]}


def grade(rows: list[dict]) -> dict:
    reach = statistics.median(r["reach"] for r in rows)
    depth = statistics.median(r["depth"] for r in rows)
    null = statistics.median(r["null_reach"] for r in rows)
    if null >= BAR["null_inconclusive_at"]:
        verdict = "inconclusive (noise floor)"
    else:
        verdict = "pass" if reach >= BAR["reach_min"] and depth >= BAR["depth_min"] else "fail"
    return {"verdict": verdict, "median_reach": reach, "median_depth": depth, "median_null_reach": null,
            "reach_minus_null": reach - null, "bar": BAR}


def t15(base_path: str, live, out: str, arm: str | None, serve: bool, workers: int) -> dict:
    base = read_log(base_path)
    os.makedirs(out, exist_ok=True)
    ps = placements(base)
    with open(os.path.join(out, "placements.json"), "w") as f:  # the draw, before any twin runs
        json.dump(ps, f, indent=1)
    rows = [read_placement(base, p, run_placement(base, live, out, p, arm, serve, workers)) for p in ps]
    report = {"test": 15, "base": os.path.abspath(base_path), "arm": arm, "model": getattr(live, "model", None),
              **grade(rows), "placements": rows}
    with open(os.path.join(out, "t15.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def main():
    from .minds import StubMind
    from .run import llm_from_args
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["t15"])
    ap.add_argument("--base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="full", help="the scorecard arm the base was run as")
    ap.add_argument("--mind", choices=["stub", "llm"], default="llm")
    ap.add_argument("--pod")
    ap.add_argument("--llm-url")
    ap.add_argument("--llm-model")
    ap.add_argument("--llm-key")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg_path = os.path.join(args.out, "t13.json")
    if os.path.exists(cfg_path):  # continuing: the base, arm and mind are the first call's
        with open(cfg_path) as f:
            cfg = json.load(f)
        if args.base and os.path.abspath(args.base) != cfg["base"]:
            raise SystemExit(f"{args.out} was run on {cfg['base']}")
    else:
        if not args.base:
            raise SystemExit("--base is required")
        with open(args.base, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        cfg = {"base": os.path.abspath(args.base), "base_sha256": sha, "arm": args.arm, "mind": args.mind}
    with open(cfg["base"], "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != cfg["base_sha256"]:
            raise SystemExit(f"{cfg['base']} changed since these twins started")
    live = StubMind() if cfg["mind"] == "stub" else llm_from_args(args)[0]
    if cfg.setdefault("model", live.model) != live.model:
        raise SystemExit(f"these twins were run with model {cfg['model']!r}; refusing {live.model!r}")
    os.makedirs(args.out, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    workers = args.workers or (64 if cfg["mind"] == "llm" else 1)
    try:
        rep = t15(cfg["base"], live, args.out, cfg["arm"], cfg["mind"] != "stub", workers)
    except ForkLeak as e:
        raise SystemExit(f"fork leak: {e}")
    print(json.dumps({k: v for k, v in rep.items() if k != "placements"}, indent=1))
    for r in rep["placements"]:
        print(r["name"], "family", r["family"], "reach", r["reach"], "null", r["null_reach"], "depth", r["depth"],
              "raw", r["raw_agents_diverged"])


if __name__ == "__main__":
    main()
