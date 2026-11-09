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
import math
import os
import random
import statistics
from collections import Counter, defaultdict

from .events import write_jsonl
from .fork import ForkLeak, end_tick as end_tick_of, read_log, run_twin, summary
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


# ---------------------------------------------------------------------------- test 8: soul swap
SOUL_KEYS = ("traits", "values", "need_weights", "cognition")  # the soul's params; body, rhythm and circumstance stay
T8_SALT = 9


def swap_pairs(pop: dict) -> list[tuple[int, int]]:
    """Pairs with the same role, age band (age // 20, as test 11's matching) and family size: within each
    cell, ids in order, paired consecutively; an odd one out is left unpaired."""
    size = {f["id"]: len(f["members"]) for f in pop["families"]}
    cell = defaultdict(list)
    for a in sorted(pop["agents"], key=lambda a: a["id"]):
        cell[(a["role"], a["age"] // 20, size[a["family"]])].append(a["id"])
    return sorted((ids[i], ids[i + 1]) for ids in cell.values() for i in range(0, len(ids) - 1, 2))


def swapped_genesis(pop: dict, pairs) -> dict:
    out = json.loads(json.dumps(pop))
    for x, y in pairs:
        ax, ay = out["agents"][x], out["agents"][y]
        for k in SOUL_KEYS:
            ax[k], ay[k] = ay[k], ax[k]
    return out


def swapped_souls(pop: dict, pairs) -> dict[int, str]:
    """The primary swap (Steering's ruling, 2026-11-09): each agent's soul text is rendered from its partner's
    traits and values, anchors included, under its own name, age, role and rhythm lines (those are the
    body's). Params stay with the body, so the habit layer and the stub read the original agent."""
    from .memory import render_soul
    ag = pop["agents"]
    out = {}
    for x, y in pairs:
        for body, donor in ((x, y), (y, x)):
            out[body] = render_soul({**ag[body], "traits": ag[donor]["traits"], "values": ag[donor]["values"]})
    return out


def _period_stats(events, features, skip):
    from .scorecard import extract, graded
    g = graded(extract(events), skip)
    means = {a: {f: statistics.mean(p[f] for p in ps) for f in features} for a, ps in g.items() if ps}
    return g, means


def read_t8(base, swap, noise, pairs, features, skip, boot=2000) -> dict:
    """The frozen v2 follow index. A = the lower id of a pair, B its partner. A cell (pair x feature) is
    kept when |B_orig - A_orig| exceeds 1.96 * sqrt(2) * the noise twin's within-agent SE for that feature
    (the mean within-agent SD over the swapped agents with >= 2 graded periods, over sqrt of their mean
    period count, as results/tools/power_v2.py models it). follow = mean clip((A_swap - A)/(B - A));
    the noise index is the same with the noise twin's A. CI: percentile bootstrap over kept cells of
    follow - noise, random.Random("t13-t8")."""
    _, orig = _period_stats(base, features, skip)
    _, sw = _period_stats(swap, features, skip)
    gn, nz = _period_stats(noise, features, skip)
    swapped = {a for p in pairs for a in p}
    band = {}
    for f in features:
        sds = [statistics.stdev([p[f] for p in gn[a]]) for a in swapped if len(gn.get(a, [])) >= 2]
        ks = [len(gn[a]) for a in swapped if len(gn.get(a, [])) >= 2]
        band[f] = 1.96 * math.sqrt(2) * statistics.mean(sds) / math.sqrt(statistics.mean(ks)) if sds else 0.0
    clip = lambda x: max(-1.0, min(2.0, x))  # noqa: E731
    cells = []
    for a, b in pairs:
        if not all(x in orig and x in sw and x in nz for x in (a, b)):
            continue
        for f in features:
            A, B = orig[a][f], orig[b][f]
            if abs(B - A) > band[f]:
                cells.append({"pair": [a, b], "feature": f, "follow": clip((sw[a][f] - A) / (B - A)),
                              "noise": clip((nz[a][f] - A) / (B - A)),
                              "reverse": clip((sw[b][f] - B) / (A - B))})  # B's side, a diagnostic
    th = {"min_pairs": 20, "min_kept_cells": 40, "follow_index_min": 0.50, "ci_lower_above_noise_index_plus": 0.20}
    out = {"pairs": len(pairs), "kept_cells": len(cells), "band": band, "bar": th}
    if not cells:
        return {**out, "verdict": "inconclusive (no kept cells)"}
    follow = statistics.mean(c["follow"] for c in cells)
    noise_i = statistics.mean(c["noise"] for c in cells)
    rng = random.Random("t13-t8")
    diffs = []
    for _ in range(boot):
        idx = [rng.randrange(len(cells)) for _ in cells]
        diffs.append(statistics.mean(cells[i]["follow"] - cells[i]["noise"] for i in idx))
    diffs.sort()
    lo = diffs[int(0.025 * boot)]
    if len(pairs) < th["min_pairs"] or len(cells) < th["min_kept_cells"]:
        verdict = "inconclusive (too few pairs or cells)"
    else:
        verdict = "pass" if follow >= th["follow_index_min"] and lo > th["ci_lower_above_noise_index_plus"] else "fail"
    by_feature = {f: statistics.mean(c["follow"] for c in cells if c["feature"] == f)
                  for f in features if any(c["feature"] == f for c in cells)}
    return {**out, "verdict": verdict, "follow_index": follow, "noise_index": noise_i, "ci95_lower_follow_minus_noise": lo,
            "reverse_index": statistics.mean(c["reverse"] for c in cells), "follow_by_feature": by_feature,
            "cells": cells}


def _run_or_load(base, live, d, arm, serve, workers, **kw):
    path = os.path.join(d, "events.jsonl")
    if os.path.exists(path):
        return read_log(path)
    partial = os.path.join(d, "events.partial.jsonl")
    prior = read_log(partial) if os.path.exists(partial) else []
    prior = kw.pop("prior", None) or prior
    sim, mind = run_twin(base, live, d, arm=arm, workers=workers, prior=prior, serve=serve, **kw)
    write_jsonl(sim.events, path)
    if os.path.exists(partial):
        os.remove(partial)
    rep = summary(base, sim.events, kw.get("live_agents", ()), kw.get("live_from", 0))
    rep["served"] = dict(mind.counts)
    if hasattr(live, "stats"):
        rep["llm"] = live.stats()
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return sim.events


def t8(base_path: str, live, out: str, arm: str | None, serve: bool, workers: int) -> dict:
    from .authorship import replay_authorship, summarize
    from .scorecard import load_thresholds
    proto = load_thresholds()[0]["protocol"]
    base = read_log(base_path)
    pop = base[0].data["population"]
    pairs = swap_pairs(pop)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "pairs.json"), "w") as f:  # before any twin runs
        json.dump(pairs, f)
    swapped = sorted(a for p in pairs for a in p)
    swap = _run_or_load(base, live, os.path.join(out, "swap-text"), arm, serve, workers,
                        souls=swapped_souls(pop, pairs))
    full = _run_or_load(base, live, os.path.join(out, "swap-full"), arm, serve, workers,
                        genesis=swapped_genesis(pop, pairs))
    noise = _run_or_load(base, live, os.path.join(out, f"noise-{T8_SALT}"), arm, serve, workers,
                         live_agents=swapped, live_from=0, salt=T8_SALT)
    feats, skip = proto["graded_features"], proto["skip_first_waking_periods"]
    rep = read_t8(base, swap, noise, pairs, feats, skip)
    beside = read_t8(base, full, noise, pairs, feats, skip)
    beside = {k: beside.get(k) for k in ("kept_cells", "follow_index", "noise_index", "ci95_lower_follow_minus_noise",
                                         "follow_by_feature")}
    # Test 7 on the base (its own world; the text swap's twin log replays only with its souls).
    t7 = summarize(replay_authorship(base, arm or "full"))["test7"]
    report = {"test": 8, "base": os.path.abspath(base_path), "arm": arm, "model": getattr(live, "model", None),
              "swap": "soul text only (Steering 2026-11-09)", "test7_on_base": t7,
              "full_swap_descriptive": beside, **rep}
    with open(os.path.join(out, "t8.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


# ---------------------------------------------------------------------------- test 9: memory after a theft
T9_DAYS = 3
T9_MIN_THEFTS = 20
T9_MAX_ATTEMPTS = 80  # a bound on paid attempts; short of 20 thefts by then, the test is inconclusive


def theft_candidates(base, arm: str | None, last_tick: int) -> list[tuple[int, int, int]]:
    """(tick, thief, victim) at the start of each tick before `last_tick`: both awake, at the same place
    (not walking), the victim carrying food, the thief awake before its bedtime and out of cooldown.
    Read by replaying the base, every event checked."""
    from .fork import build_sim
    from .minds import ReplayMind
    sim = build_sim(base, ReplayMind([e for e in base if e.origin == "mind"]), arm)
    sim.expected = base
    player = defaultdict(list)
    for e in base:
        if e.origin == "player":
            player[e.tick].append({"kind": e.kind, **e.data})
    out = []
    while sim.state["tick"] < last_tick:
        t = sim.state["tick"]
        here = defaultdict(list)
        for a in sim.state["agents"]:
            if not a["asleep"] and a["place"] is not None and not a["route"]:
                here[a["place"]].append(a)
        for people in here.values():
            for th in people:
                if t >= th["sleep_tick"] or t < th["cooldown_until"]:
                    continue
                out += [(t, th["id"], v["id"]) for v in people if v is not th and v["inv"]["food"] > 0]
        sim.step(player.get(t, []))
    return out


def theft_acts(t: int, thief: int, victim: int, action: str) -> dict:
    act = {"kind": "act", "action": action} | ({"target": victim} if action == "steal" else {})
    return {t: [{"kind": "possess", "agent": thief}, act], t + 1: [{"kind": "release"}]}


def _mind_authored(e, byid) -> bool:
    """The act descends, within two parent steps, from one of its actor's own mind events."""
    p = e
    for _ in range(2):
        if not p.parents:
            return False
        p = byid[p.parents[0]]
        if p.origin == "mind" and p.actor == e.actor:
            return True
    return False


def interactions(events, actor: int, other: int, start: int, end: int) -> int:
    """Mind-authored person-directed acts of `actor` toward `other` in [start, end), as intents.summarize
    counts person-directed acts (a meet from seek, a gift that went through, an ask, buy or steal that
    reached its target), with the possessed body's own events left out (drop_possessed)."""
    from .events import drop_possessed
    events, _ = drop_possessed(events)
    byid = {e.id: e for e in events}
    n = 0
    for e in events:
        if e.actor != actor or not start <= e.tick < end:
            continue
        to = None
        if e.kind == "meet":
            to = e.data["with"]
        elif e.kind == "receipt" and e.data["action"] in ("give_food", "give_coin") and e.data["why"] == "gave":
            to = e.data["to"]
        elif e.kind == "receipt" and e.data["action"] in ("ask_kin", "ask_food", "buy_from", "steal") and "roll" in e.data:
            to = e.data.get("victim", e.data.get("to"))
        n += to == other and _mind_authored(e, byid)
    return n


def stole(events, t: int, thief: int) -> bool:
    """Whether the injected steal at tick t took food (not caught, not empty, not refused)."""
    acts = {e.id for e in events if e.tick == t and e.kind == "act" and e.origin == "player"}
    return any(e.kind == "receipt" and e.parents and e.parents[0] in acts and e.data["action"] == "steal"
               and e.data["ok"] and e.data.get("food", 1) > 0 for e in events if e.tick == t)


def sign_test(down: int, n: int) -> float:
    return sum(math.comb(n, k) for k in range(down, n + 1)) / 2 ** n if n else 1.0


def read_t9(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["stole"]]
    down = sum(r["theft"] < r["control"] for r in ok)
    ties = sum(r["theft"] == r["control"] for r in ok)
    tot_c, tot_t = sum(r["control"] for r in ok), sum(r["theft"] for r in ok)
    rate = tot_c / len(ok) if ok else 0.0
    p = sign_test(down, len(ok) - ties)
    drop = (tot_c - tot_t) / tot_c if tot_c else None
    pre = (len(ok) >= 20 and rate >= 4) or (len(ok) >= 40 and rate >= 2)
    if len(ok) < T9_MIN_THEFTS or not pre:
        verdict = "inconclusive" + (" (too few thefts)" if len(ok) < T9_MIN_THEFTS else " (precondition)")
    else:
        verdict = "pass" if p < 0.05 and drop is not None and drop >= 0.30 else "fail"
    return {"verdict": verdict, "attempts": len(rows), "thefts": len(ok), "control_rate": rate, "drop": drop,
            "down": down, "ties": ties, "sign_test_p": p,
            "bar": {"min_drop": 0.30, "sign_test_p": 0.05, "precondition": [[20, 4], [40, 2]]}}


def t9(base_path: str, live, out: str, arm: str | None, serve: bool, workers: int, precheck: bool = False) -> dict:
    base = read_log(base_path)
    os.makedirs(out, exist_ok=True)
    cpath = os.path.join(out, "draw.json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            draw = [tuple(x) for x in json.load(f)]
    else:
        last = end_tick_of(base) - T9_DAYS * TICKS_PER_DAY
        cands = theft_candidates(base, arm, last)
        rng = random.Random("t13-theft")
        rng.shuffle(cands)
        seen, draw = set(), []
        for t, th, v in cands:  # one theft per (thief, victim) pair
            if (th, v) not in seen:
                seen.add((th, v))
                draw.append((t, th, v))
        with open(cpath, "w") as f:  # before any twin runs
            json.dump(draw[:T9_MAX_ATTEMPTS], f)
    # The base stands in for the control twin (no possession): the victim's mind-authored acts toward the
    # thief over the 3 days after each drawn tick. Far below the precondition's rate, test 9 can only be
    # inconclusive, and that is known before any twin is paid for.
    pre = [interactions(base, v, th, t + 1, t + T9_DAYS * TICKS_PER_DAY) for t, th, v in draw[:T9_MAX_ATTEMPTS]]
    base_rate = statistics.mean(pre) if pre else 0.0
    with open(os.path.join(out, "precheck.json"), "w") as f:
        json.dump({"base_rate_over_draws": base_rate, "draws": len(pre), "precondition": [[20, 4], [40, 2]]}, f)
    if precheck:
        return {"test": 9, "precheck": True, "base_rate_over_draws": base_rate, "draws": len(pre)}
    rows = []
    for i, (t, th, v) in enumerate(draw[:T9_MAX_ATTEMPTS]):
        if sum(r["stole"] for r in rows) >= T9_MIN_THEFTS:
            break
        d = os.path.join(out, f"{i:03d}")
        # The theft tick alone first: a caught or empty steal stops there, before its 3 days are paid for.
        probe = _run_or_load(base, live, os.path.join(d, "probe"), arm, serve, workers,
                             player=theft_acts(t, th, v, "steal"), until=t + 1)
        row = {"i": i, "tick": t, "thief": th, "victim": v, "stole": stole(probe, t, th)}
        if row["stole"]:
            end = t + T9_DAYS * TICKS_PER_DAY
            theft = _run_or_load(base, live, os.path.join(d, "theft"), arm, serve, workers,
                                 player=theft_acts(t, th, v, "steal"), until=end, prior=probe)
            control = _run_or_load(base, live, os.path.join(d, "control"), arm, serve, workers,
                                   player=theft_acts(t, th, v, "wait"), until=end)
            row["theft"] = interactions(theft, v, th, t + 1, end)
            row["control"] = interactions(control, v, th, t + 1, end)
        rows.append(row)
    report = {"test": 9, "base": os.path.abspath(base_path), "arm": arm, "model": getattr(live, "model", None),
              "base_rate_over_draws": base_rate, **read_t9(rows), "rows": rows}
    with open(os.path.join(out, "t9.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def main():
    from .minds import StubMind
    from .run import llm_from_args
    ap = argparse.ArgumentParser()
    ap.add_argument("test", choices=["t15", "t8", "t9"])
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
    ap.add_argument("--precheck", action="store_true", help="t9: draw the thefts and read the base's rate; run no twin")
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
        run = {"t15": t15, "t8": t8, "t9": t9}[args.test]
        kw = {"precheck": True} if args.precheck and args.test == "t9" else {}
        rep = run(cfg["base"], live, args.out, cfg["arm"], cfg["mind"] != "stub", workers, **kw)
    except ForkLeak as e:
        raise SystemExit(f"fork leak: {e}")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("placements", "cells", "rows")}, indent=1))
    for r in rep.get("placements", []):
        print(r["name"], "family", r["family"], "reach", r["reach"], "null", r["null_reach"], "depth", r["depth"],
              "raw", r["raw_agents_diverged"])


if __name__ == "__main__":
    main()
