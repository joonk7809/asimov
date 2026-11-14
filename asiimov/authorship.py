"""Who authors each waking tick: the mind (a plan block or a deliberated override), the habit code, or the world.

A replay instrument, like the probe's ProbeSim: it replays a recorded log, verifying every event, and
reads authorship off the state `Sim._act` decides from. No rule or log changes, so it works on every
log of this engine. Authors (test 7, plan/variety-scorecard-v2.md):

  deliberate  an active override: a deliberated one-off step (go_market, seek, work, ...)   mind
  intent      the current step of an intent the mind set (engine 7)                       mind
  plan        a block of the agent's morning plan covers the hour             mind
  habit       neither: `Sim._habit`, the no-LLM needs-and-traits layer        code
  forced      the wake tick, which the world sets to "home" before any choice, and  world
              a bedtime walk home (the map: an agent away from home at bedtime), and a one-off step
              a rule set (the sick-neighbor channel's answer, trade-1's supply run)

Raw mind share counts every mind-authored tick. Whole-day plans (PR #8) push it to about 1 even when a
plan only restates the habits, so the instrument also records, on every tick, the activity the habit
layer would have picked then (the same 6-tick cadence `_act` uses), and a mind tick is **divergent**
when its activity differs from that pick. Divergent mind share is the share of waking ticks where the
mind changed what the agent did.

Some activity labels differ only in name: "rest" and "home" both put the agent at home, and the rules
read an activity only through its place and whether it is work (production, who can talk). So a mind
tick is also scored **effective** when it changes the place or work-or-not against the habit's pick.
Effective mind share is the share of waking ticks where the mind changed what the world sees.

The habit pick is counterfactual inside the world the mind made (its needs, coin and plan), not a twin
run: it says how often the mind overrode the habit layer, not how different the agent's life would be.

python -m asiimov.authorship --log results/raw/t1e-4b/full/events.jsonl.gz --arm full --out runs/authorship/4b-full
"""
import argparse
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict

from .memory import render_soul
from .minds import ReplayDivergence
from .probe import ProbeSim, read_log
from .scorecard import ARMS, RESPONSE_ARMS, labeled_population, load_thresholds

MIND_AUTHORS = ("deliberate", "intent", "plan")
AUTHORS = MIND_AUTHORS + ("habit", "forced")
V2_THRESHOLDS = os.path.join(os.path.dirname(__file__), "..", "plan", "variety-thresholds-v2.json")


class AuthorshipSim(ProbeSim):
    """Replays a log (any scorecard arm) and records (tick, agent, author, activity, habit pick) per waking tick."""

    def __init__(self, events, arm: str = "full", prompt_token_offset: int = 0):
        super().__init__(events, arm, set(), prompt_token_offset)
        if arm == "flat_params":  # the arm's prompts carry each agent's real soul (scorecard.run_arm)
            pop = labeled_population(self.seed, len(self.state["agents"]))
            self.souls = {a["id"]: render_soul(a) for a in pop["agents"]}
        # (tick, agent, author, activity, habit pick); the wake tick has no habit pick
        self.records: list[tuple[int, int, str, str, str | None]] = []
        # (tick, agent) -> what the rules read of this tick, and of the habit's pick: ((place, is work), ...).
        # Per tick, from where the agent actually is: a step's place (visit goes wherever its target is)
        # isn't a function of the activity label (PR #19 review).
        self.effect = {}

    def _wake(self, a, occupancy):
        super()._wake(a, occupancy)
        self.records.append((self.state["tick"], a["id"], "forced", a["activity"], None))

    def _act(self, a):
        tick, hour = self.state["tick"], self._hour()
        # Precedence, read off state independently of Sim._running: override, intent step, plan, habit.
        step = (a["override"] if a["override"] else
                a["intent"]["steps"][a["intent"]["i"]] if a["intent"] else None)
        # What the habit layer picks this tick, by _act's own rule, without touching state.
        habit = self._habit(a) if tick % 6 == 0 or a["habit"] is None else a["habit"]
        block = None
        for b in a["plan"]:  # the last covering block wins, as in _act
            if b["start_h"] <= hour < b["end_h"]:
                block = b
        expected = (self._step_activity(a, step)[0] if step else block["activity"] if block else habit)
        author = ("deliberate" if a["override"] else "intent") if step else "plan" if block else "habit"
        if a["override"] and self.events[a["override"]["event"]].origin == "rule":
            author = "forced"  # a rule's one-off step (the neighbor channel, trade-1), not a mind's
        super()._act(a)
        if a["activity"] != expected or (not step and a["habit"] != habit):
            raise AssertionError(f"authorship disagrees with _act at tick {tick}, agent {a['id']}")
        self.effect[(tick, a["id"])] = ((self._dest(a), a["activity"] == "work"), (self._place(a, habit), habit == "work"))
        self.records.append((tick, a["id"], author, a["activity"], habit))

    def _bedtime_walk(self, a):
        super()._bedtime_walk(a)
        self.records.append((self.state["tick"], a["id"], "forced", a["activity"], None))


def replay_authorship(events, arm: str = "full", prompt_token_offset: int = 0) -> AuthorshipSim:
    """Replay the whole log. Logs from asiimov.run end with `end`; scorecard arm logs end at a checkpoint."""
    sim = AuthorshipSim(events, arm, prompt_token_offset)
    last = events[-1]
    if last.kind == "end":
        sim.run_until(last.tick, sim.player)
        sim.finish()
    else:
        sim.run_until(last.tick + 1, sim.player)
    if len(sim.events) != len(events):
        raise ReplayDivergence(f"replay produced {len(sim.events)} events, log has {len(events)}")
    return sim


def _gate_stats(shares: list[float]) -> dict:
    s = sorted(shares)
    n = len(s)
    median = (s[(n - 1) // 2] + s[n // 2]) / 2
    return {"median": round(median, 4), "mean": round(sum(s) / n, 4),
            "share_of_agents_below_0_20": round(sum(x < 0.20 for x in s) / n, 4)}


def summarize(sim: AuthorshipSim) -> dict:
    per = defaultdict(lambda: {"ticks": Counter(), "divergent": Counter(), "effective": Counter()})
    pairs = Counter()  # (plan activity, habit pick) over plan ticks
    for tick, aid, author, activity, habit in sim.records:
        p = per[aid]
        p["ticks"][author] += 1
        if author in MIND_AUTHORS and activity != habit:
            p["divergent"][author] += 1
            if sim.effect[(tick, aid)][0] != sim.effect[(tick, aid)][1]:
                p["effective"][author] += 1
        if author == "plan":
            pairs[(activity, habit)] += 1
    th, sha = load_thresholds(V2_THRESHOLDS)
    t7 = th["t7_mind_share"]
    shares = {k: [] for k in ("test7", "mind", "divergent_mind", "effective_mind")}
    agents = []
    for aid in sorted(per):
        p = per[aid]
        waking = sum(p["ticks"].values())
        row = {"agent": aid, "waking_ticks": waking, "ticks": {k: p["ticks"][k] for k in AUTHORS},
               "divergent": {k: p["divergent"][k] for k in MIND_AUTHORS},
               "effective": {k: p["effective"][k] for k in MIND_AUTHORS},
               "test7_share": sum(p["ticks"][k] for k in t7["authors_counted"]) / waking,
               "mind_share": sum(p["ticks"][k] for k in MIND_AUTHORS) / waking,
               "divergent_mind_share": sum(p["divergent"].values()) / waking,
               "effective_mind_share": sum(p["effective"].values()) / waking}
        for k in shares:
            shares[k].append(row[f"{k}_share"])
        agents.append(row)
    tot = {k: Counter() for k in ("ticks", "divergent", "effective")}
    for p in per.values():
        for k in tot:
            tot[k].update(p[k])
    waking = sum(tot["ticks"].values())
    rate = lambda k, author: round(tot[k][author] / tot["ticks"][author], 4) if tot["ticks"][author] else None  # noqa: E731
    test7 = _gate_stats(shares["test7"])
    return {
        "agents": len(agents),
        "waking_ticks": waking,
        "author_share": {k: round(tot["ticks"][k] / waking, 4) for k in AUTHORS},
        # Of the ticks a plan block authored, how often it changed the habit's pick: 0 means plans copy habits.
        "divergence_rate": {a: rate("divergent", a) for a in MIND_AUTHORS},
        "effective_rate": {a: rate("effective", a) for a in MIND_AUTHORS},
        "test7": {**test7, "authors_counted": t7["authors_counted"],
                  "pass": (test7["median"] >= t7["median_share_min"]
                           and test7["share_of_agents_below_0_20"] <= t7["max_share_of_agents_below_0_20"]),
                  "thresholds_sha256": sha},
        "mind_share": _gate_stats(shares["mind"]),
        "divergent_mind_share": _gate_stats(shares["divergent_mind"]),
        "effective_mind_share": _gate_stats(shares["effective_mind"]),
        "plan_vs_habit": [{"plan": a, "habit": h, "ticks": n} for (a, h), n in sorted(pairs.items(), key=lambda x: -x[1])],
        "per_agent": agents,
    }


def render_md(rep: dict) -> str:
    t7 = rep["test7"]
    rows = [(f"Test 7 ({' + '.join(t7['authors_counted'])})", t7), ("Mind (plan + deliberate)", rep["mind_share"]),
            ("Divergent mind (activity differs from the habit pick)", rep["divergent_mind_share"]),
            ("Effective mind (place or work differs from the habit pick)", rep["effective_mind_share"])]
    lines = [f"# Authorship: {rep.get('label') or rep['log']} ({rep['arm']})", "",
             f"{rep['agents']} agents, {rep['waking_ticks']} waking ticks. Author shares: "
             + ", ".join(f"{k} {v:.3f}" for k, v in rep["author_share"].items()) + ".", "",
             "| Share per agent | median | mean | agents below 0.20 |", "|---|---|---|---|"]
    lines += [f"| {name} | {s['median']:.3f} | {s['mean']:.3f} | {s['share_of_agents_below_0_20']:.2f} |"
              for name, s in rows]
    lines += ["", f"Test 7 gate (raw, as frozen): {'pass' if t7['pass'] else 'fail'}. Plan ticks that differ from "
              f"the habit pick: {rep['divergence_rate']['plan']} (effective: {rep['effective_rate']['plan']}).", "",
              "Top plan-vs-habit pairs (plan activity / habit pick: ticks): "
              + "; ".join(f"{p['plan']}/{p['habit']}: {p['ticks']}" for p in rep["plan_vs_habit"][:8]) + "."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", required=True)
    ap.add_argument("--arm", choices=ARMS + RESPONSE_ARMS, default="full")
    ap.add_argument("--prompt-token-offset", type=int, default=0, help="see ProbeSim")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.log, "rb") as f:
        raw = f.read()
    events = read_log(args.log)
    sim = replay_authorship(events, args.arm, args.prompt_token_offset)
    rep = {"label": args.label, "log": args.log, "arm": args.arm,
           "log_sha256": hashlib.sha256(gzip.decompress(raw) if args.log.endswith(".gz") else raw).hexdigest(),
           "engine": events[0].data["engine"], "replayed_events": len(sim.events),
           **summarize(sim)}
    with open(os.path.join(args.out, "authorship.json"), "w") as f:
        json.dump(rep, f, indent=1)
    with open(os.path.join(args.out, "authorship.md"), "w") as f:
        f.write(render_md(rep))
    print(render_md(rep))


if __name__ == "__main__":
    main()
