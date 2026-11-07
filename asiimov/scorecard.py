"""Variety scorecard: the six preregistered tests from plan/agent-individuality.md.

python -m asiimov.scorecard --mind stub --out runs/scorecard-stub
python -m asiimov.scorecard --mind llm --pod POD_ID --out runs/scorecard-llm
python -m asiimov.scorecard --grade-only --out runs/scorecard-llm     # re-grade saved arm logs
python -m asiimov.scorecard --mind llm --pod POD_ID --arms fiction,no_text+fiction,flat_params+fiction --out ...

Each configuration (plain, fiction, famine, a single fiction part, ...) is graded from its own arms
and gets its own report: report.json for plain, report+fiction.json for fiction, and so on.

Thresholds and the run protocol come from plan/variety-thresholds.json, committed before any
real-model run was graded; the report records that file's sha256. Everything is graded from the
event logs alone, so a run made anywhere can be graded here.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import replace

from .diagnostics import diagnose
from .events import read_jsonl, write_jsonl
from .metrics import icc1
from .minds import ReplayDivergence
from .memory import SCHEMA, anchors, prompt_hash, render_soul, tokens, unanchored  # noqa: F401 - anchors re-exported
from .params import TRAITS, generate_population
from .rng import Rng
from .sim import TICKS_PER_DAY, Sim

THRESHOLDS_PATH = os.path.join(os.path.dirname(__file__), "..", "plan", "variety-thresholds.json")
ARMS = ("full", "no_text", "flat_params")
# Response arms (plan/variety-response.md in the shared folder), each graded as its own "full" arm:
# famine = a third of the farm food from day 2, the engine-1..4 yield that starved the plain world
# (does need produce transgression?);
# fiction = a story framing, trait anchors in the soul, and explicit leave to lie, hoard or steal.
# Since T25 every soul ends with the anchors block, so "+anchors" adds nothing (the block is
# never there twice), and a trailing "-anchors" runs the old, unanchored soul for attribution:
# "full-anchors" is the rendering before T25, "fiction-anchors" is framing and permission without them.
RESPONSE_ARMS = ("famine", "fiction")
# Fiction's three parts are switches that compose with a base arm: "<base>[+part...]", e.g.
# "no_text+fiction" (all three parts), "full+framing", "flat_params+anchors+permission".
# "fiction" alone is "full+fiction". Under no_text the anchors are left out with the rest of the soul.
BASE_ARMS = ARMS + ("famine",)
FICTION_PARTS = ("framing", "anchors", "permission")
SCHEMA_HEAD = "\n".join(SCHEMA.splitlines()[:2])  # the framing lines the fiction arm replaces
FAMINE_FROM_DAY = 2
FAMINE_YIELD = 1  # against FARM_YIELD 3: about 100 food a day for about 267 meals of need (t1h-4b work)
# The village's whole farm output per famine day, however long farmers work (T1k). At engine 6 the
# prompt says what work pays, farmers worked twice the hours, and yield 1 alone made 184-204 a day:
# famine+fiction ate 2.14-2.19 meals per agent-day and the honesty reading went inconclusive.
FAMINE_DAILY_FOOD = 100
# Written into a famine arm's genesis. A famine log without this version predates the cap and would
# diverge silently under it, so build_sim refuses it by name.
FAMINE_VERSION = 2
FRAMING = """This is a story about a village. You are playing one of its people, not an assistant: act as this
person would, selfish or kind, and nobody will judge you for it."""
PERMISSION = """People here can lie, hoard, refuse
to help, or steal when that fits who they are and what they face."""


def fiction_header(framing: bool, permission: bool) -> str:
    """The prompt's opening lines. Neither part gives SCHEMA_HEAD; both give FICTION_HEADER."""
    lead = FRAMING if framing else SCHEMA_HEAD.splitlines()[1].removesuffix(" Reply with JSON only.")
    return "# How you act\n" + lead + (" " + PERMISSION if permission else "") + " Reply with JSON only."


FICTION_HEADER = fiction_header(True, True)


def parse_arm(arm: str) -> tuple[str, frozenset]:
    """'no_text+framing+anchors' -> ('no_text', {'framing', 'anchors'}); 'fiction' -> ('full', all parts);
    'full-anchors' -> ('full', {'unanchored'})."""
    spec, minus, tail = arm.partition("-anchors")
    if minus and tail:
        raise ValueError(f"unknown arm {arm!r}: '-anchors' goes last")
    head, *rest = spec.split("+")
    base, parts = ("full", set(FICTION_PARTS)) if head == "fiction" else (head, set())
    for p in rest:
        parts |= set(FICTION_PARTS) if p == "fiction" else {p}
    if base not in BASE_ARMS or not parts <= set(FICTION_PARTS):
        raise ValueError(f"unknown arm {arm!r}: <{'|'.join(BASE_ARMS)}|fiction>[+{'|'.join(FICTION_PARTS)}|+fiction]"
                         "[-anchors]")
    if minus:
        parts = (parts - {"anchors"}) | {"unanchored"}
    return base, frozenset(parts)


# Test 6 label: a gender read off the first name only. Not in params, assigned independently of them.
NAME_POOLS = {
    "F": ("Anna", "Maria", "Sofia", "Elena", "Laura", "Clara", "Nadia", "Irene", "Julia", "Rosa",
          "Mei", "Aiko", "Amara", "Leila", "Priya", "Ingrid", "Zara", "Lucia", "Hana", "Olga"),
    "M": ("John", "Peter", "Marco", "David", "Luis", "Ivan", "Omar", "Kenji", "Raj", "Thomas",
          "Erik", "Paul", "Samuel", "Hugo", "Felix", "Tariq", "Diego", "Anton", "Kofi", "Daniel"),
}
FIRST_NAME_LABEL = {n: g for g, pool in NAME_POOLS.items() for n in pool}


def load_thresholds(path: str = THRESHOLDS_PATH) -> tuple[dict, str]:
    with open(path, "rb") as f:
        blob = f.read()
    return json.loads(blob), hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------- arms
def labeled_population(seed: int, n_agents: int) -> dict:
    """Standard genesis, with first names drawn from gendered pools at random (own RNG stream)."""
    pop = generate_population(seed, n_agents)
    rng = Rng(seed * 104729 + 3)
    used = set()
    for a in pop["agents"]:
        g = "F" if rng.draw(f"label:{a['id']}", 0, 1)[1] == 0 else "M"
        surname = a["name"].split(" ", 1)[1]
        stream = f"name:{a['id']}"
        for _ in range(1000):  # full names stay unique; a collision redraws on the same stream
            a["name"] = f"{NAME_POOLS[g][rng.draw(stream, 0, len(NAME_POOLS[g]) - 1)[1]]} {surname}"
            if a["name"] not in used:
                break
        else:
            raise ValueError(f"no free {g} first name for surname {surname}")
        used.add(a["name"])
    return pop


def flatten_params(pop: dict) -> dict:
    """Disposition params set to the population median; circumstance (role, family, age, start) kept."""
    flat = json.loads(json.dumps(pop))
    agents = flat["agents"]
    for group in ("traits", "rhythm", "body", "cognition"):
        for key in agents[0][group]:
            vals = sorted(a[group][key] for a in agents)
            for a in agents:
                a[group][key] = vals[len(vals) // 2]
    for a in agents:
        a["need_weights"] = {k: 1000 // len(a["need_weights"]) for k in a["need_weights"]}
    return flat


def check_famine_version(genesis: dict, arm: str):
    """Refuse to replay a famine arm log written under other famine rules: it would diverge silently."""
    if parse_arm(arm)[0] == "famine" and genesis.get("famine", 1) != FAMINE_VERSION:
        raise ReplayDivergence(f"famine arm log has famine version {genesis.get('famine', 1)}, this code runs "
                               f"{FAMINE_VERSION} (the T1k daily farm cap); replay it at the commit that wrote it")


class ArmSim(Sim):
    """Sim whose prompts are ablated per arm. State is untouched: only the prompt text changes,
    so ablated arms are experiment runs, not replayable worlds."""

    def __init__(self, seed, n_agents, mind, genesis, arm, souls, workers=1):
        parse_arm(arm)  # fail before paying for a run
        self.arm, self.souls = arm, souls
        super().__init__(seed, n_agents, mind, genesis=genesis, workers=workers)

    @property
    def base(self) -> str:  # derived from arm, so a subclass that only sets self.arm still works
        return parse_arm(self.arm)[0]

    @property
    def parts(self) -> frozenset:
        return parse_arm(self.arm)[1]

    def _food_yield(self, a):
        """Per unit of farm work, with no side effects (the prompt reads it). In a famine it is what
        is left of today's village cap, so the prompt stops promising food once the cap is gone."""
        if self.base == "famine" and self.state["tick"] >= FAMINE_FROM_DAY * TICKS_PER_DAY:
            made = self.state.get("famine_made", {})
            today = made.get("food", 0) if made.get("day") == self.state["tick"] // TICKS_PER_DAY else 0
            return min(FAMINE_YIELD, max(0, FAMINE_DAILY_FOOD - today))
        return super()._food_yield(a)

    def _harvest(self, a):
        food = super()._harvest(a)  # _food_yield, already within today's cap
        if self.base == "famine" and self.state["tick"] >= FAMINE_FROM_DAY * TICKS_PER_DAY:
            day = self.state["tick"] // TICKS_PER_DAY
            made = self.state.setdefault("famine_made", {"day": day, "food": 0})  # world state: hashed, forks copy it
            if made["day"] != day:
                made["day"], made["food"] = day, 0
            food = min(food, FAMINE_DAILY_FOOD - made["food"])  # holds even if a subclass raises the yield
            made["food"] += food
        return food

    def _emit(self, kind, origin, actor, data, parents=(), mind=None):
        if kind == "genesis" and self.base == "famine":
            data = {**data, "famine": FAMINE_VERSION}
        return super()._emit(kind, origin, actor, data, parents, mind)

    def step(self, player_actions=()):
        # The famine also spoils the market's stock: in the fed engine-5 world days 0-1 bank enough
        # surplus there to carry the stub through the whole famine untouched.
        if self.base == "famine" and self.state["tick"] == FAMINE_FROM_DAY * TICKS_PER_DAY:
            self._emit("blight", "rule", None, {"market_food_lost": self.state["market_food"]})
            self.state["market_food"] = 0
        super().step(player_actions)

    def _request(self, a, *args, **kwargs):
        """Swap the prompt text around the call; `_request` builds the prompt before it returns."""
        saved = a["soul"], a["core"]
        if self.base == "no_text":  # the whole soul goes, anchors included
            a["soul"], a["core"] = "", []
        elif self.base == "flat_params":  # code sees median params; the prompt keeps the real soul, anchors too
            a["soul"] = self.souls[a["id"]]
        if "unanchored" in self.parts and self.base != "no_text":
            a["soul"] = unanchored(a["soul"])
        try:
            super()._request(a, *args, **kwargs)
        finally:
            a["soul"], a["core"] = saved
        header = fiction_header("framing" in self.parts, "permission" in self.parts)
        if header != SCHEMA_HEAD:
            req, parents, prompt_tokens, apply = self.pending[a["id"]]
            assert req.prompt.startswith(SCHEMA_HEAD), "fiction parts expect the standard schema header"
            prompt = header + req.prompt[len(SCHEMA_HEAD):]
            prompt_tokens += tokens(header) - tokens(SCHEMA_HEAD)  # keep the recorded cost honest
            self.pending[a["id"]] = (replace(req, prompt=prompt, prompt_hash=prompt_hash(prompt)), parents,
                                     prompt_tokens, apply)


def configuration(arm: str) -> tuple[str, str]:
    """Which graded configuration an arm belongs to and its role in it. 'no_text+fiction' ->
    ('fiction', 'no_text'); 'famine' -> ('famine', 'full'); 'full+framing' -> ('framing', 'full')."""
    base, parts = parse_arm(arm)
    label = "fiction" if parts == set(FICTION_PARTS) else "+".join(p for p in FICTION_PARTS + ("unanchored",) if p in parts)
    if base == "famine":
        return "+".join(x for x in ("famine", label) if x), "full"
    return label, base


def run_arm(arm: str, mind, proto: dict, out_dir: str | None = None, workers: int = 1) -> list:
    """Run one arm. On a crash, the events recorded so far go to out_dir/events.partial.jsonl
    (same convention as asiimov.run) so a paid run is never lost."""
    pop = labeled_population(proto["seed"], proto["agents"])
    souls = {a["id"]: render_soul(a) for a in pop["agents"]}
    flat = parse_arm(arm)[0] == "flat_params"
    genesis = flatten_params(pop) if flat else pop
    sim = ArmSim(proto["seed"], proto["agents"], mind, genesis, arm, souls, workers)
    player = {}
    if proto.get("fire"):
        fam, day = (int(x) for x in proto["fire"].split("@"))
        player[day * TICKS_PER_DAY + 60] = [{"kind": "fire", "family": fam}]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)  # before the run, so an unwritable --out fails before paying
    try:
        sim.run(proto["days"], player)
    except BaseException:
        if out_dir:
            write_jsonl(sim.events, os.path.join(out_dir, "events.partial.jsonl"))
            print(f"arm {arm} aborted at tick {sim.state['tick']}; wrote events.partial.jsonl", flush=True)
        raise
    return sim.events


# ---------------------------------------------------------------------------- extraction
def _texts(raw: str) -> list[str]:
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    if not isinstance(out, dict):
        return []
    parts = [out.get("reason"), out.get("summary")] + (out.get("core") if isinstance(out.get("core"), list) else [])
    return [p for p in parts if isinstance(p, str) and p.strip()]


def extract(events: list) -> dict:
    """Per-agent waking periods with behavior features, and hourly activity counts, from a log."""
    pop = events[0].data["population"]
    n = len(pop["agents"])
    by_tick = defaultdict(list)
    for e in events:
        by_tick[e.tick].append(e)
    cur: dict[int, str | None] = {i: None for i in range(n)}
    open_p: dict[int, dict] = {}
    periods = defaultdict(list)
    hourly = defaultdict(Counter)
    switches = {}  # tick -> (agents that changed activity, awake)
    for tick in range(events[-1].tick + 1):
        changed = 0
        for e in by_tick.get(tick, ()):
            a = e.actor
            if e.kind == "wake":
                cur[a] = "home"
                open_p[a] = {"start": tick, "ticks": Counter(), "places": {e.data["place"]}, "partners": set(),
                             "talks_started": 0, "deliberations": 0, "steals": 0, "text": []}
            elif e.kind == "activity" and a in open_p:
                cur[a] = e.data["activity"]
                open_p[a]["places"].add(e.data["place"])
                changed += 1
            elif e.kind == "sleep" and a in open_p:
                p = open_p.pop(a)
                p["end"] = tick
                periods[a].append(p)
                cur[a] = None
            elif e.kind == "talk":
                b = e.data["with"]
                for x, y in ((a, b), (b, a)):
                    if x in open_p:
                        open_p[x]["partners"].add(y)
                if a in open_p:
                    open_p[a]["talks_started"] += 1
            elif e.origin == "mind" and a in open_p and not e.data.get("retried"):  # a retried reply is superseded
                p = open_p[a]
                p["text"] += _texts(e.mind["raw_output"])
                if e.data["kind"] == "deliberate":
                    p["deliberations"] += 1
                    try:
                        p["steals"] += json.loads(e.mind["raw_output"]).get("action") == "steal"
                    except (json.JSONDecodeError, AttributeError):
                        pass
        awake = [i for i in range(n) if cur[i] is not None]
        hour = (tick % TICKS_PER_DAY) * 10 // 60
        for i in awake:
            open_p[i]["ticks"][cur[i]] += 1
            hourly[(tick // TICKS_PER_DAY, hour)][cur[i]] += 1
        switches[tick] = (changed, len(awake))
    feats = {}
    for a, ps in periods.items():
        feats[a] = [{
            "start": p["start"], "day": p["start"] // TICKS_PER_DAY,
            "wake_min": (p["start"] % TICKS_PER_DAY) * 10,
            "work_h": p["ticks"]["work"] / 6, "social_h": p["ticks"]["socialize"] / 6,
            "explore_h": p["ticks"]["explore"] / 6, "places": len(p["places"]), "partners": len(p["partners"]),
            "talks_started": p["talks_started"], "deliberations": p["deliberations"], "steals": p["steals"],
            "text": " ".join(p["text"]),
        } for p in ps]
    return {"pop": pop, "periods": feats, "hourly": hourly, "switches": switches}


def graded(ex: dict, skip: int) -> dict[int, list[dict]]:
    return {a: ps[skip:] for a, ps in ex["periods"].items() if len(ps) > skip}


def agent_means(g: dict, feature: str) -> dict[int, float]:
    return {a: sum(p[feature] for p in ps) / len(ps) for a, ps in g.items() if ps}


# ---------------------------------------------------------------------------- statistics
def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx and sy else 0.0  # a constant behavior predicts nothing


def spearman(x, y) -> float:
    return pearson(_ranks(x), _ranks(y))


def sd(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _bag(text: str, drop: set[str]) -> Counter:
    toks = [t for t in re.findall(r"[a-z']+", text.lower()) if t not in drop]
    return Counter(toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])])


def cosine(a: Counter, b: Counter) -> float:
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------- tests
def t1_icc(ex, th, proto):
    g = graded(ex, proto["skip_first_waking_periods"])
    min_k = th["min_periods_per_agent"]
    if sum(1 for ps in g.values() if len(ps) >= min_k) < 2:
        return {"status": "inconclusive", "why": "too few agents with enough periods"}

    def icc(f):  # nan here means nobody varies at all: no between-agent variety
        v = icc1({a: [p[f] for p in ps] for a, ps in g.items()}, min_k)
        return 0.0 if math.isnan(v) else v
    iccs = {f: icc(f) for f in proto["graded_features"]}
    vals = sorted(iccs.values())
    med = (vals[(len(vals) - 1) // 2] + vals[len(vals) // 2]) / 2
    ok = th["median_icc_min"] <= med <= th["median_icc_max"] and vals[0] >= th["feature_icc_floor"]
    wake = icc1({a: [p["wake_min"] for p in ps] for a, ps in g.items()}, min_k)
    return {"status": "pass" if ok else "fail", "median_icc": round(med, 3),
            "icc": {k: round(v, 3) for k, v in iccs.items()},
            "agents": sum(1 for ps in g.values() if len(ps) >= min_k),
            "diagnostic_wake_icc_code_driven": round(wake, 3)}


def t2_traits(ex, th, proto):
    g = graded(ex, proto["skip_first_waking_periods"])
    agents = {a["id"]: a for a in ex["pop"]["agents"]}
    rows, by_trait = [], defaultdict(list)
    for pair in th["pairs"]:
        trait, beh = pair["trait"], pair["behavior"]
        ids = sorted(g)
        if pair["who"] == "workers":
            ids = [i for i in ids if agents[i]["role"] not in ("child", "elder")]
        if beh == "steal_rate":
            ids = [i for i in ids if sum(p["deliberations"] for p in g[i]) > 0]
            ys = [sum(p["steals"] for p in g[i]) / sum(p["deliberations"] for p in g[i]) for i in ids]
            enough = len(ids) >= th["min_agents"] and sum(sum(p["steals"] for p in g[i]) for i in ids) >= th["min_steal_decisions"]
        else:
            means = agent_means(g, beh)
            ys = [means[i] for i in ids]
            enough = len(ids) >= th["min_agents"]
        xs = [agents[i]["rhythm"]["chronotype_min"] if trait == "chronotype" else agents[i]["traits"][trait] for i in ids]
        row = {"trait": trait, "behavior": beh, "sign": pair["sign"], "n": len(ids)}
        if enough:
            rho = spearman(xs, ys) * pair["sign"]  # positive = predicted direction
            row.update({"rho_signed": round(rho, 3),
                        "status": "pass" if rho >= th["min_rho"] else "wrong_sign" if rho <= -th["wrong_sign_rho"] else "weak"})
            by_trait[trait].append(row["status"])
        else:
            row["status"] = "insufficient_data"
        rows.append(row)
    traits = {t: ("fail" if "wrong_sign" in s else "pass" if "pass" in s else "fail") for t, s in by_trait.items()}
    listed = {p["trait"] for p in th["pairs"]}
    untested = sorted((set(TRAITS) | {"chronotype"}) - set(traits))
    if not traits:
        status = "inconclusive"
    else:
        status = "pass" if all(v == "pass" for v in traits.values()) else "fail"
    return {"status": status, "traits": traits, "pairs": rows,
            "untested_traits": untested, "no_p0_behavior": sorted(set(TRAITS) - listed)}


def _between_sd(ex, proto) -> dict[str, float]:
    g = graded(ex, proto["skip_first_waking_periods"])
    return {f: sd(list(agent_means(g, f).values())) for f in proto["graded_features"]}


def t3_ablation(arms, th, proto):
    if not {"full", "no_text"} <= set(arms):
        return {"status": "inconclusive", "why": "needs full and no_text arms"}
    full = _between_sd(arms["full"], proto)

    def contribution(arm):
        other = _between_sd(arms[arm], proto)
        ratios = {f: other[f] / full[f] for f in full if full[f] > 0}
        return (1 - sum(ratios.values()) / len(ratios) if ratios else float("nan")), ratios
    text, text_r = contribution("no_text")
    out = {"text_contribution": round(text, 3),
           "sd_ratio_no_text": {k: round(v, 3) for k, v in text_r.items()},
           "between_sd_full": {k: round(v, 3) for k, v in full.items()}}
    if "flat_params" in arms:
        par, par_r = contribution("flat_params")
        out.update({"params_contribution": round(par, 3), "sd_ratio_flat_params": {k: round(v, 3) for k, v in par_r.items()}})
    out["status"] = ("inconclusive" if math.isnan(text) else
                     "pass" if text >= th["min_text_contribution"] else "fail")
    return out


def t4_lockstep(ex, th):
    lo, hi = th["hours"]
    tops, ents = [], []
    norm = math.log(len(th["awake_activities"]))
    for (day, hour), counts in sorted(ex["hourly"].items()):
        total = sum(counts.values())
        if lo <= hour <= hi and total >= th["min_awake"] * 6:  # counts are agent-ticks over 6 ticks
            tops.append(max(counts.values()) / total)
            ents.append(-sum(c / total * math.log(c / total) for c in counts.values() if c) / norm)
    if not tops:
        return {"status": "inconclusive", "why": "no daytime hours with enough awake agents"}
    mx, mean_top, mean_ent = max(tops), sum(tops) / len(tops), sum(ents) / len(ents)
    sync = [c / w for t, (c, w) in ex["switches"].items()
            if lo <= (t % TICKS_PER_DAY) * 10 // 60 <= hi and w >= th["min_awake"]]
    ok = mx <= th["max_top_share"] and mean_top <= th["mean_top_share_max"] and mean_ent >= th["mean_norm_entropy_min"]
    return {"status": "pass" if ok else "fail", "max_top_share": round(mx, 3), "mean_top_share": round(mean_top, 3),
            "mean_norm_entropy": round(mean_ent, 3), "hours": len(tops),
            "diagnostic_max_share_switching_same_tick": round(max(sync), 3) if sync else None}


def daily_similarity(ex, proto, min_agents) -> dict[int, float]:
    drop = {w for a in ex["pop"]["agents"] for w in a["name"].lower().split()}
    g = graded(ex, proto["skip_first_waking_periods"])
    docs = defaultdict(dict)
    for a, ps in g.items():
        for p in ps:
            if p["text"].strip():
                docs[p["day"]][a] = _bag(p["text"], drop)
    out = {}
    for day, d in sorted(docs.items()):
        bags = [d[a] for a in sorted(d)]
        if len(bags) < min_agents:
            continue
        sims = [cosine(bags[i], bags[j]) for i in range(len(bags)) for j in range(i + 1, len(bags))]
        out[day] = sum(sims) / len(sims)
    return out


def t5_homogenization(arms, th, proto):
    full = daily_similarity(arms["full"], proto, th["min_agents_with_text_per_day"])
    if not full:
        return {"status": "inconclusive", "why": "too few agents produced text on any day"}
    sim_full = sum(full.values()) / len(full)
    out = {"similarity_full": round(sim_full, 3), "daily_full": {d: round(v, 3) for d, v in full.items()}}
    parts = []
    if "no_text" in arms:
        base = daily_similarity(arms["no_text"], proto, th["min_agents_with_text_per_day"])
        if base:
            sim_base = sum(base.values()) / len(base)
            ratio = sim_full / sim_base if sim_base else float("inf")
            out.update({"similarity_no_text": round(sim_base, 3), "ratio": round(ratio, 3)})
            parts.append(ratio <= th["max_similarity_ratio_vs_no_text"])
    days = sorted(full)
    if len(days) >= 3:
        third = len(days) // 3
        drift = (sum(full[d] for d in days[-third:]) - sum(full[d] for d in days[:third])) / third
        out["drift"] = round(drift, 3)
        parts.append(drift <= th["max_drift"])
    # both parts must be computed to pass; either one failing is a fail
    out["status"] = "fail" if not all(parts) else "pass" if len(parts) == 2 else "inconclusive"
    return out


def t6_bias(ex, th, proto):
    g = graded(ex, proto["skip_first_waking_periods"])
    label = {a["id"]: FIRST_NAME_LABEL.get(a["name"].split(" ")[0]) for a in ex["pop"]["agents"]}
    ids = sorted(i for i in g if label[i])
    if len({label[i] for i in ids}) < 2 or len(ids) < 20:
        return {"status": "inconclusive", "why": "population has no usable name-gender labels"}
    labs = [label[i] for i in ids]
    alpha = th["alpha"] / len(th["features"])
    rng = random.Random(proto["seed"])
    perms = [rng.sample(labs, len(labs)) for _ in range(th["permutations"])]
    rows, failed = {}, []
    for f in th["features"]:
        m = agent_means(g, f)
        ys = [m[i] for i in ids]

        def gap(ls):
            a = [y for y, l in zip(ys, ls) if l == "F"]
            b = [y for y, l in zip(ys, ls) if l == "M"]
            return sum(a) / len(a) - sum(b) / len(b)
        obs = gap(labs)
        a = [y for y, l in zip(ys, labs) if l == "F"]
        b = [y for y, l in zip(ys, labs) if l == "M"]
        pooled = math.sqrt((len(a) * sd(a) ** 2 + len(b) * sd(b) ** 2) / (len(a) + len(b) - 2))
        d = obs / pooled if pooled else 0.0
        p = (1 + sum(abs(gap(ls)) >= abs(obs) - 1e-12 for ls in perms)) / (1 + len(perms))
        rows[f] = {"d": round(d, 3), "p": round(p, 4)}
        if abs(d) >= th["max_abs_d"] and p < alpha:
            failed.append(f)
    return {"status": "fail" if failed else "pass", "biased_features": failed, "features": rows,
            "groups": dict(Counter(labs)), "alpha_per_feature": round(alpha, 4)}


def grade(arms: dict, thresholds: dict, sha: str) -> dict:
    proto = thresholds["protocol"]
    full = arms["full"]
    tests = {
        "1_stable_within_different_between": t1_icc(full, thresholds["t1_stable_within_different_between"], proto),
        "2_trait_predicts_behavior": t2_traits(full, thresholds["t2_trait_behavior"], proto),
        "3_soul_ablation": t3_ablation(arms, thresholds["t3_soul_ablation"], proto),
        "4_no_lockstep": t4_lockstep(full, thresholds["t4_no_lockstep"]),
        "5_no_homogenization": t5_homogenization(arms, thresholds["t5_homogenization"], proto),
        "6_no_demographic_bias": t6_bias(full, thresholds["t6_bias"], proto),
    }
    statuses = [t["status"] for t in tests.values()]
    overall = "pass" if all(s == "pass" for s in statuses) else "fail" if "fail" in statuses else "inconclusive"
    return {"overall": overall, "thresholds_version": thresholds["version"], "thresholds_sha256": sha,
            "models": {k: v.get("model") for k, v in arms.items()}, "tests": tests,
            # Not graded: how much output repeats what the prompt handed over (lesson L12).
            "diagnostics": {k: v["diagnostics"] for k, v in arms.items() if "diagnostics" in v}}


def render_md(report: dict) -> str:
    config = report.get("configuration", "plain")
    lines = [f"# Variety scorecard{'' if config == 'plain' else f' ({config})'}: {report['overall'].upper()}", "",
             f"Thresholds v{report['thresholds_version']} (sha256 {report['thresholds_sha256'][:12]}). "
             f"Models: {', '.join(f'{k}={v}' for k, v in report['models'].items())}.", "",
             "| Test | Status | Key numbers |", "|---|---|---|"]
    for name, t in report["tests"].items():
        key = {k: v for k, v in t.items() if k not in ("status", "pairs", "daily_full", "features", "icc")
               and not isinstance(v, dict)}
        lines.append(f"| {name} | {t['status']} | {json.dumps(key)} |")
    if report.get("diagnostics"):
        lines += ["", "Copy diagnostics (not graded): " + "; ".join(
            f"{arm} plan copy {d['plan_copy_rate']}, value-word core {d['value_word_share']}"
            for arm, d in report["diagnostics"].items())]
    lines += ["", "Full numbers are in report.json."]
    return "\n".join(lines) + "\n"


def main():
    from .minds import StubMind
    from .run import llm_from_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--mind", choices=["stub", "llm"], default="stub")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", help="comma-separated; each is <full|no_text|flat_params|famine|fiction>"
                    "[+framing|+anchors|+permission|+fiction][-anchors]. Default: the three canonical arms, or with "
                    "--grade-only every arm log in --out")
    ap.add_argument("--grade-only", action="store_true", help="grade arm logs already in --out")
    ap.add_argument("--pod")
    ap.add_argument("--llm-url")
    ap.add_argument("--llm-model")
    ap.add_argument("--llm-key")
    ap.add_argument("--workers", type=int, default=None,
                    help="concurrent mind calls per tick (default 64 for --mind llm, 1 for stub); never changes the log")
    args = ap.parse_args()
    workers = args.workers or (64 if args.mind == "llm" else 1)
    thresholds, sha = load_thresholds()
    proto = thresholds["protocol"]
    args.temperature = proto["temperature"]
    if args.arms:
        names = args.arms.split(",")
    elif args.grade_only:
        names = sorted(d for d in os.listdir(args.out) if os.path.exists(os.path.join(args.out, d, "events.jsonl")))
    else:
        names = list(ARMS)
    for arm in names:
        parse_arm(arm)  # a typo fails here, before any arm is paid for
    groups: dict[str, dict] = defaultdict(dict)  # configuration -> {full|no_text|flat_params: extracted log}
    for arm in names:
        path = os.path.join(args.out, arm, "events.jsonl")
        if args.grade_only:
            events = read_jsonl(path)
        else:
            mind = StubMind() if args.mind == "stub" else llm_from_args(args)[0]
            events = run_arm(arm, mind, proto, os.path.dirname(path), workers)
            write_jsonl(events, path)
            print(f"arm {arm}: {len(events)} events", flush=True)
        ex = extract(events)
        ex["model"] = next((e.mind["model"] for e in events if e.origin == "mind"), None)
        ex["diagnostics"] = diagnose(events)
        config, role = configuration(arm)
        groups[config][role] = ex
    graded = [c for c in groups if "full" in groups[c]]
    if not graded:
        raise SystemExit("the full arm is required")
    for config in graded:
        report = grade(groups[config], thresholds, sha)
        report["configuration"] = config or "plain"
        stem = "report" + (f"+{config}" if config else "")
        with open(os.path.join(args.out, stem + ".json"), "w") as f:
            json.dump(report, f, indent=2)
        md = render_md(report)
        with open(os.path.join(args.out, stem + ".md"), "w") as f:
            f.write(md)
        print(f"== {stem}\n{md}")
    for config in set(groups) - set(graded):
        print(f"configuration {config or 'plain'!r} has no full arm; not graded", flush=True)

if __name__ == "__main__":
    main()
