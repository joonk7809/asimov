"""World invariants and the food/coin ledger, checked tick by tick.

Shared by the property tests (tests/test_invariants.py, which assert there are none) and the long
unattended run (asiimov/soak.py, which counts them and keeps going). Each check returns a list of
violation strings instead of raising, so a long run can report every one it hits.
"""
from .events import ORIGINS, Event
from .sim import FARMER_SELL_PRICE, FOOD_PRICE, PERSON_STEPS, STEP_DOS, Sim

BOOL_FIELDS = ("asleep", "has_food", "kin")


def type_violations(x, path="state") -> list[str]:
    """Floats anywhere, and bools outside the fields that are meant to hold them."""
    if isinstance(x, bool):
        return [] if path.endswith(BOOL_FIELDS) else [f"bool at {path}"]
    if isinstance(x, float):
        return [f"float at {path}"]
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            out += type_violations(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += type_violations(v, f"{path}[{i}]")
    return out


def state_violations(sim: Sim) -> list[str]:
    st = sim.state
    n = len(st["agents"])
    out = [] if st["market_food"] >= 0 else [f"market_food {st['market_food']}"]
    out += type_violations(st)
    for a in st["agents"]:
        aid = a["id"]
        if a["inv"]["food"] < 0 or a["inv"]["coin"] < 0:
            out.append(f"agent {aid} inventory {a['inv']}")
        if not all(0 <= v <= 1000 for v in a["needs"].values()):
            out.append(f"agent {aid} needs {a['needs']}")
        for o, r in a["relationships"].items():
            if int(o) == aid or not -1000 <= r["affinity"] <= 1000:
                out.append(f"agent {aid} relationship {o} {r}")
        for b in a["plan"]:
            if not (type(b["start_h"]) is int and type(b["end_h"]) is int and 0 <= b["start_h"] < b["end_h"] <= 24):
                out.append(f"agent {aid} plan block {b}")
        if not (len(a["core"]) <= 6 and all(isinstance(c, str) and len(c) <= 200 for c in a["core"])):
            out.append(f"agent {aid} core {a['core']!r}")
        it = a["intent"]
        if it and not (1 <= len(it["steps"]) <= 4 and 0 <= it["i"] < len(it["steps"])
                       and isinstance(it["goal"], str) and 0 < len(it["goal"]) <= 120):
            out.append(f"agent {aid} intent {it}")
        for s in (it["steps"] if it else []) + ([a["override"]] if a["override"] else []):
            if s["do"] not in STEP_DOS:
                out.append(f"agent {aid} step {s}")
            elif s["do"] in PERSON_STEPS:
                if not (type(s["target"]) is int and 0 <= s["target"] < n and s["target"] != aid):
                    out.append(f"agent {aid} step target {s}")
            elif s["target"] is not None:
                out.append(f"agent {aid} step target {s}")
        if not (a["asleep"] or not a["override"] or a["override"]["left"] > 0):
            out.append(f"agent {aid} override {a['override']}")
    return out


def log_violations(events: list[Event], start: int = 0) -> list[str]:
    """Ids dense from 0, known origins, parents earlier than the event, ticks non-decreasing."""
    out = []
    for i in range(start, len(events)):
        e = events[i]
        if e.id != i:
            out.append(f"event {i} has id {e.id}")
        if e.origin not in ORIGINS:
            out.append(f"event {i} origin {e.origin}")
        if not all(isinstance(p, int) and 0 <= p < e.id for p in e.parents):
            out.append(f"event {i} parents {e.parents}")
        if i and events[i - 1].tick > e.tick:
            out.append(f"event {i} tick {e.tick} after {events[i - 1].tick}")
    return out


def totals(sim: Sim) -> tuple[int, int]:
    ag = sim.state["agents"]
    return sum(a["inv"]["food"] for a in ag) + sim.state["market_food"], sum(a["inv"]["coin"] for a in ag)


class Checker:
    """Wraps sim.step: the food and coin ledgers balance against the tick's events, and the state,
    the new log entries and every agent's traits (which no mind may change) stay valid."""

    def __init__(self, sim: Sim):
        self.sim = sim
        self.traits = [dict(a["traits"]) for a in sim.state["agents"]]

    def before(self, acts=()):
        """Call right before sim.step(acts)."""
        sim = self.sim
        self._tick, self._n0, self._totals = sim.state["tick"], len(sim.events), totals(sim)
        self._burned = sum(sim.state["agents"][m]["inv"]["food"] for act in acts if act["kind"] == "fire"
                           for m in sim.state["families"][act["family"]]["members"])

    def after(self) -> list[str]:
        """Call right after the step; returns its violations, each prefixed with the tick."""
        sim, n0 = self.sim, self._n0
        new = sim.events[n0:]
        produced = sum(e.data.get("food", 0) for e in new if e.kind == "produce")
        eaten = sum(1 for e in new if e.kind == "eat")
        spoiled = sum(e.data["market_food_lost"] for e in new if e.kind in ("spoil", "blight"))
        coin_made = sum(e.data.get("coin", 0) for e in new if e.kind == "produce")
        sold = sum(e.data["food"] for e in new if e.kind == "sell")
        bought = sum(e.data["food"] for e in new
                     if e.kind == "receipt" and e.data["action"] == "buy_food" and e.data["ok"])
        (food0, coin0), (food1, coin1) = self._totals, totals(sim)
        want_food = produced - eaten - self._burned - spoiled
        want_coin = coin_made + FARMER_SELL_PRICE * sold - FOOD_PRICE * bought
        out = []
        if food1 - food0 != want_food:
            out.append(f"food ledger: {food1 - food0} vs {want_food}")
        if coin1 - coin0 != want_coin:
            out.append(f"coin ledger: {coin1 - coin0} vs {want_coin}")
        out += state_violations(sim)
        out += log_violations(sim.events, n0)
        if [a["traits"] for a in sim.state["agents"]] != self.traits:
            out.append("a mind changed someone's traits")
        return [f"tick {self._tick}: {v}" for v in out]

    def step(self, acts=()) -> list[str]:
        self.before(acts)
        self.sim.step(acts)
        return self.after()
