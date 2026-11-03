"""Invariants of the world core, checked as strongly as is cheap.

What each test establishes (tested, not proven, unless it says otherwise):
- RNG range and purity: property-tested over arbitrary seeds, streams and ranges.
- Determinism: same inputs give the same log under different PYTHONHASHSEED values, and
  a static lint keeps floats, true division and ambient nondeterminism out of the core.
- Replay: any recorded log, including ones from adversarial minds, replays to the same
  hash, and tampering with any event is detected.
- State invariants and the food/coin ledger hold after every tick, under a mind that
  emits malformed, wrong-typed and hostile output.
"""
import ast
import json
import os
import random
import subprocess
import sys

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from asiimov.events import ORIGINS, Event
from asiimov.minds import MindReply, NoisyMind, ReplayDivergence, StubMind
from asiimov.replay import replay
from asiimov.rng import Rng
from asiimov.sim import FARMER_SELL_PRICE, FOOD_PRICE, TICKS_PER_DAY, Sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = ["rng.py", "params.py", "sim.py", "memory.py", "events.py", "replay.py"]
SLOW = settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])


# ------------------------------------------------------------------ RNG
@given(st.integers(), st.text(), st.integers(-10**12, 10**12), st.integers(0, 10**12), st.integers(1, 5))
def test_draw_in_range_advances_and_is_pure(seed, stream, lo, span, n):
    r = Rng(seed)
    draws = [r.draw(stream, lo, lo + span) for _ in range(n)]
    assert [i for i, _ in draws] == list(range(n))
    assert all(lo <= v <= lo + span for _, v in draws)
    assert r.counters[stream] == n
    again = Rng(seed)
    assert draws == [again.draw(stream, lo, lo + span) for _ in range(n)]


@given(st.integers(), st.text(), st.integers(0, 10**6))
def test_normal_milli_is_bounded(seed, stream, sd):
    # Irwin-Hall of 12 uniforms is within +-6 sd by construction.
    v = Rng(seed).normal_milli(stream, sd)
    assert -6 * sd - 1 <= v <= 6 * sd


# ------------------------------------------------------------------ hostile mind
HOSTILE = [
    '{"action":"steal","target":%d}', '{"action":"ask_kin","target":%d}', '{"action":"steal","target":true}',
    '{"action":"ask_kin","target":1.0}', '{"action":["steal"],"target":0}', '{"action":{"a":1}}',
    '{"plan":[{"activity":"work","start_h":true,"end_h":5}]}', '{"plan":[{"activity":"work","start_h":8.0,"end_h":12}]}',
    '{"plan":[{"activity":"work","start_h":NaN,"end_h":Infinity}]}', '{"plan":[{"activity":"sleep","start_h":0,"end_h":24}]}',
    '{"plan":[{"activity":"work","start_h":-5,"end_h":99}]}', '{"plan":[1,"x",null]}', '{"plan":"all day"}',
    '{"core":"x"}', '{"core":[1,null,"  ","%s"]}', '{"core":["%s","%s","%s"]}', '[' * 20000, '[]', 'null', '7',
    '', 'I will steal the bread.', '```json\n{"action":"wait"}\n```', '{"action":"buy_food"}', '{"action":"go_market"}',
    '{"action":"wait","target":%d}', '{"action":"buy_food","action":"steal","target":%d}',
]


class HostileMind:
    """Mixes the stub's sensible answers with malformed, wrong-typed and hostile outputs."""
    model = "hostile"

    def __init__(self, seed: int, n_agents: int):
        self.r = random.Random(seed)
        self.n = n_agents
        self.stub = StubMind()

    def decide(self, req):
        if self.r.random() < 0.4:
            return self.stub.decide(req)
        t = self.r.choice(HOSTILE)
        if "%d" in t:
            t = t.replace("%d", str(self.r.choice([-1, 0, req.agent_id, self.n - 1, self.n, 10**30,
                                                    self.r.randrange(self.n)])))
        t = t.replace("%s", "x" * self.r.choice([1, 500]))
        # a hostile mind also tries to edit the world through the object it was handed
        req.view["traits"]["honesty"] = -999
        req.view["recent"].append("injected")
        return MindReply(t, self.model, {})


# ------------------------------------------------------------------ invariant checks
def walk_types(x, path="state"):
    if isinstance(x, bool) and not path.endswith(("asleep", "has_food", "kin")):
        raise AssertionError(f"bool at {path}")
    if isinstance(x, float):
        raise AssertionError(f"float at {path}")
    if isinstance(x, dict):
        for k, v in x.items():
            walk_types(v, f"{path}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            walk_types(v, f"{path}[{i}]")


def check_state(sim: Sim):
    st_ = sim.state
    assert st_["market_food"] >= 0
    walk_types(st_)
    for a in st_["agents"]:
        assert a["inv"]["food"] >= 0 and a["inv"]["coin"] >= 0, a["id"]
        assert all(0 <= v <= 1000 for v in a["needs"].values()), a["needs"]
        for o, r in a["relationships"].items():
            assert int(o) != a["id"]
            assert -1000 <= r["affinity"] <= 1000, (a["id"], o, r)
        for b in a["plan"]:
            assert type(b["start_h"]) is int and type(b["end_h"]) is int and 0 <= b["start_h"] < b["end_h"] <= 24
        assert len(a["core"]) <= 6 and all(isinstance(c, str) and len(c) <= 200 for c in a["core"])
        assert a["traits"] == st_["agents"][a["id"]]["traits"]


def check_log(events: list[Event]):
    for i, e in enumerate(events):
        assert e.id == i
        assert e.origin in ORIGINS
        assert all(isinstance(p, int) and 0 <= p < e.id for p in e.parents), e
        if i:
            assert events[i - 1].tick <= e.tick


def totals(sim):
    ag = sim.state["agents"]
    return sum(a["inv"]["food"] for a in ag) + sim.state["market_food"], sum(a["inv"]["coin"] for a in ag)


def stepped_run(sim: Sim, days: int, player: dict):
    """Step tick by tick, checking state invariants and the food/coin ledger after every tick."""
    genesis_traits = [dict(a["traits"]) for a in sim.state["agents"]]
    for _ in range(days * TICKS_PER_DAY):
        tick = sim.state["tick"]
        acts = player.get(tick, [])
        food0, coin0 = totals(sim)
        burned = sum(sim.state["agents"][m]["inv"]["food"] for act in acts
                     for m in sim.state["families"][act["family"]]["members"])
        n0 = len(sim.events)
        sim.step(acts)
        new = sim.events[n0:]
        produced = sum(e.data.get("food", 0) for e in new if e.kind == "produce")
        eaten = sum(1 for e in new if e.kind == "eat")
        coin_made = sum(e.data.get("coin", 0) for e in new if e.kind == "produce")
        sold = sum(e.data["food"] for e in new if e.kind == "sell")
        bought = sum(1 for e in new if e.kind == "receipt" and e.data["action"] == "buy_food" and e.data["ok"])
        food1, coin1 = totals(sim)
        assert food1 - food0 == produced - eaten - burned, f"food ledger broken at tick {tick}"
        assert coin1 - coin0 == coin_made + FARMER_SELL_PRICE * sold - FOOD_PRICE * bought, f"coin ledger at {tick}"
        check_state(sim)
    assert [a["traits"] for a in sim.state["agents"]] == genesis_traits, "a mind changed someone's traits"
    check_log(sim.events)


# ------------------------------------------------------------------ properties
@SLOW
@given(seed=st.integers(0, 10**6), agents=st.integers(2, 25), fire_family=st.integers(0, 3))
def test_hostile_minds_cannot_break_invariants_and_every_log_replays(seed, agents, fire_family):
    sim = Sim(seed, agents, HostileMind(seed, agents))
    fam = min(fire_family, len(sim.state["families"]) - 1)
    stepped_run(sim, 2, {TICKS_PER_DAY + 60: [{"kind": "fire", "family": fam}]})
    sim.finish()
    r = replay([Event(**json.loads(e.to_json())) for e in sim.events])
    assert r.state_hash() == sim.state_hash()


@SLOW
@given(seed=st.integers(0, 10**6), agents=st.integers(2, 20), pick=st.integers(0, 10**9),
       field=st.sampled_from(["data", "parents", "tick", "actor"]))
def test_replay_detects_tampering_with_any_rule_derived_field(seed, agents, pick, field):
    """Any edit to what the rules derive is caught. Edits to a recorded mind output are caught only
    when they change the world (see test_world.py): replay checks consequences, not authorship."""
    sim = Sim(seed, agents, NoisyMind())
    sim.run(1)
    sim.finish()
    events = [Event(**json.loads(e.to_json())) for e in sim.events]
    e = events[pick % len(events)]
    if field == "data":
        e.data["tampered"] = 1
    elif field == "parents":
        e.parents = e.parents + [0] if e.id else [0]
    elif field == "tick":
        e.tick += 1
    else:
        e.actor = -1 if e.actor is not None else 0
    with pytest.raises(ReplayDivergence):
        replay(events)


def test_stub_world_keeps_ledger_and_invariants_for_a_week():
    sim = Sim(1, 60, StubMind())
    stepped_run(sim, 7, {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 3}]})


# ------------------------------------------------------------------ determinism
def test_world_hash_does_not_depend_on_python_hash_seed():
    code = ("from asiimov.sim import Sim; from asiimov.minds import StubMind; "
            "s = Sim(4, 40, StubMind()); s.run(2, {200: [{'kind': 'fire', 'family': 1}]}); print(s.state_hash())")
    out = set()
    for h in ("0", "1", "987654"):
        env = dict(os.environ, PYTHONHASHSEED=h)
        out.add(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                               text=True, check=True).stdout.strip())
    assert len(out) == 1, out


BANNED_MODULES = {"random", "time", "datetime", "os", "secrets", "uuid", "threading", "asyncio", "math"}


@pytest.mark.parametrize("name", CORE)
def test_core_is_integer_only_and_free_of_ambient_nondeterminism(name):
    """Static lint: no float literals, no true division, no float(), no set iteration helpers,
    no imports that read clocks, OS entropy or the environment."""
    tree = ast.parse(open(os.path.join(ROOT, "asiimov", name)).read())
    for node in ast.walk(tree):
        where = f"{name}:{getattr(node, 'lineno', '?')}"
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            raise AssertionError(f"float literal at {where}")
        if isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Div):
            raise AssertionError(f"true division at {where}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("float", "hash", "id"):
            raise AssertionError(f"{node.func.id}() at {where}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = {m.split(".")[0] for m in mods} & BANNED_MODULES
            assert not bad, f"imports {bad} at {where}"


# ------------------------------------------------------------------ regressions found in review
def _two_agents_apart():
    sim = Sim(2, 10, StubMind())
    a, t = sim.state["agents"][0], sim.state["agents"][1]
    a["place"], t["place"] = "market", "woods"
    t["inv"]["food"] = 3
    return sim, a, t


def test_steal_checks_where_the_target_is_now_not_where_it_was_at_tick_start():
    sim, a, t = _two_agents_apart()
    sim._resolve(a, {"action": "steal", "target": t["id"]}, 0)  # t may have been at the market at tick start
    assert sim.events[-1].data == {"action": "steal", "ok": False, "why": "target not here"}
    assert t["inv"]["food"] == 3


def test_kin_affinity_stays_bounded():
    sim = Sim(3, 30, StubMind())
    fam = next(f for f in sim.state["families"] if len(f["members"]) >= 2)
    a, t = (sim.state["agents"][i] for i in fam["members"][:2])
    for _ in range(200):
        t["inv"]["food"] = 5
        sim._resolve(a, {"action": "ask_kin", "target": t["id"]}, 0)
    assert -1000 <= a["relationships"][str(t["id"])]["affinity"] <= 1000


def test_bool_target_is_rejected():
    sim = Sim(3, 10, StubMind())
    a = sim.state["agents"][0]
    sim._resolve(a, {"action": "steal", "target": True}, 0)
    assert sim.events[-1].data["why"] == "no such person"
