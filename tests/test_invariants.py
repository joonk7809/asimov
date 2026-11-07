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

from asiimov.events import Event
from asiimov.invariants import Checker
from asiimov.minds import MindReply, NoisyMind, ReplayDivergence, StubMind
from asiimov.replay import replay
from asiimov.rng import Rng
from asiimov.sim import FARMER_SELL_PRICE, FOOD_PRICE, TICKS_PER_DAY, Sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = ["rng.py", "params.py", "sim.py", "memory.py", "events.py", "replay.py", "worldmap.py"]
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
    # engine 7: intents, person steps and gifts
    '{"action":"seek","target":%d,"goal":"%s","then":[{"do":"give_coin","target":%d},{"do":"steal","target":%d},'
    '{"do":"ask_kin","target":%d}]}',
    '{"action":"give_food","target":%d,"goal":"g","then":"x"}', '{"action":"give_coin","target":%d,"goal":"","then":[]}',
    '{"action":"avoid","target":%d,"goal":7,"then":[1,null,{"do":"fly"},{"do":"seek","target":true}]}',
    '{"action":"work","target":null,"goal":"%s","then":[{"do":"shop"},{"do":"give_food","target":%d},'
    '{"do":"seek","target":%d},{"do":"avoid","target":%d},{"do":"rest"}]}',
    '{"action":"wait","target":null,"goal":"g","then":[{"do":"steal","target":%d},{"do":"ask_kin","target":%d}]}',
    '{"plan":[{"activity":"shop","start_h":0,"end_h":24}]}',
    # engine 10: asking or buying food from anyone here, as an action and as steps
    '{"action":"ask_food","target":%d}', '{"action":"buy_from","target":%d}', '{"action":"buy_from","target":null}',
    '{"action":"ask_food","target":%d,"goal":"%s","then":[{"do":"buy_from","target":%d},{"do":"ask_food","target":%d}]}',
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
def stepped_run(sim: Sim, days: int, player: dict):
    """Step tick by tick, checking state invariants and the food/coin ledger after every tick."""
    checker = Checker(sim)
    for _ in range(days * TICKS_PER_DAY):
        bad = checker.step(player.get(sim.state["tick"], []))
        assert not bad, bad[:5]


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
@pytest.mark.parametrize("order", ["id", "shuffle"])
def test_world_hash_does_not_depend_on_python_hash_seed(order):
    code = ("from asiimov.sim import Sim; from asiimov.minds import StubMind; "
            f"s = Sim(4, 40, StubMind(), order={order!r}); s.run(2, {{200: [{{'kind': 'fire', 'family': 1}}]}}); "
            "print(s.state_hash())")
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
    a["place"], t["place"], t["asleep"] = "plaza", "plaza", False  # ask_kin needs them together
    for _ in range(200):
        t["inv"]["food"] = 5
        sim._resolve(a, {"action": "ask_kin", "target": t["id"]}, 0)
    assert -1000 <= a["relationships"][str(t["id"])]["affinity"] <= 1000


def test_bool_target_is_rejected():
    sim = Sim(3, 10, StubMind())
    a = sim.state["agents"][0]
    sim._resolve(a, {"action": "steal", "target": True}, 0)
    assert sim.events[-1].data["why"] == "no such person"


def test_missing_target_gets_its_own_receipt_reason():
    """'You named nobody' and 'nobody by that id' are different mistakes; F4 needs them apart."""
    sim = Sim(3, 10, StubMind())
    a = sim.state["agents"][0]
    for action in ("ask_kin", "steal"):
        for out in ({"action": action, "target": None}, {"action": action}):
            sim._resolve(a, out, 0)
            assert sim.events[-1].data == {"action": action, "ok": False, "why": "no target named"}
        sim._resolve(a, {"action": action, "target": 99}, 0)
        assert sim.events[-1].data["why"] == "no such person"


def test_one_buy_covers_the_rest_of_the_day_capped_by_coin_and_stock():
    sim = Sim(2, 10, StubMind())
    a = sim.state["agents"][0]
    a["place"], a["inv"]["food"], a["needs"]["hunger"] = "market", 0, 600
    a["sleep_tick"] = sim.state["tick"] + 60
    want = sim._meals_left_today(a)
    assert want >= 2  # an hour of rising hunger past the next meal
    a["inv"]["coin"], sim.state["market_food"] = 100, 100
    sim._resolve(a, {"action": "buy_food"}, 0)
    assert sim.events[-1].data["food"] == a["inv"]["food"] == want
    a["inv"]["food"], a["inv"]["coin"] = 0, FOOD_PRICE * (want - 1) + 1  # coin for one fewer
    sim._resolve(a, {"action": "buy_food"}, 0)
    assert a["inv"]["food"] == want - 1 and a["inv"]["coin"] == 1
    a["inv"]["food"], a["inv"]["coin"], sim.state["market_food"] = 0, 100, 1  # one left on the stall
    sim._resolve(a, {"action": "buy_food"}, 0)
    assert a["inv"]["food"] == 1 and sim.state["market_food"] == 0


def test_a_hungry_child_eats_from_the_family_member_with_most_food():
    sim = Sim(3, 60, StubMind())
    fam = next(f["members"] for f in sim.state["families"]
               if sum(sim.state["agents"][m]["role"] in ("child", "elder") for m in f["members"]) >= 1
               and len(f["members"]) >= 3)
    kid = next(sim.state["agents"][m] for m in fam if sim.state["agents"][m]["role"] in ("child", "elder"))
    others = [sim.state["agents"][m] for m in fam if m != kid["id"]]
    for o in others:
        o["inv"]["food"] = 1
    others[-1]["inv"]["food"] = 4
    kid["inv"]["food"], kid["needs"]["hunger"] = 0, 700
    sim._eat(kid)
    assert sim.events[-1].kind == "eat" and sim.events[-1].data == {"household": others[-1]["id"]}
    assert others[-1]["inv"]["food"] == 3 and kid["needs"]["hunger"] == 250
    for o in others:  # everyone down to their last one: they keep it
        o["inv"]["food"] = 1
    kid["needs"]["hunger"], n = 700, len(sim.events)
    sim._eat(kid)
    assert len(sim.events) == n and kid["needs"]["hunger"] == 700


def test_working_adults_do_not_eat_from_the_household():
    sim = Sim(3, 60, StubMind())
    fam = next(f["members"] for f in sim.state["families"] if len(f["members"]) >= 2
               and any(sim.state["agents"][m]["role"] not in ("child", "elder") for m in f["members"]))
    adult = next(sim.state["agents"][m] for m in fam if sim.state["agents"][m]["role"] not in ("child", "elder"))
    for m in fam:
        sim.state["agents"][m]["inv"]["food"] = 5
    adult["inv"]["food"], adult["needs"]["hunger"], n = 0, 700, len(sim.events)
    sim._eat(adult)
    assert len(sim.events) == n


def test_a_working_adult_also_buys_what_the_dependants_at_home_still_need():
    sim = Sim(3, 60, StubMind())
    sim.run(1)  # everyone has a bedtime
    fam = next(f["members"] for f in sim.state["families"]
               if any(sim.state["agents"][m]["role"] in ("child", "elder") for m in f["members"])
               and any(sim.state["agents"][m]["role"] not in ("child", "elder") for m in f["members"]))
    ag = [sim.state["agents"][m] for m in fam]
    adult = next(a for a in ag if a["role"] not in ("child", "elder"))
    deps = [a for a in ag if a["role"] in ("child", "elder")]
    for a in ag:
        a["inv"]["food"], a["needs"]["hunger"], a["sleep_tick"] = 0, 600, sim.state["tick"] + 60
    deps[0]["inv"]["food"] = 1
    want = sum(sim._meals_left_today(a) for a in [adult] + deps) - 1
    adult["place"], adult["inv"]["coin"], sim.state["market_food"] = "market", 1000, 1000
    sim._resolve(adult, {"action": "buy_food"}, 0)
    assert adult["inv"]["food"] == want


def test_market_stock_above_the_cap_spoils_overnight_and_is_logged():
    sim = Sim(1, 100, StubMind())
    cap = sim._market_cap()
    assert 450 < cap < 650  # two days of meals for about 100 agents at about 2.7 a day
    sim.state["market_food"] = cap + 40
    sim.run(1)
    spoils = [e for e in sim.events if e.kind == "spoil"]
    assert spoils and spoils[0].data["cap"] == cap and sim.state["market_food"] <= cap
    sim2 = Sim(1, 100, StubMind())
    sim2.run(1)  # a normal first day stays under the cap
    assert not any(e.kind == "spoil" for e in sim2.events)


def test_a_days_wage_buys_a_days_meals():
    from asiimov.sim import FARM_YIELD, WORK_WAGE
    assert WORK_WAGE == FARM_YIELD * FARMER_SELL_PRICE  # a unit of any work earns what a farmer's unit sells for
    sim = Sim(1, 100, StubMind())
    sim.run(2)
    pays = {e.data["coin"] for e in sim.events if e.kind == "produce" and "coin" in e.data}
    assert pays == {WORK_WAGE}
