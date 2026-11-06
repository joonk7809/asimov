"""The two-phase tick: mind replies are committed in one deterministic order, whatever order they arrive in.

The stateful test drives a world whose mind answers from several threads with random delays,
alongside a twin that asks serially. The logs must match event for event, the commit order
must be the policy's, and the concurrent world's log must replay.
"""
import json
import time

from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from asiimov.events import Event
from asiimov.minds import MindReply, StubMind
from asiimov.replay import replay
from asiimov.sim import TICKS_PER_DAY, Sim


class Greedy(StubMind):
    """Deterministic per request and contest-prone: hungry agents grab market food or steal from whoever is here."""
    model = "greedy"

    def __init__(self, delays=None):
        self.delays = delays  # list of seconds, consumed by hash so any thread may get any delay

    def decide(self, req):
        if self.delays:
            time.sleep(self.delays[int(req.prompt_hash[:6], 16) % len(self.delays)])
        if req.kind != "deliberate":
            return super().decide(req)
        v = req.view
        if v["at_market"] and v["coin"] >= v["food_price"]:
            out = {"action": "buy_food", "target": None, "reason": "first come"}
        elif any(p["has_food"] for p in v["present"]):
            out = {"action": "steal", "target": next(p["id"] for p in v["present"] if p["has_food"]), "reason": "x"}
        else:
            out = {"action": "go_market", "target": None, "reason": "x"}
        return MindReply(json.dumps(out, sort_keys=True), self.model, {})


def mind_actors_by_tick(events):
    out = {}
    for e in events:
        if e.origin == "mind":
            out.setdefault(e.tick, []).append(e.actor)
    return out


class ConcurrentCommit(RuleBasedStateMachine):
    @initialize(seed=st.integers(0, 10**6), agents=st.integers(3, 24), order=st.sampled_from(["id", "shuffle"]),
                workers=st.integers(2, 16))
    def setup(self, seed, agents, order, workers):
        self.delays = [0]
        self.fast = Sim(seed, agents, Greedy(self.delays), order=order, workers=workers)
        self.serial = Sim(seed, agents, Greedy(), order=order, workers=1)
        self.order = order

    @rule(ticks=st.integers(1, 90), delays=st.lists(st.sampled_from([0, 0.0005, 0.002]), min_size=1, max_size=4))
    def advance(self, ticks, delays):
        self.delays[:] = delays
        for _ in range(ticks):
            self.fast.step()
            self.serial.step()

    @rule(family=st.integers(0, 50))
    def fire(self, family):
        fam = family % len(self.fast.state["families"])
        act = [{"kind": "fire", "family": fam}]
        self.fast.step(act)
        self.serial.step(act)

    @invariant()
    def same_world_as_serial(self):
        if not hasattr(self, "fast"):
            return
        assert len(self.fast.events) == len(self.serial.events)
        assert self.fast.events[-50:] == self.serial.events[-50:]
        assert self.fast.state_hash() == self.serial.state_hash()

    @invariant()
    def commits_follow_the_policy(self):
        if not hasattr(self, "fast") or self.order != "id":
            return
        for tick, actors in mind_actors_by_tick(self.fast.events[-400:]).items():
            assert actors == sorted(actors), (tick, actors)

    def teardown(self):
        if hasattr(self, "fast"):
            assert [e.to_json() for e in self.fast.events] == [e.to_json() for e in self.serial.events]
            self.fast.finish()  # mid-day is fine: the end event says where the log stops
            r = replay([Event(**json.loads(e.to_json())) for e in self.fast.events])
            assert r.state_hash() == self.fast.state_hash()


ConcurrentCommit.TestCase.settings = settings(max_examples=12, stateful_step_count=8, deadline=None,
                                              suppress_health_check=[HealthCheck.too_slow])
TestConcurrentCommit = ConcurrentCommit.TestCase


def _two_hungry_at_market(order, seed=1):
    sim = Sim(seed, 6, Greedy(), order=order)
    sim.state["market_food"] = 1
    for a in sim.state["agents"]:
        a.update(asleep=False, wake_tick=0, sleep_tick=10**6, place="home", activity="home", plan=[], habit="home")
        a["inv"]["food"] = 0
    for aid in (2, 4):
        a = sim.state["agents"][aid]
        # Already shopping at the market, so no step or arrival buys in the rules phase: the commit decides.
        a.update(place="market", activity="shop", plan=[{"activity": "shop", "start_h": 0, "end_h": 24}])
        a["inv"]["coin"] = 50
        a["needs"]["hunger"] = 990
    return sim


def buyer(sim):
    return next(e.actor for e in sim.events if e.kind == "receipt" and e.data["action"] == "buy_food" and e.data["ok"])


def test_id_order_gives_the_last_food_to_the_lower_id():
    sim = _two_hungry_at_market("id")
    sim.step()
    assert buyer(sim) == 2
    assert [e.data["why"] for e in sim.events if e.kind == "receipt" and e.actor == 4] == ["no stock"]


def test_shuffle_lets_either_agent_win():
    winners = set()
    for seed in range(12):
        sim = _two_hungry_at_market("shuffle", seed)
        sim.step()
        winners.add(buyer(sim))
    assert winners == {2, 4}


def test_shuffled_world_replays_exactly_and_differs_from_id_order():
    a = Sim(3, 40, Greedy(), order="shuffle")
    b = Sim(3, 40, Greedy(), order="id")
    fire = {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 2}]}
    a.run(2, fire)
    b.run(2, fire)
    a.finish()
    assert a.state_hash() != b.state_hash()
    assert replay([Event(**json.loads(e.to_json())) for e in a.events]).state_hash() == a.state_hash()
