"""B5 stakes sandbox (asiimov/stakes.py, never merges): no lever is engine 17 exactly; with every lever the
ledgers and invariants hold each tick, the world is deterministic, and its log replays without a mind."""
import json

from asiimov.minds import ReplayMind, StubMind
from asiimov.sim import TICKS_PER_DAY, Sim
from asiimov.stakes import ASKS, LEVERS, StakesChecker, StakesSim, StakesStub, run_world
from asiimov.stakes_pilot import PackSim


def test_no_lever_is_the_engine():
    a = Sim(1, 40, StubMind())
    a.run_until(2 * TICKS_PER_DAY)
    b = StakesSim(1, 40, StakesStub())
    b.run_until(2 * TICKS_PER_DAY)
    assert a.state_hash() == b.state_hash()
    assert [e.to_json() for e in a.events] == [e.to_json().replace("stub-stakes-v0", "stub-rules-v0") for e in b.events]


def test_all_levers_keep_ledgers_and_replay():
    player = {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 2}, {"kind": "sick", "agent": 5},
                                   {"kind": "raid", "family": 3}]}
    sim, bad, _ = run_world(LEVERS, seed=2, agents=60, days=6, player=player, check=True)
    assert bad == []
    kinds = {e.kind for e in sim.events}
    assert {"stake", "dues", "tithe", "raid", "sick", "chest_spent"} <= kinds
    assert {e.data["what"] for e in sim.events if e.kind == "stake"} == set(ASKS)
    again, _, _ = run_world(LEVERS, seed=2, agents=60, days=6, player=player)
    assert again.state_hash() == sim.state_hash()
    ev = sim.events
    pop = ev[0].data["population"]
    rep = StakesSim(2, 60, ReplayMind([e for e in ev if e.origin == "mind"]), levers=LEVERS, genesis=pop,
                    expected=ev)
    rep.run_until(ev[-1].tick, player)
    rep.finish()
    assert rep.state_hash() == sim.state_hash() and len(rep.events) == len(ev)


def test_pack_keeps_both_prompts_and_changes_nothing():
    a = PackSim(1, 50, StakesStub(), levers=("dues", "sickness"))
    a.run_until(3 * TICKS_PER_DAY)
    b = StakesSim(1, 50, StakesStub(), levers=("dues", "sickness"))
    b.run_until(3 * TICKS_PER_DAY)
    assert a.state_hash() == b.state_hash()
    assert a.kept and all(k["full"] != k["no_text"] and "Why you stop to think" in k["no_text"] for k in a.kept)
    assert all(json.dumps(k)  for k in a.kept)
