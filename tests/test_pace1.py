"""pace-1 (Steering 2026-11-19 07:39Z; results/s16b/pt11/line7-diagnosis.md): while a player's line awaits its answer
(a say among the tick's acts, or a villager whose answer to it is due), a plan or reflect call waits one tick. It is
never held twice, it is keyed on the log only (replay is exact), and a world with no player never holds one."""
import random

import pytest

from asiimov.invariants import Checker
from asiimov.minds import StubMind
from asiimov.replay import replay
from asiimov.sim import TICKS_PER_DAY, Sim


def _waits(events):
    """For every `deferred` event, the tick its plan or reflect request was filed (the mind event's tick)."""
    out = []
    for d in (e for e in events if e.kind == "deferred"):
        kind = "mind_" + d.data["kind"]
        filed = next(e.tick for e in events if e.kind == kind and e.actor == d.actor and e.tick >= d.tick)
        out.append((d.tick, filed, d.data["kind"]))
    return out


def _talker(seed, ticks, every=1.0, start=None):
    """A player who says a line to someone here at most ticks, from the hour before dawn through the evening."""
    sim = Sim(seed, 40, StubMind())
    sim.run_until(start if start is not None else TICKS_PER_DAY - 6)
    ag = sim.state["agents"]
    me = next(a["id"] for a in ag if a["role"] == "guard")
    sim.step([{"kind": "possess", "agent": me}])
    r = random.Random(seed)
    check = Checker(sim)
    for _ in range(ticks):
        here = [o["id"] for o in ag if o["id"] != me and o["place"] is not None and o["place"] == ag[me]["place"]
                and not o["asleep"]]
        acts = []
        if here and r.random() < every:
            acts.append({"kind": "act", "action": "say", "target": r.choice(here), "text": "Morning, how are you?"})
        assert check.step(acts) == []
    sim.finish()
    return sim


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_held_call_runs_the_next_tick_and_never_waits_twice(seed):
    sim = _talker(seed, TICKS_PER_DAY + 30)  # a player talking every tick it can, through a dawn and a dusk
    waits = _waits(sim.events)
    assert {k for _, _, k in waits} == {"plan", "reflect"}, waits  # not vacuous: both kinds were held
    assert all(filed == held + 1 for held, filed, _ in waits), waits
    held = [(e.actor, e.tick) for e in sim.events if e.kind == "deferred"]
    for actor, tick in held:  # the call filed after a hold is never itself held
        assert (actor, tick + 1) not in held
    replay(sim.events)  # keyed on the log: a session with holds replays exactly


def test_holds_cover_the_say_tick_and_the_reply_tick_only():
    sim = _talker(5, 60, every=0.3)
    said = {e.tick for e in sim.events if e.kind == "said"}
    for e in (e for e in sim.events if e.kind == "deferred"):
        assert e.tick in said or e.tick - 1 in said, e.tick
    replay(sim.events)


def test_no_player_no_hold():
    sim = Sim(7, 40, StubMind())
    sim.run(2)
    assert not any(e.kind == "deferred" for e in sim.events) and "deferred" not in sim.state
