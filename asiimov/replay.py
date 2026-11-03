"""Replay a recorded log without calling any model, checking every event matches."""
from collections import defaultdict

from .events import Event
from .minds import ReplayMind
from .sim import TICKS_PER_DAY, Sim


def replay(events: list[Event]) -> Sim:
    genesis = events[0]
    assert genesis.kind == "genesis"
    pop = genesis.data["population"]
    mind_events = [e for e in events if e.origin == "mind"]
    player = defaultdict(list)
    for e in events:
        if e.origin == "player":
            player[e.tick].append({"kind": e.kind, **e.data})
    last_tick = events[-1].tick
    sim = Sim(genesis.data["seed"], len(pop["agents"]), ReplayMind(mind_events), genesis=pop, expected=events,
              order=genesis.data["order"])
    days = (last_tick + 1) // TICKS_PER_DAY
    sim.run(days, player)
    if len(sim.events) != len(events):
        from .minds import ReplayDivergence
        raise ReplayDivergence(f"replay produced {len(sim.events)} events, log has {len(events)}")
    return sim
