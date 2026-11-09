"""Write a recorded log and the engine's own positions at every tick boundary, for test_positions.gd.

    python client/godot/tests/make_fixture.py OUT_DIR [--agents 25 --days 2 --seed 1]

The reference is Sim._pos read from the live sim between ticks (worldmap.position over the sim's own
routes), never the viewer or anything rebuilt from the log. Run from the repo root.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from asiimov.minds import StubMind  # noqa: E402
from asiimov.sim import TICKS_PER_DAY, Sim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--agents", type=int, default=25)
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    sim = Sim(args.seed, args.agents, StubMind())
    player = {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 1}]}
    ticks, pos, place = [], [], []
    while sim.state["tick"] < args.days * TICKS_PER_DAY:
        ticks.append(sim.state["tick"])
        pos.append([list(sim._pos(a)) for a in sim.state["agents"]])
        place.append([a["place"] for a in sim.state["agents"]])
        sim.step(player.get(sim.state["tick"], []))
    sim.finish()
    with open(os.path.join(args.out, "events.jsonl"), "w") as f:
        for e in sim.events:
            f.write(e.to_json() + "\n")
    with open(os.path.join(args.out, "expected.json"), "w") as f:
        json.dump({"ticks": ticks, "pos": pos, "place": place}, f)
    print(f"{args.out}: {len(sim.events)} events, {len(ticks)} boundaries, {args.agents} villagers")


if __name__ == "__main__":
    main()
