"""An inhabitant's act pulls the next tick forward (readable-world E3, plan/readable-world-spec.md).

At the live pace (20 s a tick) a player's act waited up to a whole tick: p95 18.9 s from click to applied
in the blind pair. Now a durable act runs the next tick at once, at least PULL_MIN_S after the last tick
started. The world log is the same as a batch run with the same acts at the same ticks."""
import asyncio
import json
import os
import time

from asiimov import live
from asiimov.minds import StubMind
from test_inhabitant import batch_of, intent
from test_live import WS


def session(tmp_path, pull_min, acts=4):
    """A running world at 50 s a tick, an inhabitant acting `acts` times. Returns (seconds to each applied
    ack, the run dir)."""
    out = str(tmp_path / f"w{pull_min}")
    runner = live.LiveRunner(out, StubMind(), 4, 12)
    for _ in range(60):
        runner.step()
    waits = []

    async def main():
        ready = asyncio.get_running_loop().create_future()
        task = asyncio.ensure_future(live.serve(runner, port=0, tps=0.02, paused=True, pull_min=pull_min,
                                                started=lambda port, hub: ready.set_result(port)))
        port = await ready
        c = await WS.connect(port)
        c.send({"type": "hello", "protocol": 2, "scope": "inhabitant", "agent": 3})
        w = await c.until("welcome")
        god = await WS.connect(port)
        god.send({"type": "hello", "protocol": 2})
        await god.until("welcome")
        god.send({"type": "control", "op": "run"})
        await c.until("tick")  # leaving pause steps once at once; the pace starts from there
        seen = runner.sim.state["tick"]
        for k in range(acts):
            t0 = time.monotonic()
            c.send(intent(f"a{k}", "act", {"action": "wait"}, seen))
            try:
                while True:
                    m = await asyncio.wait_for(c.recv(), 3.0)
                    if m["type"] == "tick":
                        seen = m["view"]["tick"]
                    if m["type"] == "ack" and m["id"] == f"a{k}":
                        assert m["status"] == "applied", m
                        waits.append(time.monotonic() - t0)
                        break
            except asyncio.TimeoutError:
                waits.append(None)
                break
        god.send({"type": "control", "op": "pause"})
        await god.until("status")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    runner.finish()
    return waits, out


def test_an_act_lands_within_the_pull_floor_not_the_pace(tmp_path):
    waits, out = session(tmp_path, 1.0)
    assert len(waits) == 4 and all(w is not None and w < 2.5 for w in waits), waits
    assert [l.rstrip("\n") for l in open(os.path.join(out, "events.jsonl"))] == batch_of(out)
    side = [json.loads(l) for l in open(os.path.join(out, "runner.jsonl"))]
    # Each act pulled a tick, or (E4) arrived while one ran and joined it at its fixed point.
    late = sum(r["kind"] == "received" and r.get("late") is True for r in side)
    assert sum(r["kind"] == "pulled" for r in side) + late == 4


def test_without_the_pull_an_act_waits_out_the_pace(tmp_path):
    waits, _ = session(tmp_path, None, acts=1)
    assert waits == [None]  # 50 s a tick: no ack in 3 s
