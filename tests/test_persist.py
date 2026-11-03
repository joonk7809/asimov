"""Crash-safe runs: a resumed run must end with the same bytes an uninterrupted run writes."""
import json
import os
import subprocess
import sys

import pytest

from asiimov import persist
from asiimov.events import read_jsonl, write_jsonl
from asiimov.minds import NoisyMind, ReplayDivergence, StubMind
from asiimov.replay import replay
from asiimov.sim import TICKS_PER_DAY, Sim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRE = {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 1}]}
END = 3 * TICKS_PER_DAY
EVERY = 50  # snapshots off the day boundary, so resume can't lean on day alignment


class Dies(StubMind):
    """Raises at one tick, like an LLM server going away."""

    def __init__(self, at):
        self.at = at

    def decide(self, req):
        if req.tick >= self.at:
            raise RuntimeError("server gone")
        return super().decide(req)


class NoisyDies(NoisyMind):
    def __init__(self, at):
        self.at = at

    def decide(self, req):
        if req.tick >= self.at:
            raise RuntimeError("server gone")
        return super().decide(req)


class Counting(NoisyMind):
    def __init__(self):
        self.ticks = []

    def decide(self, req):
        self.ticks.append(req.tick)
        return super().decide(req)


def full_run(out, mind, end=END):
    run = persist.start(str(out), 4, 16, mind, player=FIRE, snapshot_every=EVERY)
    run.advance(end)
    run.finish()
    return run


def log_bytes(out):
    return (out / "events.jsonl").read_bytes()


def first_wake_tick_after(t):
    """A tick where rule events are emitted before the mind is asked, so a crash there leaves a partial tick."""
    s = Sim(4, 16, StubMind())
    s.run_until(END, FIRE)
    return min(e.tick for e in s.events if e.kind == "wake" and e.tick > t)


def test_resume_after_a_crash_writes_the_uninterrupted_log_bit_for_bit(tmp_path):
    ref = full_run(tmp_path / "ref", StubMind())
    crash_at = first_wake_tick_after(TICKS_PER_DAY + 70)  # mid-tick, events already emitted that tick
    assert crash_at % EVERY and crash_at % TICKS_PER_DAY
    run = persist.start(str(tmp_path / "b"), 4, 16, Dies(crash_at), player=FIRE, snapshot_every=EVERY)
    with pytest.raises(RuntimeError):
        run.advance(END)
    crash = read_jsonl(str(tmp_path / "b" / "crash.jsonl"))
    assert crash and {e.tick for e in crash} == {crash_at}

    resumed = persist.resume(str(tmp_path / "b"), StubMind(), snapshot_every=EVERY)
    resumed.advance(END)
    resumed.finish()
    assert log_bytes(tmp_path / "b") == log_bytes(tmp_path / "ref")
    assert resumed.sim.state_hash() == ref.sim.state_hash()
    assert resumed.sim.metrics == ref.sim.metrics
    assert sorted(os.listdir(tmp_path / "b" / "snapshots")) == sorted(os.listdir(tmp_path / "ref" / "snapshots"))
    assert replay(read_jsonl(str(tmp_path / "b" / "events.jsonl"))).state_hash() == ref.sim.state_hash()


KILL = """
import os, sys
from asiimov import persist
from asiimov.minds import StubMind
class Kill(StubMind):
    def decide(self, req):
        if req.tick >= {at}:
            os._exit(9)  # no exception handlers, no buffer flushes: like SIGKILL or OOM
        return super().decide(req)
run = persist.start(sys.argv[1], 4, 16, Kill(), player={fire}, snapshot_every={every})
run.advance({end})
"""


def test_resume_after_a_hard_kill_mid_write(tmp_path):
    full_run(tmp_path / "ref", StubMind())
    at = first_wake_tick_after(2 * TICKS_PER_DAY + 10)
    code = KILL.format(at=at, fire=FIRE, every=EVERY, end=END)
    p = subprocess.run([sys.executable, "-c", code, str(tmp_path / "b")], cwd=ROOT)
    assert p.returncode == 9
    killed_tick = [l for l in log_bytes(tmp_path / "ref").splitlines(keepends=True) if json.loads(l)["tick"] == at]
    assert len(killed_tick) > 2
    with open(tmp_path / "b" / "events.jsonl", "ab") as f:  # a buffer flush cut off mid-tick, mid-line
        f.write(b"".join(killed_tick[:2]) + killed_tick[2][:20])
    run = persist.resume(str(tmp_path / "b"), StubMind(), snapshot_every=EVERY)
    run.advance(END)
    run.finish()
    assert log_bytes(tmp_path / "b") == log_bytes(tmp_path / "ref")


def test_resume_reuses_recorded_mind_outputs_and_asks_only_the_last_tick_again(tmp_path):
    """With a sampled mind a re-asked tick would differ, so a matching prefix proves the recording was reused."""
    crash_at = 2 * TICKS_PER_DAY + 40
    run = persist.start(str(tmp_path), 4, 16, NoisyDies(crash_at), player=FIRE)  # daily snapshots: a long tail
    with pytest.raises(RuntimeError):
        run.advance(END)
    crashed = read_jsonl(str(tmp_path / "events.jsonl"))
    last = crashed[-1].tick
    kept = [e for e in crashed if e.tick < last]

    mind = Counting()
    resumed = persist.resume(str(tmp_path), mind)
    resumed.advance(END)
    resumed.finish()
    final = read_jsonl(str(tmp_path / "events.jsonl"))
    assert [e.to_json() for e in final[:len(kept)]] == [e.to_json() for e in kept]
    assert min(mind.ticks) >= last  # nothing before the last recorded tick was asked again
    assert replay(final).state_hash() == resumed.sim.state_hash()


def test_a_finished_world_can_be_extended_as_if_it_never_stopped(tmp_path):
    full_run(tmp_path / "ref", StubMind())
    full_run(tmp_path / "b", StubMind(), end=TICKS_PER_DAY + 7)  # stops mid-day
    first = read_jsonl(str(tmp_path / "b" / "events.jsonl"))
    assert first[-1].kind == "end" and first[-1].tick == TICKS_PER_DAY + 7
    assert replay(first).state_hash() == first[-1].data["hash"]  # partial days replay to their end tick
    run = persist.resume(str(tmp_path / "b"), StubMind(), snapshot_every=EVERY)
    run.advance(END)
    run.finish()
    assert log_bytes(tmp_path / "b") == log_bytes(tmp_path / "ref")


def test_a_corrupt_newest_snapshot_falls_back_to_an_older_one(tmp_path):
    full_run(tmp_path / "ref", StubMind())
    run = persist.start(str(tmp_path / "b"), 4, 16, Dies(2 * TICKS_PER_DAY + 3), player=FIRE, snapshot_every=EVERY)
    with pytest.raises(RuntimeError):
        run.advance(END)
    snaps = sorted(os.listdir(tmp_path / "b" / "snapshots"))
    (tmp_path / "b" / "snapshots" / snaps[-1]).write_text('{"engine": "1", "n_ev')
    snap = json.loads((tmp_path / "b" / "snapshots" / snaps[-2]).read_text())
    snap["log_sha256"] = "0" * 64  # and the one before it doesn't match the log
    (tmp_path / "b" / "snapshots" / snaps[-2]).write_text(json.dumps(snap))
    run = persist.resume(str(tmp_path / "b"), StubMind(), snapshot_every=EVERY)
    assert run.sim.state["tick"] > int(snaps[-3][:8])  # restored from snaps[-3], then re-ran the recorded tail
    run.advance(END)
    run.finish()
    assert log_bytes(tmp_path / "b") == log_bytes(tmp_path / "ref")


def test_snapshot_state_equals_a_replay_to_that_tick(tmp_path):
    full_run(tmp_path, StubMind())
    events = read_jsonl(str(tmp_path / "events.jsonl"))
    for name in sorted(os.listdir(tmp_path / "snapshots")):
        snap = json.loads((tmp_path / "snapshots" / name).read_text())
        sim = Sim(4, 16, StubMind())
        sim.run_until(snap["tick"], FIRE)
        assert sim.state_hash() == snap["hash"] and len(sim.events) == snap["n_events"]


def test_start_refuses_to_overwrite_a_run(tmp_path):
    full_run(tmp_path, StubMind(), end=10)
    with pytest.raises(FileExistsError):
        persist.start(str(tmp_path), 4, 16, StubMind())


def test_replay_refuses_unfinished_logs_and_other_engines(tmp_path):
    s = Sim(4, 16, StubMind())
    s.run(1)
    with pytest.raises(ReplayDivergence, match="no end event"):
        replay(list(s.events))
    s.finish()
    events = [type(e)(**json.loads(e.to_json())) for e in s.events]
    events[0].data["engine"] = "0"
    with pytest.raises(ReplayDivergence, match="engine 0"):
        replay(events)
