"""client/godot/tools/make_playback.py on logs that stop mid-write (review of #100): a torn last line, or a cut
inside a tick, gives the pack of the whole ticks before it. A log copied from a running world is such a log."""
import importlib.util
import os
import random

import pytest

from asiimov.minds import ReplayDivergence, StubMind
from asiimov.sim import Sim

_spec = importlib.util.spec_from_file_location(
    "make_playback", os.path.join(os.path.dirname(__file__), "..", "client", "godot", "tools", "make_playback.py"))
make_playback = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_playback)


def _log(end: bool, ticks: int = 80) -> bytes:
    sim = Sim(3, 12, StubMind())
    sim.run_until(ticks, {40: [{"kind": "fire", "family": 1}]})
    if end:
        sim.finish()
    return "".join(e.to_json() + "\n" for e in sim.events).encode()


def test_a_finished_log_and_a_live_log_give_the_same_frames():
    ev_end, hold_end, exp_end, last_end = make_playback.replay_frames(_log(True), 10)
    ev, hold, exp, last = make_playback.replay_frames(_log(False), 10)
    assert (last_end, last) == (79, 79) and ev_end[-1].kind == "end" and ev == ev_end[:-1]
    assert hold == hold_end and exp == exp_end and exp["ticks"] == list(range(0, 81, 10))
    assert len({tuple(h[2:]) for h in hold if h[1] == 0}) > 1  # holdings do change


def test_a_log_cut_anywhere_gives_the_whole_ticks_before_the_cut():
    raw = _log(False)
    events, holdings, _, _ = make_playback.replay_frames(raw, 10)
    first = raw.index(b"\n") + 1
    while events[raw[:first].count(b"\n")].tick == 0:  # past tick 0: a world torn inside its start isn't a world
        first = raw.index(b"\n", first) + 1
    rng = random.Random(5)
    cuts = [rng.randrange(first, len(raw)) for _ in range(40)]
    cuts += [raw.index(b"\n", c) + 1 for c in cuts[:20]]  # the same, at a line boundary
    torn = inside = 0
    for cut in cuts:
        part = raw[:cut]
        n = part.count(b"\n")  # whole lines
        last = events[n - 1].tick
        tick_whole = n == len(events) or events[n].tick != last  # the cut fell exactly between two ticks
        keep = last if tick_whole else last - 1
        ev, hold, exp, last_tick = make_playback.replay_frames(part, 10)
        assert last_tick == keep, cut
        assert ev == [e for e in events if e.tick <= keep], cut
        assert hold == [h for h in holdings if h[0] <= keep], cut
        assert exp["ticks"] == sorted({*range(0, keep + 2, 10), keep + 1}), cut
        torn += not part.endswith(b"\n")
        inside += part.endswith(b"\n") and not tick_whole
    assert torn >= 20 and inside >= 10  # both of the review's cuts were exercised


def test_a_break_before_the_last_tick_is_still_an_error():
    lines = _log(False).split(b"\n")
    del lines[len(lines) // 2]  # an event missing mid-log is a divergence, not a short log
    with pytest.raises(ReplayDivergence):
        make_playback.replay_frames(b"\n".join(lines), 10)
