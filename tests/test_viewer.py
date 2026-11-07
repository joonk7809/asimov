"""Replay viewer (asiimov/viewer.py): who-is-where read from the log matches the world, tick by tick.

The viewer never replays; it reads places from wake, activity and sleep events. That is only right if
every place change is emitted (AGENTS.md: no silent state edits), so this checks it against the live sim.
"""
import gzip
import json

import pytest

from asiimov.minds import StubMind
from asiimov.sim import TICKS_PER_DAY, Sim
from asiimov.viewer import build, main, moves, read_log, render

FIRE = {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 2}]}


@pytest.fixture(scope="module")
def world():
    """A 30-agent, 3-day stub world with a fire, and each villager's (place, activity) after every tick."""
    sim = Sim(7, 30, StubMind())
    truth = []
    for tick in range(3 * TICKS_PER_DAY):
        sim.step(FIRE.get(tick, []))
        truth.append([(a["place"], a["activity"]) for a in sim.state["agents"]])
    sim.finish()
    return [json.loads(e.to_json()) for e in sim.events], truth


def at(m, tick):
    return next(x for x in reversed(m) if x[0] <= tick)[1:]


def test_places_and_activities_from_the_log_match_the_world_every_tick(world):
    events, truth = world
    mv = moves(events)
    for tick, row in enumerate(truth):
        for aid, (place, activity) in enumerate(row):
            assert at(mv[aid], tick) == [place, activity], (tick, aid)
    # the check covered the cases that matter: visits to a person, the public places, and sleep at home
    seen = {(p, a) for row in truth for p, a in row}
    assert "visit" in {a for _, a in seen} and "sleep" in {a for _, a in seen}
    assert {"market", "field", "plaza"} <= {p for p, _ in seen}


def test_page_embeds_the_log_and_cant_be_broken_by_text_in_it(world):
    events, _ = world
    events = json.loads(json.dumps(events))
    m = next(e for e in events if e["origin"] == "mind")
    m["mind"]["raw_output"] = '{"reason":"</script><script>alert(1)</script>"}'
    html = render(build(events, "test"))
    assert html.count("</script>") == html.count("<script")  # the mind's text didn't close a tag
    blob = html.split('<script id="view" type="application/json">', 1)[1].split("</script>", 1)[0]
    view = json.loads(blob)
    assert len(view["events"]) == len(events) and view["complete"] and view["end_tick"] == 3 * TICKS_PER_DAY
    assert "population" not in view["events"][0][5]  # carried once, as view["agents"]


def test_a_cut_off_gzipped_log_still_opens(world, tmp_path):
    events, _ = world
    lines = [json.dumps(e) for e in events[:-1]]  # no end event, as when a run crashed
    path = tmp_path / "events.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n" + lines[-1][:40])  # and a half-written last line
    got, bad = read_log(str(tmp_path))
    assert bad == 1 and len(got) == len(events) - 1
    main([str(path), "--out", str(tmp_path / "v.html")])
    assert "application/json" in (tmp_path / "v.html").read_text()
    view = build(got, "x", bad)
    assert not view["complete"] and view["end_tick"] == got[-1]["tick"] + 1
