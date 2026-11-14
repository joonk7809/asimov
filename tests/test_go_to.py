"""The player walks its body to a named place (go-to-1, plan/readable-world-spec.md E1, Steering 2026-11-14).

`go_to` with a `place` on the map is a player-only act: a one-off step that walks the body there and keeps
it there for an activity's time, unless another act replaces it. Its receipt says when it arrives, and a
second receipt marks the arrival. A mind is never offered it, and a world with no player is unchanged."""
import json

import pytest

from asiimov.events import Event
from asiimov.invariants import state_violations
from asiimov.minds import StubMind
from asiimov.replay import replay
from asiimov.sim import DEFAULT_STEP_TICKS, DELIBERATE_ACTIONS, PLAYER_ACTIONS, TICKS_PER_DAY, Sim


def awake_body(seed=4, n=14):
    """A world at midday with an awake, unwalking body possessed. Returns (sim, body id)."""
    sim = Sim(seed, n, StubMind())
    sim.run_until(TICKS_PER_DAY // 2)
    b = next(a for a in sim.state["agents"] if not a["asleep"] and a["place"] is not None
             and a["sleep_tick"] > sim.state["tick"] + 30)
    sim.step([{"kind": "possess", "agent": b["id"]}])
    return sim, b["id"]


def go(place):
    return [{"kind": "act", "action": "go_to", "place": place}]


def receipts(sim, aid):
    return [e for e in sim.events if e.kind == "receipt" and e.actor == aid and e.data["action"] == "go_to"]


def far_place(sim, aid):
    a = sim.state["agents"][aid]
    return next(p for p in ("woods", "field", "market", "workshop", "plaza") if p != a["place"])


def test_go_to_is_the_players_alone():
    assert "go_to" in PLAYER_ACTIONS and "go_to" not in DELIBERATE_ACTIONS


def test_the_body_walks_there_says_when_and_marks_the_arrival():
    sim, b = awake_body()
    p = far_place(sim, b)
    sim.step(go(p))
    first = receipts(sim, b)[-1]
    assert first.data["ok"] and first.data["why"] == "heading there" and first.data["place"] == p
    arrive = first.data["arrive"]
    assert arrive > first.tick
    while sim.state["tick"] <= arrive:
        sim.step([])
        assert not state_violations(sim)
    a = sim.state["agents"][b]
    assert a["place"] == p
    last = receipts(sim, b)[-1]
    assert last.data == {"action": "go_to", "ok": True, "why": "arrived", "place": p}
    assert last.tick == arrive and last.parents[0] == first.parents[0]  # both cite the act


def test_it_stays_there_for_the_steps_time_then_lets_the_day_take_over():
    sim, b = awake_body()
    p = far_place(sim, b)
    sim.step(go(p))
    t0 = sim.state["tick"]
    for _ in range(DEFAULT_STEP_TICKS - 1):
        sim.step([])
        a = sim.state["agents"][b]
        if a["asleep"]:
            break
        if a["place"] is not None:
            assert a["place"] == p, f"left {p} at tick {sim.state['tick']} (went at {t0})"
    a = sim.state["agents"][b]
    assert a["override"] is None or a["override"]["do"] != "go" or a["override"]["left"] > 0


def test_already_there_stays_and_says_so():
    sim, b = awake_body()
    here = sim.state["agents"][b]["place"]
    sim.step(go(here))
    r = receipts(sim, b)
    assert len(r) == 1 and r[0].data["why"] == "already here" and r[0].data["arrive"] == r[0].tick
    sim.step([])
    assert sim.state["agents"][b]["place"] == here and len(receipts(sim, b)) == 1


def test_a_later_act_replaces_it():
    sim, b = awake_body()
    sim.step(go(far_place(sim, b)))
    sim.step([{"kind": "act", "action": "home"}])
    a = sim.state["agents"][b]
    assert a["override"]["do"] == "home"


def test_release_drops_a_go_step():
    sim, b = awake_body()
    sim.step(go(far_place(sim, b)))
    sim.step([{"kind": "release"}])
    assert sim.state["agents"][b]["override"] is None
    assert not state_violations(sim)


@pytest.mark.parametrize("act", [
    {"kind": "act", "action": "go_to"},                                   # no place
    {"kind": "act", "action": "go_to", "place": "moon"},                  # not on the map
    {"kind": "act", "action": "go_to", "place": "woods", "target": 1},    # a person and a place
    {"kind": "act", "action": "home", "place": "woods"},                  # a place on another act
])
def test_bad_go_to_is_refused_before_it_is_logged(act):
    sim, _ = awake_body()
    n = len(sim.events)
    with pytest.raises(ValueError):
        sim.step([act])
    assert all(e.kind != "act" for e in sim.events[n:])


def test_a_session_of_go_tos_replays_exactly():
    sim, b = awake_body(seed=9, n=20)
    places = ["market", "woods", "plaza", "field", sim.state["agents"][b]["home"], "workshop"]
    for t in range(2 * TICKS_PER_DAY):
        acts = go(places[(t // 15) % len(places)]) if t % 15 == 0 else []
        if sim.state["agents"][b]["asleep"]:
            acts = []
        sim.step(acts)
        assert not state_violations(sim), sim.state["tick"]
    sim.finish()
    assert sum(r.data["why"] == "arrived" for r in receipts(sim, b)) >= 3
    events = [Event(**json.loads(e.to_json())) for e in sim.events]
    assert replay(events).state_hash() == sim.state_hash()


def test_the_live_runner_takes_go_to_with_a_place_and_refuses_it_without(tmp_path):
    from asiimov import live
    from test_player_body import intent
    r = live.LiveRunner(str(tmp_path / "w"), StubMind(), 4, 10)
    assert "woods" in r.verbs()["act"]["place"]
    assert r.submit(intent("p", "possess", {"agent": 1}, 0)) is None
    no = lambda i, a: r.submit(intent(i, "act", a, 0))["reason"]
    assert no("x1", {"action": "go_to"}) == "bad_args"
    assert no("x2", {"action": "go_to", "place": "moon"}) == "bad_args"
    assert no("x3", {"action": "go_to", "place": "woods", "target": 2}) == "bad_args"
    assert no("x4", {"action": "work", "place": "woods"}) == "bad_args"
    assert no("x5", {"action": "go_to", "place": 3}) == "bad_args"
    assert r.submit(intent("ok", "act", {"action": "go_to", "place": "woods"}, 0)) is None
    _, events, acks = r.step()
    assert [a["status"] for a in acks] == ["applied"] * 2
    act = next(e for e in events if e.kind == "act")
    assert act.data == {"action": "go_to", "place": "woods"}
