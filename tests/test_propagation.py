"""Propagation metric (asiimov/propagation.py): the counts mean what the definition says.

Planted differences (test the test, S6): edit one record of a known log and check the report counts
exactly that agent, at that tick, in the right category. Then one real stub twin end to end.
"""
import copy
import json

import pytest

from asiimov.events import Event
from asiimov.propagation import report, stub_twin, stub_world
from asiimov.sim import TICKS_PER_DAY

FIRE_TICK = TICKS_PER_DAY + 60


@pytest.fixture(scope="module")
def world():
    """A 20-agent stub world and its twin with family 0's food burned on day 1."""
    base = stub_world(5, 20, 9)
    return base, stub_twin(base, {FIRE_TICK: [{"kind": "fire", "family": 0}]})


def clone(events):
    return [Event(**copy.deepcopy(e.__dict__)) for e in events]


def first_after(events, tick, kind, origin=None):
    return next(e for e in events if e.tick >= tick and e.kind == kind and (origin is None or e.origin == origin))


def test_a_rerun_without_change_reaches_nobody(world):
    base, _ = world
    r = report(base, stub_twin(base, {}))
    assert r["first_act_difference"] == {}
    assert all(v[k] == 0 for v in r["by_day"].values() for k in v if k != "complete")


def test_a_fire_shows_in_its_targets_and_the_report_is_deterministic(world):
    base, twin = world
    r = report(base, twin)
    assert r["intervention_tick"] == FIRE_TICK and r["targets"]
    assert r["by_day"]["1"]["of_which_targets"] >= 1
    assert all(t >= FIRE_TICK for t in r["first_act_difference"].values())
    days = [r["by_day"][d] for d in ("1", "3", "7")]
    for k in ("acted_differently", "beyond_targets", "beyond_targets_besides_chats", "thought_differently"):
        assert days[0][k] <= days[1][k] <= days[2][k], k  # cumulative
    assert all(d["complete"] for d in days)
    assert report(base, stub_twin(base, {FIRE_TICK: [{"kind": "fire", "family": 0}]})) == r


def planted(base, edit) -> dict:
    """Report on a 'twin' that is the base log with one record edited after a no-op player event."""
    twin = clone(base)
    edit(twin)
    return report(base, twin)


def test_one_changed_act_counts_one_agent_from_its_tick(world):
    base, _ = world
    e = first_after(base, 2 * TICKS_PER_DAY, "activity")
    r = planted(base, lambda tw: tw[e.id].data.update(place="nowhere"))
    assert r["first_act_difference"] == {str(e.actor): e.tick}
    assert r["intervention_tick"] == e.tick  # a twin without player events is read from its first difference
    assert r["by_day"]["1"]["acted_differently"] == 1 and r["by_day"]["1"]["beyond_targets_besides_chats"] == 1


def test_rng_bookkeeping_is_not_behavior(world):
    base, _ = world
    e = first_after(base, TICKS_PER_DAY, "talk")
    r = planted(base, lambda tw: tw[e.id].data.update(index=[i + 7 for i in e.data["index"]]))
    assert r["first_act_difference"] == {}


def test_a_changed_conversation_is_an_act_but_not_besides_chats(world):
    base, _ = world
    e = first_after(base, TICKS_PER_DAY, "talk")
    r = planted(base, lambda tw: tw[e.id].data.update({"with": (e.data["with"] + 1) % 20}))
    day = r["by_day"]["1"]
    assert day["acted_differently"] == 1 and day["beyond_targets_besides_chats"] == 0


def test_changed_mind_text_alone_is_thought_not_act(world):
    base, _ = world
    e = first_after(base, TICKS_PER_DAY, "mind_reflect", "mind")
    r = planted(base, lambda tw: tw[e.id].mind.update(raw_output=e.mind["raw_output"] + " "))
    day = r["by_day"]["1"]
    assert (day["acted_differently"], day["thought_differently"], day["thought_only"]) == (0, 1, 1)


def test_an_unclassified_agent_event_kind_stops_the_measurement(world):
    base, _ = world
    e = first_after(base, TICKS_PER_DAY, "eat")
    with pytest.raises(KeyError, match="unclassified"):
        planted(base, lambda tw: setattr(tw[e.id], "kind", "dance"))


def test_a_difference_before_the_intervention_is_refused(world):
    base, twin = world
    early = first_after(twin, TICKS_PER_DAY // 2, "activity")
    assert early.tick < FIRE_TICK
    bad = clone(twin)
    bad[early.id].data["place"] = "nowhere"
    with pytest.raises(ValueError, match="differently before the intervention"):
        report(base, bad)



def test_a_noise_twin_is_read_over_its_intervention_twins_days(world):
    base, _ = world
    e = first_after(base, 2 * TICKS_PER_DAY, "activity")
    noise = clone(base)
    noise[e.id].data["place"] = "nowhere"
    r = report(base, noise, t0=FIRE_TICK)
    assert r["intervention_tick"] == FIRE_TICK
    in_day_1 = e.tick < FIRE_TICK + TICKS_PER_DAY
    assert r["by_day"]["1"]["acted_differently"] == int(in_day_1) and r["by_day"]["3"]["acted_differently"] == 1


def test_whom_an_act_was_aimed_at_is_part_of_the_act(world):
    base, _ = world
    e = first_after(base, TICKS_PER_DAY, "receipt")
    r = planted(base, lambda tw: tw[e.id].data.update(victim=e.actor + 1))  # a steal receipt names its victim
    assert r["first_act_difference"] == {str(e.actor): e.tick}
    by_id = {x.id: x for x in base}
    rec = next(x for x in base if x.kind == "receipt" and x.tick >= TICKS_PER_DAY and any(
        by_id[p].origin == "mind" and by_id[p].data["kind"] == "deliberate" for p in x.parents))
    mind = next(by_id[p] for p in rec.parents if by_id[p].origin == "mind")
    out = json.loads(mind.mind["raw_output"])
    out["target"] = 999  # the receipt doesn't record it; the reply it answers does
    r = planted(base, lambda tw: tw[mind.id].mind.update(raw_output=json.dumps(out)))
    assert r["first_act_difference"].get(str(rec.actor)) == rec.tick


def test_a_missing_act_is_not_blamed_on_the_twins_next_one(world):
    base, _ = world
    e = first_after(base, 2 * TICKS_PER_DAY, "activity")
    twin = [x for x in clone(base) if x.id != e.id]
    from asiimov.propagation import first_differences
    tick, ev = first_differences(base, twin)["act"][e.actor]
    assert tick == e.tick and ev is None


def test_a_hostile_deeply_nested_reply_does_not_crash_the_metric(world):
    base, _ = world
    e = first_after(base, TICKS_PER_DAY, "mind_deliberate", "mind")
    r = planted(base, lambda tw: tw[e.id].mind.update(raw_output="[" * 20000))
    assert r["by_day"]["1"]["thought_differently"] == 1
