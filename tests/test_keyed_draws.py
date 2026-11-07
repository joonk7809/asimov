"""T22 (engine 11): chance draws that used to depend on who else was around are keyed.

A twin with one more or one fewer person somewhere must not re-roll everyone else's dice. So the
talk roll is keyed by (speaker, tick), the partner is the present candidate with the lowest
(speaker, tick, candidate) draw, and gossip credulity and the ask odds are keyed by tick too.
"""
from hypothesis import given, settings
from hypothesis import strategies as st

from asiimov.minds import StubMind
from asiimov.rng import Rng
from asiimov.sim import TICKS_PER_DAY, Sim

N = 24


def plaza(present, tick, seed=1):
    """A fresh world where exactly `present` are awake and chatting at the plaza at `tick`.
    Returns {speaker: partner} for the talks `_talk` starts there."""
    sim = Sim(seed, N, StubMind())
    for a in sim.state["agents"]:
        a.update(asleep=True)
    for i in present:
        a = sim.state["agents"][i]
        a.update(asleep=False, place="plaza", activity="socialize")
        a["traits"]["extraversion"] = 1000  # talk on most ticks, so most speakers get a partner to compare
    sim.state["tick"] = tick
    n = len(sim.events)
    sim._talk()
    return {e.actor: e.data["with"] for e in sim.events[n:] if e.kind == "talk"}


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(0, N - 1), min_size=2, max_size=12, unique=True), st.integers(0, N - 1),
       st.integers(0, 50 * TICKS_PER_DAY), st.integers(1, 5))
def test_someone_joining_changes_a_partner_only_to_themself(present, joiner, tick, seed):
    if joiner in present:
        return
    before = plaza(present, tick, seed)
    after = plaza(present + [joiner], tick, seed)
    for speaker, partner in before.items():
        # the speaker's roll is keyed by (speaker, tick): who talks doesn't change when someone joins
        assert speaker in after
        assert after[speaker] in (partner, joiner), (speaker, partner, after[speaker])
    assert set(after) - set(before) <= {joiner}


@settings(max_examples=60, deadline=None)
@given(st.lists(st.integers(0, N - 1), min_size=3, max_size=12, unique=True), st.data(),
       st.integers(0, 50 * TICKS_PER_DAY))
def test_someone_leaving_changes_only_the_partners_they_were(present, data, tick):
    leaver = data.draw(st.sampled_from(present))
    before = plaza(present, tick)
    after = plaza([p for p in present if p != leaver], tick)
    for speaker, partner in before.items():
        if speaker == leaver:
            continue
        assert speaker in after
        if partner != leaver:
            assert after[speaker] == partner


def test_the_talk_roll_does_not_depend_on_earlier_ticks():
    """At engine 10 the roll came off a counter that moved only on ticks with company, so a busier
    past shifted every later roll. Now the same tick gives the same talks whatever came before."""
    fresh = plaza([1, 2, 3, 4], 500)
    sim = Sim(1, N, StubMind())
    for a in sim.state["agents"]:
        a.update(asleep=True)
    for i in (1, 2, 3, 4):
        a = sim.state["agents"][i]
        a.update(asleep=False, place="plaza", activity="socialize")
        a["traits"]["extraversion"] = 1000
    for t in range(400, 500):  # a hundred ticks of chatting first
        sim.state["tick"] = t
        sim._talk()
    n = len(sim.events)
    sim.state["tick"] = 500
    sim._talk()
    assert {e.actor: e.data["with"] for e in sim.events[n:] if e.kind == "talk"} == fresh


def test_keyed_draws_are_recomputable_from_their_log_entry():
    """Every talk, snub and ask receipt names a (stream, index) whose keyed draw gives its logged roll."""
    sim = Sim(2, 40, StubMind())
    sim.run(3, {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 2}]})
    rng, seen = Rng(2), set()
    for e in sim.events:
        if e.kind in ("talk", "snub"):
            assert e.data["index"] == e.tick
            seen.add(e.kind)
        elif e.kind == "receipt" and e.data.get("stream", "").startswith(("kin:", "share:")):
            assert e.data["index"] == e.tick and e.data["roll"] == rng.at(e.data["stream"], e.tick, 0, 999)
            seen.add("ask")
    assert "talk" in seen


def test_shuffle_order_is_stable_to_one_more_agent():
    """--order shuffle is a keyed priority per tick: adding an id never reorders the others."""
    sim = Sim(1, N, StubMind(), order="shuffle")
    sim.state["tick"] = 77
    base = sim._order(range(10))
    more = sim._order(list(range(10)) + [15])
    assert [i for i in more if i != 15] == base
    sim.state["tick"] = 78
    assert sim._order(range(10)) != base  # still reshuffled every tick


def test_a_second_ask_of_the_same_person_in_one_tick_gets_a_fresh_roll():
    """Review of PR #27: a refused ask can wake the mind, which may ask again in the same tick. At
    engine 11 that repeat copied the first roll, so a refused repeat was certain. Engine 12 counts the
    asker's own asks of that target within the tick."""
    from test_intents import awake, deliberate_now
    sim = Sim(1, 12, StubMind())
    fam = sim.state["families"][sim.state["agents"][0]["family"]]["members"]
    other = next(i for i in range(12) if i not in fam)
    a, b = awake(sim, [0, other], place="plaza")
    a["inv"].update(food=0, coin=10)
    b["inv"].update(food=9, coin=0)
    sim.state["tick"] = 287
    for _ in range(3):
        deliberate_now(sim, a, {"action": "ask_food", "target": b["id"]})
    r = [e.data for e in sim.events if e.kind == "receipt" and "roll" in e.data]
    assert [x["stream"] for x in r] == [f"share:{other}:0", f"share:{other}:0:1", f"share:{other}:0:2"]
    assert all(x["index"] == 287 and x["roll"] == Rng(1).at(x["stream"], 287, 0, 999) for x in r)
    sim.state["tick"] = 288  # a new tick starts the count again
    deliberate_now(sim, a, {"action": "ask_food", "target": b["id"]})
    assert [e.data for e in sim.events if e.kind == "receipt"][-1]["stream"] == f"share:{other}:0"
