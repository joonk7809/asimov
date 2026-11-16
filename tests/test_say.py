"""say-1 (joon 2026-11-16 05:25Z: "i should also be able to talk to the npcs"; Steering 05:28Z, medium): the
player says a line to a villager here and awake. The line is in the act event, the listener remembers it and
its next deliberation answers it, cooldown or not, in words (a `reply` event) and with an act."""
import json

import pytest

from asiimov import sim as S
from asiimov.minds import ScriptMind, StubMind
from asiimov.replay import replay
from asiimov.sim import Sim, TICKS_PER_DAY


class Rec(StubMind):
    """The stub, keeping every request it is asked."""
    def __init__(self):
        super().__init__()
        self.reqs = []

    def decide(self, req):
        self.reqs.append(req)
        return super().decide(req)


def body_with_company(mind, seed=4, n=20):
    sim = Sim(seed, n, mind)
    sim.run_until(TICKS_PER_DAY // 2)
    ag = sim.state["agents"]
    for b in ag:
        if b["asleep"] or b["place"] is None or b["role"] in ("child", "elder"):
            continue
        others = [o["id"] for o in ag if o is not b and o["place"] == b["place"] and not o["asleep"]]
        if others:
            sim.step([{"kind": "possess", "agent": b["id"]}])
            others = [o for o in others if ag[o]["place"] == b["place"] and not ag[o]["asleep"]]
            if others:
                return sim, b["id"], others[0]
    raise AssertionError("nobody to talk to")


def say(sim, me, to, text):
    sim.step([{"kind": "act", "action": "say", "target": to, "text": text}])
    return [e for e in sim.events if e.tick == sim.state["tick"] - 1]


def test_a_said_line_is_remembered_and_answered_next_tick_in_words():
    mind = Rec()
    sim, me, to = body_with_company(mind)
    t = sim.state["agents"][to]
    t["cooldown_until"] = sim.state["tick"] + 50  # just deliberated: a spoken line is answered anyway
    evs = say(sim, me, to, 'Do you know where the "good" well is?')
    said = [e for e in evs if e.kind == "said"]
    assert len(said) == 1 and said[0].actor == me and said[0].data["with"] == to
    assert said[0].data["text"] == 'Do you know where the "good" well is?'
    rec = next(e for e in evs if e.kind == "receipt" and e.actor == me and e.data["action"] == "say")
    assert rec.data["ok"] and rec.data["why"] == "said" and said[0].id in rec.parents
    me_name = sim.state["agents"][me]["name"]
    assert any(f'{me_name} said to me: "Do you know where the "good" well is?"' in n for n in t["recent"])
    n0 = len(mind.reqs)
    evs = [e for e in sim.events if e.tick >= sim.state["tick"]]
    sim.step()
    req = next(r for r in mind.reqs[n0:] if r.agent_id == to)
    assert req.kind == "deliberate"
    assert f"Why you stop to think: {me_name} (#{me}) said to you: \"Do you know where the 'good' well is?\"." in req.prompt
    assert S.SPOKEN_LINE.strip() in req.prompt
    new = [e for e in sim.events if e.tick == sim.state["tick"] - 1]
    reply = next(e for e in new if e.kind == "reply")
    assert reply.actor == to and reply.data["with"] == me and reply.data["says"] in ("Good to hear from you.", "I've got things to do.")
    mind_ev = sim.events[reply.parents[0]]
    assert mind_ev.origin == "mind" and mind_ev.actor == to and mind_ev.parents == [said[0].id]
    # Its words come before its act, and both sides remember them.
    assert reply.id < max(e.id for e in new if e.actor == to)
    assert any(f'said to me: "{reply.data["says"]}"' in n for n in sim.state["agents"][me]["recent"])


def test_saying_to_someone_absent_or_asleep_is_refused_and_nothing_is_heard():
    sim, me, to = body_with_company(StubMind())
    ag = sim.state["agents"]
    far = next(a["id"] for a in ag if a["id"] != me and a["place"] not in (None, ag[me]["place"]))
    evs = say(sim, me, far, "Hello?")
    assert [(e.data["why"]) for e in evs if e.kind == "receipt" and e.actor == me] == ["target not here"]
    assert not any(e.kind == "said" for e in evs) and ag[far]["occasion"] is None or ag[far]["occasion"]["why"] != "spoken"
    # The tick recomputes sleep before acts apply, so a sleeping listener is set up at _say itself.
    ag[to]["asleep"] = True
    sim._acts, sim._acted = {}, set()
    n = len(sim.events)
    eid = sim._player({"kind": "act", "action": "say", "target": to, "text": "Wake up"})
    sim._acts.clear()
    sim._say(ag[me], eid, to)
    evs = sim.events[n:]
    assert [(e.data["why"]) for e in evs if e.kind == "receipt" and e.actor == me] == ["target asleep"]
    assert not any(e.kind == "said" for e in evs)


def test_only_say_takes_text_and_the_text_is_checked():
    sim, me, to = body_with_company(StubMind())
    for act in ({"kind": "act", "action": "say", "target": to},
                {"kind": "act", "action": "say", "target": to, "text": ""},
                {"kind": "act", "action": "say", "target": to, "text": "x" * (S.SAY_MAX + 1)},
                {"kind": "act", "action": "say", "target": to, "text": "two\nlines"},
                {"kind": "act", "action": "say", "text": "nobody"},
                {"kind": "act", "action": "work", "text": "hi"}):
        with pytest.raises(ValueError):
            sim._player(act)


def test_a_newer_line_replaces_a_pending_one():
    """Steering 05:28Z: a target with a spoken occasion pending takes only the newest line (one call per tick)."""
    sim, me, to = body_with_company(StubMind())
    t = sim.state["agents"][to]
    sim._acts, sim._acted = {}, set()
    for text in ("first", "second"):
        eid = sim._player({"kind": "act", "action": "say", "target": to, "text": text})
        sim._acts.clear()
        sim._say(sim.state["agents"][me], eid, to)
    assert t["occasion"]["why"] == "spoken" and t["occasion"]["detail"] == "second"


def test_a_world_with_talk_replays_exactly():
    for mind in (StubMind(), ScriptMind()):
        sim, me, to = body_with_company(mind)
        say(sim, me, to, "Good morning. How is the harvest?")
        for _ in range(3):
            sim.step()
        sim.finish()
        again = replay(list(sim.events))
        assert [e.to_json() for e in again.events] == [e.to_json() for e in sim.events]
        assert any(e.kind == "reply" for e in sim.events)


def test_the_inhabitant_sees_the_words():
    from asiimov import inhabitant
    sim, me, to = body_with_company(StubMind())
    say(sim, me, to, "Nice day.")
    sim.step()
    seen = [inhabitant.scrub(json.loads(e.to_json()), me, sim.events) for e in sim.events if e.kind in ("said", "reply")]
    assert [x["kind"] for x in seen] == ["said", "reply"]
    assert seen[1]["data"]["with"] == me and seen[1]["data"]["says"]


def test_a_pending_line_is_not_displaced_by_a_more_salient_occasion():
    """The 8B check's dry run: an intent_done occasion (salience above the threshold) took the spoken line's
    place, and the villager never answered. A pending line keeps its place until it is answered."""
    sim, me, to = body_with_company(StubMind())
    t = sim.state["agents"][to]
    sim._acts, sim._acted = {}, set()
    eid = sim._player({"kind": "act", "action": "say", "target": to, "text": "Hello there"})
    sim._acts.clear()
    sim._say(sim.state["agents"][me], eid, to)
    sim._occasion(t, "intent_done", 999, detail="spend time with a friend")
    assert t["occasion"]["why"] == "spoken" and t["occasion"]["detail"] == "Hello there"
