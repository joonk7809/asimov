"""script-voice (Steering 2026-11-16 10:18Z, plan/test16-live-rule.md, "The script arm's answers to the player"):
with --mind script, a scripted villager's answer to the player is worded by the narrator from the player's line,
the stub's own act and the place facts; the act stays the stub's, the words are the mind's recorded output so
replay is exact, and the runner holds the answer for a delay drawn from the minds arm's distribution."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from asiimov import narrate
from asiimov.live import ReplyHold
from asiimov.minds import ScriptMind
from asiimov.replay import replay
from tests.test_say import body_with_company, say


class Voice:
    model = "fake-8b"
    sampling = {"temperature": 0.7}

    def __init__(self, says="Morning! The market's that way, past the plaza."):
        self.says, self.asked = says, []

    def reply(self, line, act, facts):
        self.asked.append((line, act, facts))
        return self.says


def test_the_voice_words_the_reply_and_the_act_stays_the_stubs():
    plain, voiced = ScriptMind(), ScriptMind(Voice())
    a, me, to = body_with_company(plain)
    b, me2, to2 = body_with_company(voiced)
    assert (me, to) == (me2, to2)
    for sim in (a, b):
        say(sim, me, to, "Where is the market?")
        sim.step()
    ra = next(e for e in a.events if e.kind == "reply")
    rb = next(e for e in b.events if e.kind == "reply")
    assert rb.data["says"] == voiced.voice.says and ra.data["says"] != rb.data["says"]
    line, act, facts = voiced.voice.asked[0]
    assert line == "Where is the market?" and act == ScriptMind.NEXT_WORDS[rb.data["next"]]
    assert set(facts) == {"place", "hour", "role"} and ra.data["next"] == rb.data["next"]
    # Only the words differ: the same acts, and the voice was asked only for the spoken occasion.
    acts = lambda s: [(e.kind, e.actor, json.dumps({k: v for k, v in e.data.items() if k not in ("says", "text")}, sort_keys=True))
                      for e in s.events if e.origin == "rule" and e.kind != "reply"]
    assert acts(a) == acts(b) and len(voiced.voice.asked) == 1
    mev = b.events[rb.parents[0]]
    assert mev.mind["model"] == "script-v0+voice:fake-8b" and mev.mind["sampling"]["voice"] == Voice.sampling


def test_a_voiced_world_replays_exactly_without_the_voice():
    mind = ScriptMind(Voice())
    sim, me, to = body_with_company(mind)
    say(sim, me, to, "Good morning. How is the harvest?")
    for _ in range(3):
        sim.step()
    sim.finish()
    again = replay(list(sim.events))
    assert [e.to_json() for e in again.events] == [e.to_json() for e in sim.events]


def test_no_words_from_the_voice_keeps_the_scripts_sentence():
    a, me, to = body_with_company(ScriptMind())
    b, _, _ = body_with_company(ScriptMind(Voice(says=None)))
    for sim in (a, b):
        say(sim, me, to, "Hello there")
        sim.step()
    assert [e.data for e in a.events if e.kind == "reply"] == [e.data for e in b.events if e.kind == "reply"]


class FakeClient:
    def __init__(self, content):
        self.model, self.temperature, self.content, self.bodies = "m", 0.7, content, []

    def post(self, path, body):
        self.bodies.append(body)
        return {"choices": [{"message": {"content": self.content}}]}


@pytest.mark.parametrize("content,want", [
    ('{"says": "Morning! I\'m off to work now."}', "Morning! I'm off to work now."),
    ('{"says": "' + "x" * 300 + '"}', "x" * 140),
    ('{"says": "Have some bread, friend."}', None),       # invented: the village has only "food"
    ('{"says": "As an AI model I cannot."}', None),        # out of fiction
    ('not json', None),
    ('{"says": 3}', None),
])
def test_the_narrators_reply_is_clipped_and_checked(content, want):
    n = narrate.LLMNarrator.__new__(narrate.LLMNarrator)
    n.client, n.model = FakeClient(content), "m"
    got = n.reply("Hi", "go back to work", {"place": "field", "hour": 8, "role": "farmer"})
    assert got == want
    p = n.client.bodies[0]["messages"][0]["content"]
    assert '"Hi"' in p and "go back to work" in p and "Good morning" in p and "no food or coin" in p


def test_the_hold_waits_out_a_drawn_delay_from_the_click_and_logs_it():
    logged = []
    runner = SimpleNamespace(sim=SimpleNamespace(state={"player": {"agent": 3}}), say_wall={7: time.time()},
                             _side=lambda kind, **d: logged.append((kind, d)))
    ev = SimpleNamespace(kind="reply", actor=7, id=11, tick=42, data={"with": 3, "says": "hi"})
    other = SimpleNamespace(kind="reply", actor=8, id=12, tick=42, data={"with": 5, "says": "hi"})
    h = ReplyHold([0.3], "17-42")
    t0 = time.monotonic()
    asyncio.run(h.hold(runner, [other, ev]))
    assert 0.25 <= time.monotonic() - t0 < 1.0
    assert logged == [("held", {"reply": 11, "listener": 7, "drawn_s": 0.3, "wait_s": logged[0][1]["wait_s"]}),
                      ("sent", {"reply": 11, "tick": 42})]  # review of #81: the wall the held reply went out
    assert runner.say_wall == {}
    # Seeded: the same seed draws the same delays.
    d = [0.5, 1.0, 2.0, 4.0, 8.0]
    assert [ReplyHold(d, "s").rng.choice(d) for _ in range(1)] == [ReplyHold(d, "s").rng.choice(d)]
    with pytest.raises(ValueError):
        ReplyHold([], "s")
