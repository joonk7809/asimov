"""T21 (engine 13): a reflection cut off at max_tokens is asked once more with a keyed seed and frequency
penalty 1.0, recorded like any mind call, and replayed exactly (plan/t21-reflect-fix-spec.md)."""
import json

from asiimov.minds import MindReply, StubMind
from asiimov.replay import replay
from asiimov.rng import Rng
from asiimov.sim import TICKS_PER_DAY, Sim

LOOP = '{"summary": "' + "ngày ngày " * 40  # a reply cut off inside a loop, as 4B fiction's were


class CutOnce(StubMind):
    """Cuts off the first reflection of each agent in `who`; the retry and everything else answer as the stub."""

    def __init__(self, who, fail_retry=False):
        self.who, self.fail_retry, self.asked = set(who), fail_retry, []

    def decide(self, req):
        if req.kind == "reflect" and req.agent_id in self.who:
            self.asked.append((req.agent_id, req.tick, req.attempt, req.seed))
            assert req.frequency_penalty_milli == (1000 if req.attempt else 0)
            if req.attempt == 0 or self.fail_retry:
                return MindReply(LOOP, self.model, {"seed": req.seed}, "length")
        return super().decide(req)


def world(mind):
    sim = Sim(3, 12, mind)
    sim.run(2)
    sim.finish()
    return sim


def test_a_cut_off_reflection_is_retried_once_recorded_and_replayed():
    mind = CutOnce({4})
    sim = world(mind)
    first = [a for a in mind.asked if a[2] == 0]
    retries = [a for a in mind.asked if a[2] == 1]
    assert first and len(retries) == len(first)  # once per cut-off, never twice
    for (aid, tick, _, seed), _ in zip(retries, first):
        assert seed == Rng(3).raw(f"reflect-retry:{aid}:{tick // TICKS_PER_DAY}", 0) % 2 ** 32
    ev = [e for e in sim.events if e.kind == "mind_reflect" and e.actor == 4]
    cut = [e for e in ev if e.data.get("retried")]
    again = [e for e in ev if e.data.get("attempt") == 1]
    assert len(cut) == len(again) == len(first)
    for c, r in zip(cut, again):
        assert c.mind["finish_reason"] == "length" and r.mind["finish_reason"] == "stop"
        assert r.parents == [c.id] and r.tick == c.tick
        sleep = next(e for e in sim.events if e.kind == "sleep" and e.actor == 4 and e.tick == r.tick)
        assert sleep.parents == [r.id]  # the day's memory comes from the retry
    assert replay(sim.events).state_hash() == sim.state_hash()
    # only agent 4's reflections changed: everyone else's log is the stub's
    plain = world(StubMind())
    key = lambda s: [(e.tick, e.kind, e.actor, json.dumps(e.data, sort_keys=True)) for e in s.events  # noqa: E731
                     if e.actor != 4 and e.kind not in ("checkpoint", "end")]
    assert key(sim) == key(plain)


def test_a_retry_that_fails_too_keeps_the_no_memory_fallback():
    mind = CutOnce({4}, fail_retry=True)
    sim = world(mind)
    again = [e for e in sim.events if e.kind == "mind_reflect" and e.data.get("attempt") == 1]
    assert again and all(e.mind["finish_reason"] == "length" for e in again)
    assert sum(1 for a in mind.asked if a[2] == 1) == len(again)  # still only one retry each
    for r in again:
        sleep = next(e for e in sim.events if e.kind == "sleep" and e.actor == 4 and e.tick == r.tick)
        assert sleep.data["promoted"] == 0
    assert replay(sim.events).state_hash() == sim.state_hash()


def test_plans_and_deliberations_are_never_retried():
    class CutAll(StubMind):
        def decide(self, req):
            r = super().decide(req)
            return MindReply(r.raw_output, r.model, r.sampling, "length") if req.kind != "reflect" else r
    sim = world(CutAll())
    assert not any(e.data.get("attempt") or e.data.get("retried") for e in sim.events if e.origin == "mind")


def test_an_unparsed_reflection_that_was_not_cut_off_is_not_retried():
    class Garbage(StubMind):
        def decide(self, req):
            return MindReply("not json", self.model, {}, "stop") if req.kind == "reflect" else super().decide(req)
    sim = world(Garbage())
    assert not any(e.data.get("attempt") or e.data.get("retried") for e in sim.events if e.origin == "mind")


def test_the_llm_body_carries_the_penalty_only_on_the_retry():
    from asiimov.llm import LLMMind
    from asiimov.minds import MindRequest
    m = LLMMind("http://x", "m")
    req = MindRequest("reflect", 1, 5, "p", "ab12cd34", {})
    assert "frequency_penalty" not in m.body(req)
    assert m.body(MindRequest("reflect", 1, 5, "p", "ab12cd34", {}, seed=7, attempt=1,
                              frequency_penalty_milli=1000))["frequency_penalty"] == 1.0
