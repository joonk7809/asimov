import json

import pytest

from asiimov import authorship as au
from asiimov import scorecard as sc
from asiimov.minds import MindReply, NoisyMind, ReplayDivergence, StubMind
from asiimov.sim import Sim

PROTO = {"seed": 1, "agents": 30, "days": 2, "fire": "3@1"}


def world(mind, days=2, agents=30):
    sim = Sim(seed=1, n_agents=agents, mind=mind)
    sim.run(days, {200: [{"kind": "fire", "family": 2}]})
    sim.finish()
    return sim


class AllDayMind(StubMind):
    """Plans one activity for the whole day; deliberates like the stub."""

    def __init__(self, activity):
        self.activity = activity

    def decide(self, req):
        if req.kind != "plan":
            return super().decide(req)
        out = {"plan": [{"activity": self.activity, "start_h": 0, "end_h": 24}]}
        return MindReply(json.dumps(out), self.model, {})


def test_every_waking_tick_gets_one_author_and_the_log_replays_exactly():
    sim = world(NoisyMind())
    au_sim = au.replay_authorship(sim.events)
    assert au_sim.state_hash() == sim.state_hash()
    # Independent count: the world's own per-tick tally of awake agents.
    awake = sum(n for c in sim.metrics["activity"].values() for n in c.values())
    assert len(au_sim.records) == awake
    assert len({(t, a) for t, a, *_ in au_sim.records}) == awake
    assert all(r[2] in au.AUTHORS for r in au_sim.records)
    # A deliberated override is go_market, which the habit layer never picks, or D2's going on at a work
    # block's end, which only an agent that answered "work" there carries.
    overrides = [(a, act) for _, a, author, act, _ in au_sim.records if author == "deliberate"]
    assert overrides and {act for _, act in overrides} <= {"shop", "work"}
    went_on = {e.actor for e in sim.events if e.kind == "receipt" and e.data["why"] == "went on working"}
    assert {a for a, act in overrides if act == "work"} <= went_on


@pytest.mark.parametrize("arm", sc.ARMS + sc.RESPONSE_ARMS)
def test_every_scorecard_arm_log_replays(arm):
    events = sc.run_arm(arm, StubMind(), PROTO)
    assert au.replay_authorship(events, arm).records


def test_a_tampered_log_is_refused():
    events = world(StubMind(), days=1).events
    i = next(i for i, e in enumerate(events) if e.kind == "mind_plan")
    events[i].mind = {**events[i].mind, "raw_output": json.dumps({"plan": []})}
    with pytest.raises(ReplayDivergence):
        au.replay_authorship(events)


def test_label_only_differences_are_divergent_but_never_effective():
    """Planted truth: "rest" and "home" differ only in name, so whole-day rest and whole-day home plans
    make the same world and the same habit picks. Divergence tells them apart; effective share can't."""
    rep = {act: au.summarize(au.replay_authorship(world(AllDayMind(act)).events)) for act in ("home", "rest")}
    assert rep["home"]["effective_mind_share"] == rep["rest"]["effective_mind_share"]
    assert rep["rest"]["divergent_mind_share"]["mean"] > rep["home"]["divergent_mind_share"]["mean"]
    assert rep["home"]["author_share"]["habit"] == 0  # a 0-24 plan covers every non-override tick


def test_no_plan_means_the_code_authors_the_day():
    rep = au.summarize(au.replay_authorship(world(AllDayMind("none-such")).events))  # every block dropped
    assert rep["author_share"]["plan"] == 0
    assert rep["divergence_rate"]["plan"] is None
    assert rep["test7"]["median"] == 0 and not rep["test7"]["pass"]
