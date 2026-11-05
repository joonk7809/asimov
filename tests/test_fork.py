"""Fork mind (T12): a twin serves the base log's outputs wherever its prompts match, asks live elsewhere.

The claim a twin rests on: for a mind that sees only its prompt, a fork twin is the world a full
rerun with the same change would make, so every difference from the base is inside the change's
causal cone and nothing outside it moves. PromptMind below is such a mind (deterministic, like an
LLM at temperature 0), so a full rerun with it is ground truth for its fork twin.
"""
import hashlib
import json
import re

import pytest

import asiimov.sim as sim_mod
from asiimov.fork import ForkLeak, ForkMind, divergence, live_calls, live_seed, run_twin, summary
from asiimov.minds import MindReply, NoisyMind, StubMind
from asiimov.replay import replay
from asiimov.scorecard import run_arm
from asiimov.sim import TICKS_PER_DAY, Sim


class PromptMind:
    """Answers from the prompt text alone: ids it lists, whether it is at the market, and a hash.
    Without a seed it is a function of the prompt; with one, the seed varies the answer."""
    model = "prompt-only-v0"

    def decide(self, req):
        h = int(req.prompt_hash, 16)
        if req.seed is not None:  # sampled like LLMMind: a request's seed picks among the answers
            h = int(hashlib.sha256(f"{req.prompt_hash}:{req.seed}".encode()).hexdigest(), 16)
        if req.kind == "plan":
            start = 7 + h % 3
            mid = start + 3 + (h >> 4) % 5
            out = {"plan": [{"activity": "work", "start_h": start, "end_h": mid},
                            {"activity": ("socialize", "explore", "home")[(h >> 8) % 3], "start_h": mid,
                             "end_h": min(23, mid + 4)}]}
        elif req.kind == "deliberate":
            here = re.search(r"Here: ([^.]*)\.", req.prompt).group(1)
            present = [int(x) for x in re.findall(r"#(\d+) \(", here)]
            kin = [int(x) for x in re.findall(r"\(#(\d+)\)", req.prompt.split("Your family:")[1])]
            if "The market has" in req.prompt:
                out = {"action": "buy_food", "target": None, "reason": "here"}
            else:
                pick = (h >> 12) % 4
                if pick == 0 and present:
                    out = {"action": "steal", "target": present[h % len(present)], "reason": "hungry"}
                elif pick == 1 and kin:
                    out = {"action": "ask_kin", "target": kin[h % len(kin)], "reason": "family"}
                elif pick == 2:
                    out = {"action": "wait", "target": None, "reason": "tired"}
                else:
                    out = {"action": "go_market", "target": None, "reason": "food"}
        else:
            out = {"core": [], "summary": "a day"}
        return MindReply(json.dumps(out, sort_keys=True), self.model, {"temperature": 0})


class Recording:
    """Wraps a mind and keeps every request it is asked."""

    def __init__(self, mind):
        self.mind, self.model, self.asked = mind, mind.model, []

    def decide(self, req):
        self.asked.append(req)
        return self.mind.decide(req)


def world(mind, seed=5, agents=40, days=4, player=None):
    s = Sim(seed, agents, mind)
    s.run(days, player or {})
    s.finish()
    return s


def fire(family, day=1):
    return {day * TICKS_PER_DAY + 60: [{"kind": "fire", "family": family}]}


def logs(events):
    return [e.to_json() for e in events]


@pytest.mark.parametrize("farm_yield,seed,family", [(1, 5, 2), (1, 6, 9), (3, 5, 2), (3, 2, 4)])
def test_a_fork_twin_is_the_full_rerun_and_moves_nothing_outside_the_cone(monkeypatch, farm_yield, seed, family):
    # Yield 1 is the engine 1-4 famine, where a fire spreads through theft and kin; (1, 6, 9) is a
    # world where a stub served from the record misses a theft (see the stub test below).
    monkeypatch.setattr(sim_mod, "FARM_YIELD", farm_yield)
    base = world(PromptMind(), seed)
    truth = world(PromptMind(), seed, player=fire(family))
    twin, mind = run_twin(base.events, PromptMind(), player=fire(family))
    assert logs(twin.events) == logs(truth.events)
    cone = divergence(base.events, truth.events)
    assert cone, "the fire changed nothing, so this world tests nothing"
    assert divergence(base.events, twin.events) == cone
    assert min(cone.values()) >= TICKS_PER_DAY + 60
    # Only the cone was paid for: a large share of the calls came from the record.
    assert mind.counts["base"] > 0 and mind.counts["miss"] < sum(mind.counts.values())
    assert sum(mind.counts.values()) == sum(1 for e in twin.events if e.origin == "mind")
    assert len(live_calls(base.events, twin.events)) == mind.counts["miss"]


def test_a_null_twin_of_a_sampled_world_is_the_base_log_and_asks_nothing():
    base = world(NoisyMind())
    live = Recording(NoisyMind())
    twin, mind = run_twin(base.events, live)
    assert logs(twin.events) == logs(base.events)
    assert live.asked == [] and mind.counts["miss"] == 0
    # Without the fork, the same sampled mind moves the whole village: that is what the fork removes.
    assert len(divergence(base.events, world(NoisyMind()).events)) > 20


def test_a_forced_live_agent_is_the_only_source_of_difference():
    base = world(PromptMind())
    agent, start = 7, TICKS_PER_DAY
    live = Recording(PromptMind())
    twin, mind = run_twin(base.events, live, live_agents=[agent], live_from=start, salt=1)
    b, t = logs(base.events), logs(twin.events)
    first = next(i for i, (x, y) in enumerate(zip(b, t)) if x != y)
    e = twin.events[first]
    assert e.origin == "mind" and e.actor == agent and e.tick >= start  # the first difference is its answer
    # Anyone else is asked only once its prompt changed, which can only follow that answer.
    others = [r for r in live.asked if r.agent_id != agent]
    assert others and min(r.tick for r in others) > e.tick and mind.counts["miss"] == len(others)
    assert all(r.tick >= start for r in live.asked)
    forced = [c for c in live_calls(base.events, twin.events, [agent], start) if c[2] == "forced"]
    assert len(forced) == mind.counts["forced"] == len(live.asked) - len(others)


def test_forcing_a_deterministic_mind_live_changes_nothing():
    base = world(StubMind())
    twin, mind = run_twin(base.events, StubMind(), live_agents=[3, 11], salt=2)
    assert logs(twin.events) == logs(base.events) and mind.counts["forced"] > 0


def test_forced_calls_get_a_salted_seed_and_misses_are_asked_as_a_fresh_run_would():
    base = world(PromptMind(), days=2)
    seeds = {}
    for salt in (1, 2):
        live = Recording(PromptMind())
        run_twin(base.events, live, live_agents=[4], salt=salt)
        forced = [r for r in live.asked if r.agent_id == 4]
        assert forced and all(r.seed == live_seed(salt, r) for r in forced)
        # LLMMind's default seed for an unchanged prompt would repeat the base run's own sample.
        assert all(r.seed != int(r.prompt_hash[:8], 16) for r in forced)
        assert all(r.seed is None for r in live.asked if r.agent_id != 4)
        seeds[salt] = [r.seed for r in forced]
    assert all(x != y for x, y in zip(seeds[1], seeds[2]))


def test_a_twin_replays_exactly_without_a_model():
    base = world(NoisyMind())
    twin, _ = run_twin(base.events, NoisyMind(), player=fire(2), live_agents=[5], salt=3)
    assert replay(twin.events).state_hash() == twin.state_hash()


def test_concurrent_calls_give_the_serial_twin():
    base = world(PromptMind(), days=3)
    one, _ = run_twin(base.events, PromptMind(), player=fire(3))
    many, _ = run_twin(base.events, PromptMind(), player=fire(3), workers=8)
    assert logs(one.events) == logs(many.events)


def test_a_crashed_twin_resumes_asking_only_from_the_tick_it_died(tmp_path):
    base = world(PromptMind())
    die_at = 2 * TICKS_PER_DAY + 30

    class Dying(PromptMind):
        def decide(self, req):
            if req.tick >= die_at:
                raise RuntimeError("LLM server unreachable")
            return super().decide(req)

    with pytest.raises(RuntimeError):
        run_twin(base.events, Dying(), str(tmp_path), player=fire(2), live_agents=[6], salt=1)
    from asiimov.fork import read_log
    prior = read_log(str(tmp_path / "events.partial.jsonl"))
    live = Recording(PromptMind())
    resumed, mind = run_twin(base.events, live, player=fire(2), live_agents=[6], salt=1, prior=prior)
    whole, _ = run_twin(base.events, PromptMind(), player=fire(2), live_agents=[6], salt=1)
    assert logs(resumed.events) == logs(whole.events)
    assert live.asked and min(r.tick for r in live.asked) >= die_at
    assert mind.counts["prior"] > 0


def test_a_twin_of_another_arm_stops_before_it_asks_anything():
    proto = {"seed": 3, "agents": 20, "days": 2}
    base = run_arm("fiction", NoisyMind(), proto)
    live = Recording(NoisyMind())
    with pytest.raises(ForkLeak):
        run_twin(base, live, player=fire(1))  # built as the plain world: every prompt differs
    assert live.asked == []
    twin, mind = run_twin(base, live, arm="fiction")
    assert logs(twin.events)[:-1] == logs(base) and live.asked == []


def test_an_edited_genesis_is_a_change_from_tick_zero():
    base = world(PromptMind(), days=2)
    pop = json.loads(json.dumps(base.events[0].data["population"]))
    a, b = pop["agents"][2], pop["agents"][9]
    a["traits"], b["traits"] = b["traits"], a["traits"]
    twin, mind = run_twin(base.events, PromptMind(), genesis=pop)
    assert mind.counts["miss"] > 0
    assert {2, 9} <= set(divergence(base.events, twin.events))


def test_two_records_for_one_agent_tick_are_refused():
    base = world(PromptMind(), days=1)
    ev = next(e for e in base.events if e.origin == "mind")
    with pytest.raises(Exception, match="two mind events"):
        ForkMind(base.events + [ev], PromptMind())


def test_stub_twins_are_full_reruns(monkeypatch):
    # StubMind reads who carries food from its view, which the prompt doesn't show. Served from the
    # record, its twin keeps an old answer where a full rerun changes it (seed 6, family 11, yield 1 at
    # engine 8; 3 at engine 7, 9 at engine 6), so stub controls run with serve=False.
    monkeypatch.setattr(sim_mod, "FARM_YIELD", 1)
    base = Sim(6, 50, StubMind()); base.run(4); base.finish()
    truth = Sim(6, 50, StubMind()); truth.run(4, fire(11)); truth.finish()
    served, _ = run_twin(base.events, StubMind(), player=fire(11))
    assert logs(served.events) != logs(truth.events)
    rerun, mind = run_twin(base.events, StubMind(), player=fire(11), serve=False)
    assert logs(rerun.events) == logs(truth.events) and mind.counts["base"] == 0
    s = summary(base.events, rerun.events)
    assert s["agents_diverged"] == len(divergence(base.events, truth.events))
    with pytest.raises(ForkLeak, match="own mind"):
        run_twin(world(NoisyMind(), days=1).events, StubMind(), serve=False)
