import json

import pytest

from asiimov.events import ORIGINS, read_jsonl, write_jsonl
from asiimov.memory import SECTION_BUDGETS, assemble_prompt, render_files
from asiimov.metrics import icc1, summarize
from asiimov.minds import NoisyMind, ReplayDivergence, StubMind
from asiimov.replay import replay
from asiimov.rng import Rng
from asiimov.sim import TICKS_PER_DAY, Sim

FIRE = {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 2}]}


def run(mind, agents=30, days=2, seed=7, player=None):
    sim = Sim(seed, agents, mind)
    sim.run(days, player or {})
    return sim


def test_rng_is_a_pure_function_of_seed_stream_index():
    a, b = Rng(3), Rng(3)
    assert [a.draw("x", 0, 99) for _ in range(5)] == [b.draw("x", 0, 99) for _ in range(5)]
    assert Rng(3).raw("x", 4) == Rng(3).raw("x", 4) != Rng(4).raw("x", 4)


def test_same_seed_same_world():
    s1, s2 = run(StubMind(), player=FIRE), run(StubMind(), player=FIRE)
    assert [e.to_json() for e in s1.events] == [e.to_json() for e in s2.events]
    assert s1.state_hash() == s2.state_hash()


def test_noisy_minds_diverge_but_each_log_replays_exactly(tmp_path):
    s1, s2 = run(NoisyMind(), player=FIRE), run(NoisyMind(), player=FIRE)
    assert s1.state_hash() != s2.state_hash()  # sampled minds are not reproducible by recomputation
    for s in (s1, s2):
        path = tmp_path / f"{id(s)}.jsonl"
        write_jsonl(s.events, str(path))
        r = replay(read_jsonl(str(path)))  # no model is called; recorded outputs are served
        assert r.state_hash() == s.state_hash()
        assert len(r.events) == len(s.events)


def test_replay_detects_a_tampered_mind_output():
    s = run(NoisyMind())
    events = [type(e)(**json.loads(e.to_json())) for e in s.events]
    target = next(e for e in events if e.kind == "mind_plan")
    target.mind["raw_output"] = json.dumps({"plan": [{"activity": "rest", "start_h": 8, "end_h": 20}]})
    with pytest.raises(ReplayDivergence):
        replay(events)


def test_every_event_has_provenance():
    s = run(StubMind(), player=FIRE)
    seen = set()
    for e in s.events:
        assert e.origin in ORIGINS
        assert all(p in seen for p in e.parents), f"event {e.id} cites a later or missing parent"
        if e.origin == "mind":
            assert {"model", "prompt_hash", "prompt_tokens", "sampling", "raw_output"} <= set(e.mind)
        if e.kind == "receipt":
            assert s.events[e.parents[0]].origin == "mind"
        seen.add(e.id)
    deliberations = sum(1 for e in s.events if e.kind == "mind_deliberate")
    assert deliberations == sum(1 for e in s.events if e.kind == "receipt")


def test_individual_rhythms_are_stable_within_and_different_between_agents():
    s = run(StubMind(), agents=100, days=5, seed=1)
    m = summarize(s.events, s.metrics["activity"], 100)
    assert m["variety"]["wake_icc"] > 0.8
    lo, hi = m["variety"]["wake_range_min"]
    assert hi - lo >= 180


def test_icc_separates_character_from_noise():
    assert icc1({1: [1, 1, 1], 2: [5, 5, 5], 3: [9, 9, 9]}) > 0.99
    assert icc1({1: [1, 9, 5], 2: [9, 1, 5], 3: [5, 9, 1]}) < 0.1


def test_player_intervention_propagates_against_a_twin():
    base = run(StubMind(), agents=60, days=4, seed=5)
    fire = run(StubMind(), agents=60, days=4, seed=5, player={TICKS_PER_DAY + 60: [{"kind": "fire", "family": 2}]})

    def seq(sim):
        out = {}
        for e in sim.events:
            if e.kind == "activity" and e.tick >= 3 * TICKS_PER_DAY:
                out.setdefault(e.actor, []).append((e.tick, e.data["activity"]))
        return out

    a, b = seq(base), seq(fire)
    changed = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
    family = set(fire.state["families"][2]["members"])
    assert set(changed) - family, "the fire changed nobody outside the burned family"
    fire_id = next(e.id for e in fire.events if e.kind == "fire")
    assert any(fire_id in e.parents for e in fire.events), "nothing cites the fire as a cause"


def test_memory_files_and_prompt_budgets():
    s = run(StubMind(), agents=40, days=2)
    souls = {a["soul"] for a in s.state["agents"]}
    assert len(souls) == 40
    for a in s.state["agents"]:
        files = render_files(a, s.state)
        assert set(files) == {"soul.md", "core.md", "goals.md", "relationships.md", "beliefs.md",
                              "recent.md", "episodes.md"}
        _, counts = assemble_prompt(a, s.state, "deliberate", "now")
        for name, n in counts.items():
            assert n <= SECTION_BUDGETS[name] + 1
