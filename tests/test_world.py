import hashlib
import json

import pytest

from asiimov.events import ORIGINS, read_jsonl, write_jsonl
from asiimov.memory import SECTION_BUDGETS, assemble_prompt, render_files
from asiimov.metrics import icc1, summarize
from asiimov.minds import NoisyMind, ReplayDivergence, StubMind
from asiimov.replay import replay
from asiimov.rng import Rng
import asiimov.sim as sim_mod
from asiimov.sim import ENGINE_VERSION, TICKS_PER_DAY, Sim

FIRE = {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 2}]}


def run(mind, agents=30, days=2, seed=7, player=None):
    sim = Sim(seed, agents, mind)
    sim.run(days, player or {})
    sim.finish()
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
        if e.kind == "receipt":  # a deliberation or plan, or an intent a deliberation set
            p = s.events[e.parents[0]]
            assert p.origin == "mind" or (p.kind == "intent_set" and s.events[p.parents[0]].origin == "mind")
        seen.add(e.id)
    deliberations = {e.id for e in s.events if e.kind == "mind_deliberate"}
    assert deliberations == {e.parents[0] for e in s.events if e.kind == "receipt"} & deliberations  # each has one


def test_individual_rhythms_are_stable_within_and_different_between_agents():
    s = run(StubMind(), agents=100, days=5, seed=1)
    m = summarize(s.events, s.metrics["activity"], 100)
    assert m["variety"]["wake_icc"] > 0.8
    lo, hi = m["variety"]["wake_range_min"]
    assert hi - lo >= 180


def test_icc_separates_character_from_noise():
    assert icc1({1: [1, 1, 1], 2: [5, 5, 5], 3: [9, 9, 9]}) > 0.99
    assert icc1({1: [1, 9, 5], 2: [9, 1, 5], 3: [5, 9, 1]}) < 0.1


def test_player_intervention_propagates_against_a_twin(monkeypatch):
    # Propagation runs through scarcity (hunger, theft, grudges). With engine 5's fed world the
    # stub's fire reaches nobody outside the family (seeds 1, 2, 5, every family), a true negative
    # (S4); the propagation machinery is tested in the engine-1..4 economy.
    monkeypatch.setattr(sim_mod, "FARM_YIELD", 1)
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


def test_aborted_run_keeps_its_recorded_events(tmp_path, monkeypatch):
    """A paid run that dies mid-way (here: the mind fails on day 2) must not lose the log, and
    --resume must finish it with exactly the log an uninterrupted run writes."""
    import sys

    from asiimov import run as run_mod

    class Dying(StubMind):
        def decide(self, req):
            if req.tick >= 2 * TICKS_PER_DAY:
                raise RuntimeError("LLM server unreachable")
            return super().decide(req)

    args = ["run", "--agents", "10", "--days", "4", "--fire", "1@1"]
    monkeypatch.setattr(run_mod, "StubMind", Dying)
    monkeypatch.setattr(sys, "argv", args + ["--out", str(tmp_path / "a")])
    with pytest.raises(RuntimeError):
        run_mod.main()
    events = read_jsonl(str(tmp_path / "a" / "events.jsonl"))
    assert events[0].kind == "genesis" and events[-1].tick == 2 * TICKS_PER_DAY - 1

    monkeypatch.setattr(run_mod, "StubMind", StubMind)
    monkeypatch.setattr(sys, "argv", ["run", "--resume", "--days", "4", "--out", str(tmp_path / "a")])
    run_mod.main()
    monkeypatch.setattr(sys, "argv", args + ["--out", str(tmp_path / "b")])
    run_mod.main()
    a, b = ((tmp_path / d / "events.jsonl").read_bytes() for d in "ab")
    assert a == b
    assert replay(read_jsonl(str(tmp_path / "a" / "events.jsonl"))).state_hash() == \
        json.loads((tmp_path / "b" / "metrics.json").read_text())["final_hash"]


# sha256 of a fixed world's log, per engine version. If this fails, the rules or the log format changed:
# bump ENGINE_VERSION (old logs then refuse to replay) and record the new hash here.
GOLDEN = {"1": "fa6d1e9898c97fe570590e1ad322d952375b2941c4828001b5ea933c6e4d369b",
          "2": "6dd170b56c91610ef4359f4850f451a87224fb9ed5cb9edf7ec552d0fe884f4d",  # 42adb9e, never merged
          "3": "7b0e876ed43e135033a9b9b6e819c176e70c2e4aeb3b3b14f9b2f8d3ad71088d",
          "4": "87be8cccf05d032f2dc428e615b7f45ad3981da0c735ff9edb2ce74cb48d3c32",
          "5": "49abfd07071dbe24572a47b4ce6716dbbee71d069b82cd3f6353679e04a20350",
          "6": "c7dfb5029e3ea1dbff76b179761d367530fc7e3cbd54d3508c04e76b1344ced8",
          "7": "0b3a32589883725f55439d837727557df5a288eebba3234aaa59f1f2629d1fb9",  # PR #19
          "8": "845a7abb49589da562cbd54dbc952d62a7d61606d00f13a390f4d3cc669b5560"}


def test_log_changes_bump_the_engine_version():
    s = run(StubMind(), agents=20, days=2, seed=3, player=FIRE)
    digest = hashlib.sha256("".join(e.to_json() + "\n" for e in s.events).encode()).hexdigest()
    assert GOLDEN.get(ENGINE_VERSION) == digest, f"engine {ENGINE_VERSION} log hash is now {digest}"
