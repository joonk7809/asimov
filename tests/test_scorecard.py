import json
import math
import random

import pytest

from asiimov import scorecard as sc
from asiimov.metrics import summarize
from asiimov.minds import StubMind
from asiimov.sim import TICKS_PER_DAY

TH, SHA = sc.load_thresholds()
PROTO = dict(TH["protocol"], agents=30, days=4)


@pytest.fixture(scope="module")
def arms():
    return {arm: sc.run_arm(arm, StubMind(), PROTO) for arm in sc.ARMS}


def test_log_extraction_matches_live_sim_counters():
    pop = sc.labeled_population(PROTO["seed"], PROTO["agents"])
    sim = sc.ArmSim(PROTO["seed"], PROTO["agents"], StubMind(), pop, "full", {})
    sim.run(3)
    ex = sc.extract(sim.events)
    live = {k: dict(v) for k, v in sim.metrics["activity"].items()}
    assert {k: dict(v) for k, v in ex["hourly"].items()} == live
    s = summarize(sim.events, sim.metrics["activity"], PROTO["agents"])
    assert s["events_by_kind"]["wake"] == sum(len(p) for p in ex["periods"].values()) + sum(
        1 for a in sim.state["agents"] if not a["asleep"])


def test_arms_ablate_only_the_prompt():
    seen = {}

    class Spy(StubMind):
        def decide(self, req):
            seen.setdefault(self.arm, {})[req.agent_id] = req.prompt
            return super().decide(req)
    pop = sc.labeled_population(1, 12)
    souls = {a["id"]: sc.render_soul(a) for a in pop["agents"]}
    hashes = {}
    for arm in sc.ARMS:
        spy = Spy()
        spy.arm = arm
        genesis = sc.flatten_params(pop) if arm == "flat_params" else pop
        sim = sc.ArmSim(1, 12, spy, genesis, arm, souls)
        sim.run(1)
        hashes[arm] = sim.state_hash()
    for aid, soul in souls.items():
        assert soul in seen["full"][aid] and soul in seen["flat_params"][aid]
        assert soul not in seen["no_text"][aid]
    # the stub reads params, not text: blanking the soul changes nothing, flattening params does
    assert hashes["full"] == hashes["no_text"] != hashes["flat_params"]
    flat = sc.flatten_params(pop)["agents"]
    assert len({json.dumps(a["traits"]) for a in flat}) == 1
    assert [a["role"] for a in flat] == [a["role"] for a in pop["agents"]]


def test_a_crashed_arm_keeps_its_recorded_events(tmp_path):
    class DiesOnDay2(StubMind):
        def decide(self, req):
            if req.tick >= 2 * TICKS_PER_DAY:
                raise RuntimeError("pod went away")
            return super().decide(req)
    with pytest.raises(RuntimeError):
        sc.run_arm("full", DiesOnDay2(), PROTO, str(tmp_path))
    saved = sc.read_jsonl(str(tmp_path / "events.partial.jsonl"))
    assert saved[0].kind == "genesis" and saved[-1].tick >= 2 * TICKS_PER_DAY - 1


def test_name_labels_are_balanced_and_independent_of_params():
    pop = sc.labeled_population(1, 100)
    labels = [sc.FIRST_NAME_LABEL[a["name"].split(" ")[0]] for a in pop["agents"]]
    assert 30 < labels.count("F") < 70
    plain = sc.generate_population(1, 100)
    for a, b in zip(pop["agents"], plain["agents"]):
        assert {k: v for k, v in a.items() if k != "name"} == {k: v for k, v in b.items() if k != "name"}
        assert a["name"].split(" ", 1)[1] == b["name"].split(" ", 1)[1]


def test_icc_separates_noise_from_character():
    rng = random.Random(0)
    noise = {a: [rng.gauss(0, 1) for _ in range(6)] for a in range(50)}
    char = {a: [a + rng.gauss(0, 0.1) for _ in range(6)] for a in range(50)}
    assert sc.icc1(noise, 3) < 0.15 < 0.95 < sc.icc1(char, 3)
    assert math.isnan(sc.icc1({a: [1.0] * 4 for a in range(5)}, 3))  # t1 grades this as 0


def test_spearman_handles_ties_and_constants():
    assert sc.spearman([1, 2, 2, 3], [10, 20, 20, 30]) == pytest.approx(1.0)
    assert sc.spearman([1, 2, 3], [5, 5, 5]) == 0.0


def _synthetic(n=60, days=6, effect=0.0, text=lambda a, d: "same words every day"):
    agents = [{"id": i, "name": f"{'Anna' if i % 2 else 'John'} X", "role": "farmer",
               "traits": {t: 500 for t in sc.TRAITS}, "rhythm": {"chronotype_min": 0}} for i in range(n)]
    rng = random.Random(1)
    periods = {}
    for a in range(n):
        base = rng.gauss(4, 1) + (effect if a % 2 else 0)
        periods[a] = [{"day": d, "start": d * TICKS_PER_DAY, "wake_min": 390, "work_h": base + rng.gauss(0, 1),
                       "social_h": rng.gauss(2, 1), "explore_h": 0.0, "places": 2, "partners": 3,
                       "talks_started": 1, "deliberations": 0, "steals": 0, "text": text(a, d)} for d in range(days)]
    return {"pop": {"agents": agents}, "periods": periods, "hourly": {}, "switches": {}}


def test_bias_test_catches_a_planted_gap_and_passes_without_one():
    th = TH["t6_bias"]
    assert sc.t6_bias(_synthetic(effect=2.0), th, PROTO)["biased_features"] == ["work_h"]
    assert sc.t6_bias(_synthetic(effect=0.0), th, PROTO)["status"] == "pass"


def test_homogenization_needs_distinct_voices_and_no_drift():
    th = TH["t5_homogenization"]
    words = ["bread", "river", "debt", "storm", "kin", "market", "field", "fire", "trust", "coin"]
    generic = _synthetic()
    voiced = _synthetic(text=lambda a, d: f"{words[a % 10]} {words[(a // 10) % 10]} worries me today")
    assert sc.t5_homogenization({"full": generic, "no_text": generic}, th, PROTO)["status"] == "fail"
    assert sc.t5_homogenization({"full": voiced, "no_text": generic}, th, PROTO)["status"] == "pass"
    drifting = _synthetic(text=lambda a, d: "same words" if d >= 4 else f"{words[a % 10]} {words[a // 10 % 10]}")
    r = sc.t5_homogenization({"full": drifting, "no_text": generic}, th, PROTO)
    assert r["drift"] > th["max_drift"] and r["status"] == "fail"


def test_lockstep_flags_a_crowd():
    th = TH["t4_no_lockstep"]
    crowd = {(0, h): {"work": 600} for h in range(8, 21)}
    mixed = {(0, h): {"work": 240, "home": 120, "socialize": 120, "rest": 60, "explore": 60} for h in range(8, 21)}
    assert sc.t4_lockstep({"hourly": crowd, "switches": {}}, th)["status"] == "fail"
    assert sc.t4_lockstep({"hourly": mixed, "switches": {}}, th)["status"] == "pass"


def test_end_to_end_report_on_the_stand_in(arms, tmp_path):
    ex = {k: sc.extract(v) for k, v in arms.items()}
    report = sc.grade(ex, dict(TH, protocol=PROTO), SHA)
    t = report["tests"]
    # the stub ignores prompt text: the scorecard must say the soul isn't earning its tokens
    assert t["3_soul_ablation"]["text_contribution"] == 0.0
    assert t["3_soul_ablation"]["status"] == "fail"
    assert t["5_no_homogenization"]["ratio"] == 1.0
    assert t["2_trait_predicts_behavior"]["traits"]["chronotype"] == "pass"
    assert report["overall"] == "fail"
    assert "Variety scorecard" in sc.render_md(report)


def test_concurrent_mind_calls_leave_every_arm_log_unchanged():
    proto = dict(PROTO, days=2)
    for arm in ("no_text", "flat_params"):
        serial = [e.to_json() for e in sc.run_arm(arm, StubMind(), proto)]
        assert [e.to_json() for e in sc.run_arm(arm, StubMind(), proto, workers=8)] == serial
