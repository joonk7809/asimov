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


def test_famine_cuts_farm_food_from_its_start_day_only():
    from collections import Counter
    proto = dict(PROTO, days=4)
    food = {}
    for arm in ("full", "famine"):
        food[arm] = Counter()
        for e in sc.run_arm(arm, StubMind(), proto):
            if e.kind == "produce" and "food" in e.data:
                food[arm][e.tick // TICKS_PER_DAY] += e.data["food"]
    for day in range(sc.FAMINE_FROM_DAY):
        assert food["famine"][day] == food["full"][day]
    # Agents work a bit differently once hungry, so roughly a third. The top is 0.5 from engine 10: stub
    # farmers who buy from or are given food by a neighbour work longer (day 3: 55 of 117; engine 9, 48).
    for day in range(sc.FAMINE_FROM_DAY, 4):
        assert 0.2 < food["famine"][day] / food["full"][day] < 0.5


def test_fiction_arm_changes_only_the_framing_and_the_soul_holds_its_anchors_once():
    seen = {}

    class Spy(StubMind):
        def decide(self, req):
            seen.setdefault(self.arm, {}).setdefault((req.agent_id, req.kind), req)
            return super().decide(req)
    pop = sc.labeled_population(1, 30)
    souls = {a["id"]: sc.render_soul(a) for a in pop["agents"]}
    for arm in ("full", "fiction"):
        spy = Spy()
        spy.arm = arm
        sc.ArmSim(1, 30, spy, pop, arm, souls).run(1)
    key = next(iter(seen["full"]))
    full, fic = seen["full"][key].prompt, seen["fiction"][key].prompt
    assert full.startswith(sc.SCHEMA_HEAD) and fic.startswith(sc.FICTION_HEADER)
    block = sc.anchors(pop["agents"][key[0]])
    assert fic.count("# How you tend to act") == full.count("# How you tend to act") == 1  # T25: in every soul
    assert block in fic and block in full
    assert fic.replace(sc.FICTION_HEADER, sc.SCHEMA_HEAD) == full
    assert seen["fiction"][key].prompt_hash != seen["full"][key].prompt_hash


def test_fiction_arm_records_the_tokens_of_the_prompt_it_sent():
    from asiimov.memory import tokens
    sent = {}

    class Spy(StubMind):
        def decide(self, req):
            sent[(req.agent_id, req.tick)] = tokens(req.prompt)
            return super().decide(req)
    pop = sc.labeled_population(1, 30)
    sim = sc.ArmSim(1, 30, Spy(), pop, "fiction", {a["id"]: sc.render_soul(a) for a in pop["agents"]})
    sim.run(1)
    recorded = {(e.actor, e.tick): e.mind["prompt_tokens"] for e in sim.events if e.origin == "mind"}
    # Section clipping and joins make the per-section sum differ from tokens(prompt) by a few at most.
    assert all(abs(recorded[k] - sent[k]) <= 8 for k in recorded)


def test_fiction_parts_compose_and_all_three_equal_the_fiction_arm():
    assert sc.fiction_header(False, False) == sc.SCHEMA_HEAD
    assert sc.FICTION_HEADER.startswith("# How you act\nThis is a story about a village.")
    assert sc.parse_arm("fiction") == sc.parse_arm("full+framing+anchors+permission") == ("full", frozenset(sc.FICTION_PARTS))
    assert sc.parse_arm("no_text+fiction") == ("no_text", frozenset(sc.FICTION_PARTS))
    for bad in ("fiction+bogus", "nope", "no_text+no_text", "framing"):
        with pytest.raises(ValueError):
            sc.parse_arm(bad)
    proto = dict(PROTO, days=2)
    logs = {arm: [e.to_json() for e in sc.run_arm(arm, StubMind(), proto)]
            for arm in ("fiction", "full+framing+anchors+permission")}
    assert logs["fiction"] == logs["full+framing+anchors+permission"]


def _first_prompts(arm, n=30):
    seen = {}

    class Spy(StubMind):
        def decide(self, req):
            seen.setdefault(req.agent_id, req.prompt)
            return super().decide(req)
    pop = sc.labeled_population(1, n)
    base = sc.parse_arm(arm)[0]
    genesis = sc.flatten_params(pop) if base == "flat_params" else pop
    sim = sc.ArmSim(1, n, Spy(), genesis, arm, {a["id"]: sc.render_soul(a) for a in pop["agents"]})
    sim.run(1)
    return pop, seen


def test_each_fiction_part_adds_only_its_own_text():
    pop, full = _first_prompts("full")
    for part in sc.FICTION_PARTS:
        _, got = _first_prompts(f"full+{part}")
        for a in pop["agents"]:
            p, anchor = got[a["id"]], sc.anchors(a)
            assert (sc.FRAMING in p) == (part == "framing")
            assert (sc.PERMISSION in p) == (part == "permission")
            assert p.count(anchor) == 1  # "+anchors" adds nothing since T25: the soul has them
            back = p.replace(sc.fiction_header(part == "framing", part == "permission"), sc.SCHEMA_HEAD)
            assert back == full[a["id"]]


def test_minus_anchors_runs_the_old_unanchored_soul():
    """T25: 'full-anchors' is the soul as rendered before T25, for attribution in the same session."""
    from asiimov.memory import unanchored
    assert sc.parse_arm("full-anchors") == ("full", frozenset({"unanchored"}))
    assert sc.parse_arm("fiction-anchors") == ("full", frozenset({"framing", "permission", "unanchored"}))
    assert sc.configuration("full-anchors") == ("unanchored", "full")
    assert sc.configuration("no_text-anchors") == ("unanchored", "no_text")
    for bad in ("full-anchors+framing", "full-anchors-anchors", "bogus-anchors"):
        with pytest.raises(ValueError):
            sc.parse_arm(bad)
    pop, full = _first_prompts("full")
    _, old = _first_prompts("full-anchors")
    _, flat_old = _first_prompts("flat_params-anchors")
    for a in pop["agents"]:
        soul = sc.render_soul(a)
        assert unanchored(soul) + "\n\n" + sc.anchors(a) == soul
        assert old[a["id"]] == full[a["id"]].replace("\n\n" + sc.anchors(a), "", 1)
        assert "# How you tend to act" not in old[a["id"]] and "# How you tend to act" not in flat_old[a["id"]]
        assert unanchored(soul) in flat_old[a["id"]]


def test_fiction_composes_with_the_ablation_arms():
    pop, nt = _first_prompts("no_text+fiction")
    _, flat = _first_prompts("flat_params+fiction")
    for a in pop["agents"]:
        assert nt[a["id"]].startswith(sc.FICTION_HEADER) and flat[a["id"]].startswith(sc.FICTION_HEADER)
        assert sc.render_soul(a) not in nt[a["id"]] and "# How you tend to act" not in nt[a["id"]]
        # flat_params: code runs on median params, but the prompt keeps the real soul, real-param anchors included
        assert sc.render_soul(a) in flat[a["id"]] and flat[a["id"]].count(sc.anchors(a)) == 1
    assert len({sc.anchors(a) for a in pop["agents"]}) > 1


def test_grade_only_grades_each_configuration_from_its_own_arms(tmp_path, monkeypatch):
    from asiimov.events import write_jsonl
    names = ["full", "no_text", "fiction", "no_text+fiction", "flat_params+fiction", "full+framing", "no_text+anchors"]
    for arm in names:
        (tmp_path / arm).mkdir()
        write_jsonl(sc.run_arm(arm, StubMind(), PROTO), str(tmp_path / arm / "events.jsonl"))
    monkeypatch.setattr("sys.argv", ["scorecard", "--grade-only", "--out", str(tmp_path)])
    sc.main()
    plain = json.loads((tmp_path / "report.json").read_text())
    fic = json.loads((tmp_path / "report+fiction.json").read_text())
    framing = json.loads((tmp_path / "report+framing.json").read_text())
    assert (plain["configuration"], fic["configuration"], framing["configuration"]) == ("plain", "fiction", "framing")
    # fiction's test 3 and 5 compare fiction's own arms, not the plain ones
    assert "needs" not in json.dumps(fic["tests"]["3_soul_ablation"])
    assert "needs full and no_text" in framing["tests"]["3_soul_ablation"]["why"]
    assert not (tmp_path / "report+anchors.json").exists()  # no full arm for anchors: not graded
    assert sc.configuration("famine") == ("famine", "full") and sc.configuration("no_text+fiction") == ("fiction", "no_text")


def test_famine_spoils_the_market_once_at_its_start():
    """Engine 5 banks surplus in the market on days 0-1; the famine must not live off it."""
    ev = sc.run_arm("famine", StubMind(), dict(PROTO, days=3))
    blights = [e for e in ev if e.kind == "blight"]
    assert len(blights) == 1 and blights[0].tick == sc.FAMINE_FROM_DAY * TICKS_PER_DAY
    assert blights[0].data["market_food_lost"] > 0
    assert not any(e.kind == "blight" for e in sc.run_arm("full", StubMind(), dict(PROTO, days=3)))


def test_famine_caps_the_village_farm_output_per_day_and_stays_hungry():
    """T1k acceptance at $0: whatever hours farmers work, a famine day makes at most FAMINE_DAILY_FOOD,
    and the stub's 100-agent famine averages under 2.0 meals per agent-day, under 1.5 from day 2."""
    from collections import Counter
    from asiimov.diagnostics import food

    class Diligent(sc.ArmSim):  # farmers who worked every hour would still be held to the cap
        def _food_yield(self, a):
            return 7 * super()._food_yield(a)

    proto = dict(TH["protocol"], days=5)
    ev = sc.run_arm("famine", StubMind(), proto)
    made = Counter()
    for e in ev:
        if e.kind == "produce" and "food" in e.data:
            made[e.tick // TICKS_PER_DAY] += e.data["food"]
    assert all(made[d] <= sc.FAMINE_DAILY_FOOD for d in range(sc.FAMINE_FROM_DAY, 5))
    by_day = food(ev)["food_by_day"]
    n = proto["agents"]
    assert sum(d["eaten"] for d in by_day) / (n * len(by_day)) < 2.0
    late = by_day[sc.FAMINE_FROM_DAY:]
    assert sum(d["eaten"] for d in late) / (n * len(late)) < 1.5
    sim = Diligent(1, 30, StubMind(), sc.labeled_population(1, 30), "famine", {})
    sim.run(4)
    big = Counter()
    for e in sim.events:
        if e.kind == "produce" and "food" in e.data:
            big[e.tick // TICKS_PER_DAY] += e.data["food"]
    assert big[2] == big[3] == sc.FAMINE_DAILY_FOOD and big[1] > sc.FAMINE_DAILY_FOOD


def test_a_famine_farmer_is_told_when_the_day_is_spent():
    """The prompt stays true (T14a): once today's village cap is gone, a farmer's economy line says so."""
    proto = dict(TH["protocol"], days=3)
    pop = sc.labeled_population(proto["seed"], proto["agents"])
    sim = sc.ArmSim(proto["seed"], proto["agents"], StubMind(), pop, "famine", {})
    sim.run(2)
    farmer = next(a for a in sim.state["agents"] if a["role"] == "farmer")
    assert "yields 1 food" in sim._economy(farmer)
    sim.state["famine_made"] = {"day": 2, "food": sc.FAMINE_DAILY_FOOD}
    assert sim._economy(farmer).startswith("The field will give no more food today.")
    assert sim._harvest(farmer) == 0 and sim.state["famine_made"]["food"] == sc.FAMINE_DAILY_FOOD
    sim.state["tick"] += TICKS_PER_DAY  # a new day refills the yield
    assert "yields 1 food" in sim._economy(farmer)


def test_a_famine_log_from_before_the_cap_refuses_to_replay_by_name():
    from asiimov.fork import build_sim
    from asiimov.minds import ReplayDivergence
    from asiimov.probe import ProbeSim
    ev = sc.run_arm("famine", StubMind(), dict(PROTO, days=1))
    assert ev[0].data["famine"] == sc.FAMINE_VERSION
    assert "famine" not in sc.run_arm("full", StubMind(), dict(PROTO, days=1))[0].data  # other arms unchanged
    old = [type(ev[0])(**{**ev[0].__dict__, "data": {k: v for k, v in ev[0].data.items() if k != "famine"}})] + ev[1:]
    with pytest.raises(ReplayDivergence, match="famine version 1"):
        build_sim(old, StubMind(), "famine")
    with pytest.raises(ReplayDivergence, match="famine version 1"):
        ProbeSim(old, "famine", set())
    build_sim(ev, StubMind(), "famine")  # the current version builds
