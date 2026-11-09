"""T13 test 15 driver (asiimov.twins): placements, the noise band, the null floor and causal depth."""
import json
import random
from collections import Counter

import pytest

from asiimov.events import Event
from asiimov.fork import divergence, run_twin
from asiimov.minds import StubMind
from asiimov.sim import TICKS_PER_DAY, Sim
from asiimov.twins import causal_depth, day_features, diverging, grade, placements, t15
from asiimov.events import write_jsonl
from test_fork import PromptMind, Recording, logs

FIRE = 1 * TICKS_PER_DAY + 60


def base_world(mind, agents=40, days=5, family=3, seed=5):
    s = Sim(seed, agents, mind)
    s.run(days, {FIRE: [{"kind": "fire", "family": family}]})
    s.finish()
    return s.events


def test_dropping_the_base_fire_is_the_world_without_it():
    base = base_world(PromptMind())
    truth = Sim(5, 40, PromptMind())
    truth.run(5)
    truth.finish()
    twin, mind = run_twin(base, PromptMind(), drop=[(FIRE, {"kind": "fire", "family": 3})])
    assert logs(twin.events) == logs(truth.events)
    assert mind.counts["base"] > 0  # before the fire, everything came from the record
    assert min(divergence(base, twin.events).values()) >= FIRE


def test_dropping_an_action_the_base_does_not_have_is_refused():
    base = base_world(StubMind(), days=2)
    with pytest.raises(ValueError, match="no player action"):
        run_twin(base, StubMind(), drop=[(FIRE, {"kind": "fire", "family": 4})], serve=False)


def test_placements_follow_the_spec_draw():
    base = base_world(StubMind(), days=2)
    ps = placements(base)
    fams = base[0].data["population"]["families"]
    pool = sorted(f["id"] for f in fams if len(f["members"]) >= 3 and f["id"] != 3)
    assert [p["family"] for p in ps[1:]] == random.Random("t13").sample(pool, 2)
    assert ps[0] == {"name": "P1", "kind": "remove", "tick": FIRE, "family": 3}
    assert {p["tick"] for p in ps[1:]} == {3 * TICKS_PER_DAY + 60}
    assert ps[1]["family"] != ps[2]["family"]


def test_a_cell_counts_only_beyond_every_noise_twin():
    ref = {1: Counter({(0, "work_h"): 4}), 2: Counter({(0, "talks"): 2})}
    twin = {1: Counter({(0, "work_h"): 6}), 2: Counter({(0, "talks"): 3})}
    band = [{1: Counter({(0, "work_h"): 5}), 2: Counter({(0, "talks"): 3})},
            {1: Counter({(0, "work_h"): 2}), 2: Counter()}]
    # agent 1: |6-4| = 2 does not exceed the noise's |2-4| = 2 (a tie is not beyond the band)
    # agent 2: |3-2| = 1 does not exceed |0-2| = 2
    assert diverging(ref, twin, band, [1, 2]) == set()
    twin[1][(1, "receipt:steal")] = 1  # a cell no noise twin moved
    assert diverging(ref, twin, band, [1, 2]) == {1}
    assert diverging(ref, twin, [], [1, 2]) == {1, 2}  # no band (a stub): any difference counts


def test_day_features_count_hours_talks_and_receipts_from_the_placement_tick():
    def ev(i, tick, kind, actor, data, parents=()):
        return Event(i, tick, kind, "rule", actor, data, list(parents))
    t0 = 10
    events = [ev(0, 0, "genesis", None, {}), ev(1, 2, "wake", 1, {"place": "home_0"}),
              ev(2, 4, "activity", 1, {"activity": "work", "place": "farm"}),
              ev(3, 12, "talk", 1, {"with": 2}), ev(4, 13, "receipt", 1, {"action": "steal", "ok": True}),
              ev(5, 22, "sleep", 1, {}), ev(6, 5, "receipt", 1, {"action": "steal", "ok": True})]
    f = day_features(sorted(events, key=lambda e: e.tick), t0)[1]
    assert f[(0, "work_h")] == pytest.approx(12 / 6)  # ticks 10..21 at work, before the window not counted
    assert f[(0, "talks")] == 1 and f[(0, "receipt:steal")] == 1  # the tick-5 receipt is before t0


def test_causal_depth_is_the_longest_chain_to_a_reached_agent():
    def ev(i, parents, actor, tick=1):
        return Event(i, tick, "x", "rule", actor, {}, list(parents))
    events = [ev(0, [], None), ev(1, [0], 5), ev(2, [1], 5), ev(3, [2], 6), ev(4, [0], 7), ev(5, [3, 4], 8),
              ev(6, [5], 9, tick=99)]
    assert causal_depth(events, 0, {6}, 50) == 3
    assert causal_depth(events, 0, {7, 8}, 50) == 4  # 0-1-2-3-5, the longer of the two paths into 5
    assert causal_depth(events, 0, {9}, 50) == 0  # past the window
    assert causal_depth(events, 0, {42}, 50) == 0


def test_the_stub_driver_runs_every_placement_with_a_zero_band(tmp_path):
    base = base_world(StubMind())
    path = tmp_path / "base.jsonl"
    write_jsonl(base, str(path))
    rep = t15(str(path), StubMind(), str(tmp_path / "t15"), None, serve=False, workers=1)
    assert [p["name"] for p in rep["placements"]] == ["P1", "P2", "P3"]
    assert json.load(open(tmp_path / "t15" / "placements.json"))[1]["family"] == rep["placements"][1]["family"]
    for p in rep["placements"]:
        assert p["null_reach_by_noise_twin"] == [0] * 8  # a deterministic mind has no noise
        assert not set(p["reached"]) & set(p["family_members"])
    assert any(p["family_diverging"] for p in rep["placements"])  # the fires did change the burned families
    # A second call reloads every finished twin and reads the same.
    again = t15(str(path), StubMind(), str(tmp_path / "t15"), None, serve=False, workers=1)
    assert again["placements"] == rep["placements"]


def test_a_sampled_mind_gets_a_band_and_a_null_floor(tmp_path):
    base = base_world(PromptMind())
    path = tmp_path / "base.jsonl"
    write_jsonl(base, str(path))
    live = Recording(PromptMind())
    rep = t15(str(path), live, str(tmp_path / "t15"), None, serve=True, workers=1)
    assert all(r.tick >= FIRE for r in live.asked)  # nothing before the first placement was paid for
    noisy = [p for p in rep["placements"] if p["agents_live"]]
    assert noisy and any(sum(p["null_reach_by_noise_twin"]) > 0 for p in noisy)
    for p in rep["placements"]:
        s = json.load(open(tmp_path / "t15" / p["name"] / "noise-1" / "summary.json")) if p["agents_live"] else None
        if s:  # a noise twin forces exactly the intervention twin's live set, from the placement tick
            assert set(s["agents_live"]) >= set(p["agents_live"]) and s["live_by_reason"].get("forced")


def test_grade_reads_the_frozen_bar():
    row = {"reach": 12, "depth": 7, "null_reach": 1}
    assert grade([row] * 3)["verdict"] == "pass"
    assert grade([row, row, {**row, "reach": 3}])["verdict"] == "pass"  # medians
    assert grade([{**row, "depth": 5}] * 3)["verdict"] == "fail"
    assert grade([{**row, "null_reach": 5}] * 3)["verdict"] == "inconclusive (noise floor)"


# ---------------------------------------------------------------------------- test 8
from asiimov.twins import (interactions, read_t9, sign_test, stole, swap_pairs, swapped_genesis,  # noqa: E402
                           swapped_souls, t8, t9, theft_acts)


def test_swap_pairs_match_role_age_band_and_family_size():
    pop = base_world(StubMind(), days=1)[0].data["population"]
    size = {f["id"]: len(f["members"]) for f in pop["families"]}
    key = lambda i: (pop["agents"][i]["role"], pop["agents"][i]["age"] // 20, size[pop["agents"][i]["family"]])  # noqa: E731
    pairs = swap_pairs(pop)
    assert pairs and all(key(x) == key(y) and x < y for x, y in pairs)
    flat = [a for p in pairs for a in p]
    assert len(flat) == len(set(flat))


def test_the_text_swap_moves_only_the_soul_lines_and_leaves_params_with_the_body():
    pop = base_world(StubMind(), days=1)[0].data["population"]
    (x, y), = swap_pairs(pop)[:1]
    souls = swapped_souls(pop, [(x, y)])
    from asiimov.memory import render_soul
    ax, ay = pop["agents"][x], pop["agents"][y]
    assert souls[x].splitlines()[:3] == render_soul(ax).splitlines()[:3]  # name, age and role, rhythm: the body's
    assert souls[x].splitlines()[3:] == render_soul(ay).splitlines()[3:]  # character lines and anchors: the donor's
    g = swapped_genesis(pop, [(x, y)])
    assert g["agents"][x]["traits"] == ay["traits"] and g["agents"][x]["rhythm"] == ax["rhythm"]


def test_a_soul_text_twin_changes_prompts_from_tick_zero_but_no_params():
    base = base_world(PromptMind(), days=2)
    pop = base[0].data["population"]
    pairs = swap_pairs(pop)[:2]
    live = Recording(PromptMind())
    sim, mind = run_twin(base, live, souls=swapped_souls(pop, pairs))
    swapped = {a for p in pairs for a in p}
    assert {r.agent_id for r in live.asked if r.tick < TICKS_PER_DAY} & swapped  # their prompts changed at once
    for a in sim.state["agents"]:
        assert a["traits"] == pop["agents"][a["id"]]["traits"]


def test_on_the_stub_the_text_swap_fails_and_the_full_swap_is_reported_beside_it(tmp_path):
    base = base_world(StubMind(), agents=60, days=4)
    path = tmp_path / "base.jsonl"
    write_jsonl(base, str(path))
    rep = t8(str(path), StubMind(), str(tmp_path / "t8"), None, serve=False, workers=1)
    assert rep["follow_index"] == 0 and rep["verdict"] != "pass"  # the stub can't read the prompt
    assert rep["full_swap_descriptive"]["follow_index"] > 0.3  # but its code follows the params


# ---------------------------------------------------------------------------- test 9
def test_a_theft_is_possess_act_and_release_after_the_receipt():
    assert theft_acts(50, 3, 4, "steal") == {50: [{"kind": "possess", "agent": 3},
                                                 {"kind": "act", "action": "steal", "target": 4}],
                                            51: [{"kind": "release"}]}
    assert theft_acts(50, 3, 4, "wait")[50][1] == {"kind": "act", "action": "wait"}


def test_interactions_count_only_mind_authored_acts_toward_the_other():
    def ev(i, tick, kind, origin, actor, data, parents=()):
        return Event(i, tick, kind, origin, actor, data, list(parents))
    events = [ev(0, 0, "genesis", "chance", None, {}),
              ev(1, 5, "mind_deliberate", "mind", 4, {"kind": "deliberate"}),
              ev(2, 5, "receipt", "rule", 4, {"action": "ask_food", "ok": True, "why": "", "to": 3, "roll": 1}, [1]),
              ev(3, 6, "intent_set", "rule", 4, {}, [1]),
              ev(4, 7, "meet", "rule", 4, {"with": 3, "place": "plaza"}, [3]),
              ev(5, 8, "receipt", "rule", 4, {"action": "ask_food", "ok": True, "why": "", "to": 3, "roll": 1}),  # code
              ev(6, 9, "meet", "rule", 4, {"with": 7, "place": "plaza"}, [3]),  # someone else
              ev(7, 30, "meet", "rule", 4, {"with": 3, "place": "plaza"}, [3])]  # after the window
    assert interactions(events, 4, 3, 0, 20) == 2


def test_the_sign_test_and_the_precondition():
    assert sign_test(5, 5) == pytest.approx(1 / 32)
    rows = [{"stole": True, "theft": 1, "control": 5}] * 20
    assert read_t9(rows)["verdict"] == "pass"
    assert read_t9([{**r, "control": 3, "theft": 1} for r in rows])["verdict"] == "inconclusive (precondition)"
    assert read_t9(rows[:19] + [{"stole": False}])["verdict"] == "inconclusive (too few thefts)"
    assert read_t9([{**r, "theft": 4} for r in rows])["verdict"] == "fail"  # a 20% drop


def test_the_stub_theft_driver_injects_real_thefts(tmp_path):
    base = base_world(StubMind(), agents=60, days=4)
    path = tmp_path / "base.jsonl"
    write_jsonl(base, str(path))
    rep = t9(str(path), StubMind(), str(tmp_path / "t9"), None, serve=False, workers=1)
    assert rep["verdict"] != "pass" and rep["thefts"] > 0
    for r in rep["rows"]:
        theft = read_log_events(tmp_path / "t9" / f"{r['i']:03d}" / "probe" / "events.jsonl")
        assert stole(theft, r["tick"], r["thief"]) == r["stole"]
        if r["stole"]:
            ctl = read_log_events(tmp_path / "t9" / f"{r['i']:03d}" / "control" / "events.jsonl")
            acts = [e for e in ctl if e.kind == "act"]
            assert [a.data for a in acts] == [{"action": "wait"}] and acts[0].tick == r["tick"]


def read_log_events(p):
    from asiimov.fork import read_log
    return read_log(str(p))
