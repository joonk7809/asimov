"""Long unattended run (asiimov/soak.py): a clean world passes, and a broken one is reported, not hidden."""
import json

from asiimov import sim as sim_mod
from asiimov.soak import soak


def test_a_short_stub_soak_passes_and_replays(tmp_path):
    res = soak(str(tmp_path), seed=4, agents=12, days=2, player={sim_mod.TICKS_PER_DAY + 60: [{"kind": "fire", "family": 0}]})
    assert res["ok"] and res["violations"] == 0 and res["replay_hash"] == res["live_hash"]
    assert [d["day"] for d in res["per_day"]] == [0, 1]
    assert json.load(open(tmp_path / "soak.json"))["ok"]


def test_a_coin_leak_is_counted_on_every_tick_and_the_run_keeps_going(tmp_path, monkeypatch):
    step = sim_mod.Sim.step

    def leaky(self, acts=()):
        step(self, acts)
        self.state["agents"][0]["inv"]["coin"] += 1  # money from nowhere, outside any event

    monkeypatch.setattr(sim_mod.Sim, "step", leaky)
    res = soak(str(tmp_path), seed=4, agents=6, days=1)
    assert not res["ok"] and res["error"] is None
    assert res["violations"] == sim_mod.TICKS_PER_DAY  # the coin ledger, once per tick
    assert res["first_violations"][0].startswith("tick 0: coin ledger")
    assert len(res["first_violations"]) == 50


def test_a_crash_is_reported_with_its_tick(tmp_path, monkeypatch):
    step = sim_mod.Sim.step

    def dies(self, acts=()):
        if self.state["tick"] == 30:
            raise RuntimeError("boom")
        step(self, acts)

    monkeypatch.setattr(sim_mod.Sim, "step", dies)
    res = soak(str(tmp_path), seed=4, agents=6, days=1)
    assert not res["ok"] and res["error"] == "RuntimeError at tick 30: boom"


# ------------------------------------------------------------------ soak --check (G1's verdict, plan/g1-g3-runs-spec.md)
def test_a_stub_soak_log_passes_the_check_with_three_equal_hashes(tmp_path):
    from asiimov.soak import check
    res = soak(str(tmp_path / "w"), seed=4, agents=12, days=2,
               player={sim_mod.TICKS_PER_DAY + 60: [{"kind": "fire", "family": 0}]})
    assert res["ok"]
    rep = check(str(tmp_path / "w" / "events.jsonl"))
    assert rep["ok"] and rep["error"] is None and rep["violations"] == 0
    assert rep["run_hash"] == rep["check_hash"] == rep["replay_hash"] == res["live_hash"]
    assert [d["day"] for d in rep["per_day"]] == [0, 1]
    assert "ever_sick" in rep["per_day"][-1]
    assert json.load(open(tmp_path / "w" / "soak-check.json"))["ok"]


def test_a_log_with_one_edited_reply_fails_the_check(tmp_path):
    from asiimov.soak import check
    soak(str(tmp_path / "w"), seed=4, agents=12, days=2)
    log = tmp_path / "w" / "events.jsonl"
    lines = log.read_text().splitlines()
    i = next(i for i, l in enumerate(lines) if json.loads(l).get("kind") == "mind_plan")
    ev = json.loads(lines[i])
    ev["mind"]["raw_output"] = json.dumps({"plan": [{"activity": "home", "start_h": 6, "end_h": 22}]})
    lines[i] = json.dumps(ev, sort_keys=True, separators=(",", ":"))
    log.write_text("\n".join(lines) + "\n")
    rep = check(str(log), str(tmp_path / "rep.json"))
    assert not rep["ok"] and rep["error"] is not None and not rep["hashes_equal"]


def test_an_unfinished_log_fails_the_check(tmp_path):
    from asiimov.soak import check
    soak(str(tmp_path / "w"), seed=4, agents=6, days=1)
    log = tmp_path / "w" / "events.jsonl"
    log.write_text("".join(log.read_text().splitlines(keepends=True)[:-1]))  # drop the end event
    rep = check(str(log), str(tmp_path / "rep.json"))
    assert not rep["ok"] and "no end event" in rep["error"]
