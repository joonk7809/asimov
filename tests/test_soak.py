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
