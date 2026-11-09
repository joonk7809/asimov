"""B5 stakes in the engine (plan/b5-build-spec.md). tithe-1: the coin sink. pay-1: pay compression."""
from asiimov.invariants import Checker
from asiimov.minds import ReplayMind, StubMind
from asiimov.params import generate_population
from asiimov.replay import replay
from asiimov.sim import DEFAULT_STAKES, TICKS_PER_DAY, Sim

FIRE = {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 2}]}


def checked(sim, days, player=None):
    c, bad = Checker(sim), []
    player = player or {}
    while sim.state["tick"] < days * TICKS_PER_DAY:
        bad += c.step(player.get(sim.state["tick"], []))
    sim.finish()
    return bad


def test_genesis_records_the_stakes_and_a_replay_reads_them_from_the_log():
    sim = Sim(4, 30, StubMind())
    assert checked(sim, 3, FIRE) == []
    assert sim.events[0].data["population"]["stakes"] == DEFAULT_STAKES
    assert replay(sim.events).state_hash() == sim.state_hash()
    # an ablation with the tithe off is recorded as such, and its replay keeps it off
    off = Sim(4, 30, StubMind(), genesis={**generate_population(4, 30), "stakes": {}})
    assert checked(off, 3, FIRE) == []
    assert not any(e.kind in ("tithe", "chest_spent") for e in off.events)
    assert replay(off.events).state_hash() == off.state_hash() != sim.state_hash()


def test_midnight_tithe_takes_its_share_above_the_free_coin_from_adults_only():
    sim = Sim(5, 30, StubMind())
    sim.run_until(TICKS_PER_DAY)
    before = {a["id"]: a["inv"]["coin"] for a in sim.state["agents"]}
    chest = sim.state["chest"]
    n0 = len(sim.events)
    sim.step()
    t = DEFAULT_STAKES["tithe"]
    paid = {e.actor: e.data["coin"] for e in sim.events[n0:] if e.kind == "tithe"}
    for a in sim.state["agents"]:
        want = 0 if a["role"] in ("child", "elder") else max(0, before[a["id"]] - t["free"]) * t["rate_pct"] // 100
        assert paid.get(a["id"], 0) == want
    assert paid and sim.state["chest"] == chest + sum(paid.values())


def test_the_chest_keeps_its_reserve_and_the_rest_leaves_the_economy():
    sim = Sim(1, 100, StubMind())
    assert checked(sim, 4) == []  # Checker's coin and chest ledgers balance every tick
    spent = [e for e in sim.events if e.kind == "chest_spent"]
    assert spent and all(e.data["chest"] == DEFAULT_STAKES["tithe"]["reserve"] for e in spent)
    assert sim.state["chest"] <= DEFAULT_STAKES["tithe"]["reserve"]


def test_a_world_without_stakes_keeps_engine_17_state():
    off = Sim(2, 20, StubMind(), genesis={**generate_population(2, 20), "stakes": {}})
    assert "chest" not in off.state and "stakes" not in off.state
    rep = Sim(2, 20, ReplayMind([]), genesis={**generate_population(2, 20), "stakes": {}})
    assert rep.state_hash() == off.state_hash()


def test_a_tithe_note_is_an_everyday_memory_not_a_gift():
    # "gave" weighs 4 in IMPORTANT_WORDS (meant for gifts); a nightly tithe note scored so would crowd out
    # everyday memories, then real gifts (steering's condition on tithe-1)
    sim = Sim(1, 60, StubMind())
    sim.run(3)
    eps = [e for a in sim.state["agents"] for e in a["episodes"] if "village chest" in e["text"]]
    assert eps and all(e["importance"] == 1 for e in eps)
def _rates(sim):
    return {a["id"]: sim._work_rate(a) for a in sim.state["agents"] if a["role"] not in ("child", "elder")}


def test_pay_pulls_each_working_adult_toward_its_trades_mean_and_keeps_that_mean():
    """Steering's capacity check (21:22Z): each trade's mean work speed equals the unpulled world's (tithe-1's),
    within rounding to nearest, at every seed tried."""
    pull = DEFAULT_STAKES["pay"]["pull_pct"]
    for seed in range(1, 11):
        on = Sim(seed, 100, StubMind())
        off = Sim(seed, 100, StubMind(), genesis={**generate_population(seed, 100), "stakes": {"tithe": DEFAULT_STAKES["tithe"]}})
        r_on, r_off = _rates(on), _rates(off)
        for t in {a["role"] for a in on.state["agents"] if a["id"] in r_on}:
            ids = [a["id"] for a in on.state["agents"] if a["role"] == t]
            mean = sum(r_off[i] for i in ids) / len(ids)  # tithe-1's exact trade average
            for i in ids:
                assert abs(r_on[i] - (mean + (r_off[i] - mean) * (100 - pull) / 100)) <= 0.5
                assert min(r_off[i], mean) - 0.5 <= r_on[i] <= max(r_off[i], mean) + 0.5  # nobody passes the mean
            assert abs(sum(r_on[i] for i in ids) / len(ids) - mean) <= 0.5
        assert max(r_on.values()) / min(r_on.values()) < max(r_off.values()) / min(r_off.values())


def test_pay_off_is_the_body_speed_and_is_recorded_in_genesis():
    off = Sim(3, 30, StubMind(), genesis={**generate_population(3, 30), "stakes": {"tithe": DEFAULT_STAKES["tithe"]}})
    for a in off.state["agents"]:
        assert off._work_rate(a) == a["body"]["strength"] * a["body"]["stamina"] // 1000
    assert "pay" not in off.events[0].data["population"]["stakes"]
    assert checked(off, 2, FIRE) == []
    assert replay(off.events).state_hash() == off.state_hash()
