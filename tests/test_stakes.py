"""B5 stakes in the engine (plan/b5-build-spec.md). tithe-1: the coin sink. pay-1: pay compression. year-1: the year
and the floating price."""
from asiimov.invariants import Checker
from asiimov.minds import ReplayMind, StubMind
from asiimov.params import generate_population
from asiimov.replay import replay
from asiimov.sim import DEFAULT_STAKES, FARM_YIELD, FOOD_PRICE, STALL_PARENTS, TICKS_PER_DAY, Sim

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


# ------------------------------------------------------------------ year-1
YEAR = DEFAULT_STAKES["year"]
NO_YEAR = {k: v for k, v in DEFAULT_STAKES.items() if k != "year"}


def _farmer(sim):
    return next(a for a in sim.state["agents"] if a["role"] == "farmer")


def test_the_year_yields_what_a_flat_year_yields_and_nothing_in_winter():
    assert sum(YEAR["yields"]) == FARM_YIELD * len(YEAR["yields"]) == 21
    sim = Sim(1, 20, StubMind())
    f, n = _farmer(sim), len(YEAR["yields"])
    made = []
    for day in range(2 * n):
        sim.state["tick"] = day * TICKS_PER_DAY + 70
        made.append(sim._food_yield(f))
    assert made == YEAR["yields"] * 2
    winter = [d for d in range(n) if YEAR["yields"][d] == 0]
    assert winter == [5, 6]
    sim.state["tick"] = 5 * TICKS_PER_DAY + 70
    assert "will give no more food today" in sim._economy(f)
    off = Sim(1, 20, StubMind(), genesis={**generate_population(1, 20), "stakes": NO_YEAR})
    assert "stall" not in off.state and off._food_yield(_farmer(off)) == FARM_YIELD and off._price() == FOOD_PRICE


def test_the_price_ladder_follows_the_stall_against_a_day_of_the_villages_meals():
    sim = Sim(3, 60, StubMind())
    need = sim._day_need()
    p = YEAR["prices"]
    for stock, want in ((need * 3, p[0]), (need, p[0]), (need - 1, p[1]), ((need + 1) // 2, p[1]),
                        ((need + 1) // 2 - 1, p[2]), ((need + 3) // 4, p[2]), ((need + 3) // 4 - 1, p[3]), (0, p[3])):
        sim.state["market_food"] = stock
        assert sim._price_now() == want, (stock, need)
    # the step's end sets the next tick's price, as a step event; farmers sell at it less 1
    sim.state["market_food"] = need * 2
    n0 = len(sim.events)
    sim.step()
    assert sim._price() == p[0] and sim._sell_price() == p[0] - 1 and sim._gift_coin() == p[0]
    assert [(e.data["was"], e.data["price"]) for e in sim.events[n0:] if e.kind == "price"] == [(p[3], p[0])]


def test_the_live_price_is_in_every_clause_and_line_that_names_it():
    from asiimov.sim import action_clauses
    sim = Sim(1, 12, StubMind())
    a = next(x for x in sim.state["agents"] if x["role"] == "farmer")
    sim.state["stall"]["price"] = 5
    assert "sell at 4 coin each. Food costs 5 coin" in sim._economy(a)
    c = action_clauses(5)
    assert c["buy_from"] == "buy_from offers a person here 5 coin for 1 food; they may say no"
    assert c["give_coin"] == "give_coin hands them 5 coin"
    view = sim._view(a, {a["place"]: [a["id"]]})
    assert view["food_price"] == 5


def _harvest_day_awake(sim):
    """Run to the harvest day (the last day with a yield before winter), with its adults up two hours."""
    day = next(d for d in range(len(YEAR["yields"])) if YEAR["yields"][d] and not YEAR["yields"][d + 1])
    sim.run_until(day * TICKS_PER_DAY)
    return day


def test_on_the_harvest_day_every_working_adult_stocks_its_households_winter_by_rule():
    sim = Sim(2, 40, StubMind())
    c, bad = Checker(sim), []
    day = _harvest_day_awake(sim)
    while sim.state["tick"] < (day + 1) * TICKS_PER_DAY:
        bad += c.step()
    assert bad == []
    stock = [e for e in sim.events if e.kind == "stock"]
    assert stock and all(e.tick // TICKS_PER_DAY == day for e in stock)
    assert len({e.actor for e in stock}) == len(stock)  # once each
    ags = sim.state["agents"]
    for e in stock:
        a = ags[e.actor]
        assert a["role"] not in ("child", "elder") and e.data["winter_days"] == 2
        assert e.data["order"] == 3 * 2 * sim._stock_share(a)
    for f in sim.state["families"]:  # each person's winter meals are ordered once (steering on #48)
        workers = [m for m in f["members"] if ags[m]["role"] not in ("child", "elder")]
        if workers:
            assert sum(sim._stock_share(ags[m]) for m in workers) == len(f["members"])
    buys = [e for e in sim.events if e.kind == "receipt" and e.data.get("stock")]
    assert buys and all(e.data["food"] <= next(s.data["order"] for s in stock if s.actor == e.actor) for e in buys)
    # a stock buy is the order's own trip, on its day; an order its trip didn't fill lapses at the day's end
    order_of = {s.actor: s.id for s in stock}
    assert all(e.parents[0] == order_of[e.actor] and e.tick // TICKS_PER_DAY == day for e in buys)
    assert sim.state["stall"]["orders"] == {}


def test_a_winter_order_is_filled_only_by_its_own_trip():
    sim = Sim(1, 40, StubMind())
    day = _harvest_day_awake(sim)
    while not sim.state["stall"]["orders"]:
        sim.step()
    who = int(next(iter(sim.state["stall"]["orders"])))
    a = sim.state["agents"][who]
    a["place"], a["inv"]["coin"] = "market", 200
    sim._buy(a, 0)  # some other buy (a plan's or a player's): an ordinary one, and the order still stands
    assert not sim.events[-1].data.get("stock") and str(who) in sim.state["stall"]["orders"]
    sim.step([{"kind": "possess", "agent": who}])
    a["place"] = "market"
    sim._buy(a, sim.state["stall"]["orders"].get(str(who), {"event": -1})["event"])
    assert not sim.events[-1].data.get("stock") and str(who) not in sim.state["stall"]["orders"]  # dropped


def test_a_possessed_body_gets_no_winter_order_but_lives_in_the_year():
    sim = Sim(2, 40, StubMind())
    who = next(a["id"] for a in sim.state["agents"] if a["role"] == "crafter")
    sim.run_until(TICKS_PER_DAY)
    sim.step([{"kind": "possess", "agent": who}])
    day = _harvest_day_awake(sim)
    sim.run_until((day + 1) * TICKS_PER_DAY)
    assert sim.state["player"]["agent"] == who
    assert not any(e.kind == "stock" and e.actor == who for e in sim.events)
    assert any(e.kind == "stock" for e in sim.events)
    assert sim._price() in YEAR["prices"]  # its coin buys at the live price like anyone's


def test_the_stall_keeps_its_stock_from_the_harvest_night_through_winter():
    sim = Sim(1, 60, StubMind())
    sim.run(15)
    nights = {e.tick // TICKS_PER_DAY % len(YEAR["yields"]) for e in sim.events if e.kind == "spoil"}
    assert nights and not nights & {4, 5, 6}


def test_every_price_step_and_failed_buy_cites_the_stall_moves_that_caused_it():
    sim = Sim(1, 60, StubMind(), genesis={**generate_population(1, 60),
                                          "stakes": {**DEFAULT_STAKES, "year": {**YEAR, "yields": [2, 2, 2, 2, 3, 0, 0]}}})
    c, bad = Checker(sim), []
    while sim.state["tick"] < 10 * TICKS_PER_DAY:
        bad += c.step()
    sim.finish()
    assert bad == []
    by_id = {e.id: e for e in sim.events}
    steps = [e for e in sim.events if e.kind == "price"]
    assert any(e.data["price"] > e.data["was"] for e in steps) and any(e.data["price"] < e.data["was"] for e in steps)
    for e in steps:
        kinds = {by_id[p].kind for p in e.parents}
        assert e.parents and len(e.parents) <= STALL_PARENTS
        if e.data["price"] > e.data["was"]:  # a rise: the buys that drew the stall down
            assert all(by_id[p].kind == "receipt" and by_id[p].data["action"] == "buy_food" and by_id[p].data["food"]
                       for p in e.parents)
        else:  # a fall: the sells that filled it
            assert kinds == {"sell"}
        assert all(by_id[p].tick <= e.tick for p in e.parents)
    fails = [e for e in sim.events if e.kind == "receipt" and e.data["action"] == "buy_food" and not e.data["ok"]
             and e.data["why"] in ("no stock", "no coin")]
    assert fails
    base = YEAR["prices"][0]
    for e in fails:  # review of #48: only a real cause is cited
        cause = e.parents[1:]
        if e.data["why"] == "no stock":
            assert all(by_id[p].kind == "receipt" and by_id[p].data["action"] == "buy_food" for p in cause)
        else:  # too little coin: the step that put the price above its base, and only then
            step = [x for x in steps if x.id < e.id][-1:]
            if step and step[0].data["price"] > base:
                assert cause == [step[0].id]
            else:
                assert cause == []
    assert replay(sim.events).state_hash() == sim.state_hash()


def test_a_trade_at_a_raised_price_cites_the_step_in_force():
    """Steering on #48: a market buy, a sell, a buy_from and a give_coin made while the price is off its base cite
    the price step that set it; at the base price they cite none."""
    sim = Sim(1, 100, StubMind())
    sim.run(8)
    sim.finish()
    by_id = {e.id: e for e in sim.events}
    steps = [e for e in sim.events if e.kind == "price"]
    base = YEAR["prices"][0]

    def in_force(e):
        last = [x for x in steps if x.id < e.id][-1:]
        return last[0] if last and last[0].data["price"] != base else None

    trades = [e for e in sim.events if (e.kind == "receipt" and e.data["ok"] and (
        (e.data["action"] == "buy_food" and e.data.get("food")) or e.data["action"] in ("buy_from", "give_coin")))
        or e.kind == "sell"]
    raised = 0
    for e in trades:
        step = in_force(e)
        cites = [p for p in e.parents if by_id[p].kind == "price"]
        assert cites == ([step.id] if step else []), (e.kind, e.data)
        raised += step is not None
    assert raised  # some trades happened at a raised price
