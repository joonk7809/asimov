"""Engine 7 intent loop (T15, plan/t15-intent-loop-spec.md in the shared folder): intents that persist,
appraised occasions, person steps and gifts, plannable shopping, the call budget and authorship."""
import json
from collections import Counter

from asiimov import authorship as au
from asiimov.fork import run_twin
from asiimov.minds import MindReply, StubMind
from asiimov.sim import DAILY_MIND_CALLS, FOOD_PRICE, GIFT_COIN, TICKS_PER_DAY, Sim


class Scripted(StubMind):
    """Plans like the stub unless given a plan; deliberates with `answer(req)` (None: like the stub)."""

    def __init__(self, answer=None, plan=None):
        self.answer, self.plan = answer, plan
        self.asked = []

    def decide(self, req):
        if req.kind == "deliberate":
            self.asked.append(req)
            out = self.answer(req) if self.answer else None
            if out is not None:
                return MindReply(json.dumps({"reason": "x", "goal": "", "then": [], **out}), "scripted", {})
        if req.kind == "plan" and self.plan is not None:
            return MindReply(json.dumps({"plan": self.plan}), "scripted", {})
        return super().decide(req)


def awake(sim, ids, place=None):
    for a in sim.state["agents"]:
        a.update(asleep=True, wake_tick=10**6, sleep_tick=None)
    for i in ids:
        a = sim.state["agents"][i]
        a.update(asleep=False, wake_tick=0, sleep_tick=10**6, plan=[], habit="home", activity="home",
                 place=place or a["home"])
    return [sim.state["agents"][i] for i in ids]


def deliberate_now(sim, a, out, occupancy=None):
    """Apply one deliberation reply to `a` as the commit would."""
    eid = sim._emit("mind_deliberate", "mind", a["id"], {"kind": "deliberate"}, [], mind={})
    sim._resolve(a, out, eid)
    return eid


def kinds(sim, start=0):
    return [e.kind for e in sim.events[start:]]


def test_a_goal_makes_an_intent_whose_steps_run_in_order_and_log_each_stage():
    sim = Sim(1, 12, StubMind())
    a, b = awake(sim, [0, 5])
    a["inv"]["coin"] = 20
    n0 = len(sim.events)
    deliberate_now(sim, a, {"action": "seek", "target": 5, "reason": "x", "goal": "make up with Bo",
                            "then": [{"do": "give_coin", "target": 5}, {"do": "rest", "target": None}]})
    assert a["intent"]["goal"] == "make up with Bo" and a["override"] is None
    set_ev = next(e for e in sim.events[n0:] if e.kind == "intent_set")
    assert [s["do"] for s in set_ev.data["steps"]] == ["seek", "give_coin", "rest"]
    for _ in range(30):
        sim.step()
    log = [e for e in sim.events[n0:] if e.actor == 0 and e.kind in ("intent_step", "intent_done", "meet")]
    assert [e.kind for e in log] == ["meet", "intent_step", "intent_step", "intent_step", "intent_done"]
    assert log[0].data["with"] == 5
    gift = next(e for e in sim.events[n0:] if e.kind == "receipt" and e.data["action"] == "give_coin")
    assert gift.data["ok"] and gift.data["coin"] == GIFT_COIN and gift.parents == [set_ev.id]
    assert any(e.kind == "intent_done" for e in sim.events[n0:]) and a["intent"] is None
    assert sim._rel(b, 0)["affinity"] >= 100 and b["inv"]["coin"] >= GIFT_COIN


def test_an_intent_outlives_the_night_and_the_next_prompts_show_it():
    from asiimov.memory import assemble_prompt
    mind = Scripted(lambda req: {"action": "work", "target": None, "goal": "save for a plough",
                                 "then": [{"do": "work", "target": None}] * 3})
    sim = Sim(1, 20, mind)
    a = sim.state["agents"][3]
    sim.run_until(a["wake_tick"] + 3)
    sim.run_until(a["sleep_tick"] - 10)  # late in the day: 4 steps of 3 hours can't finish before bed
    a["occasion"] = {"why": "fire", "salience": 1000, "who": None, "event": None, "detail": "", "tick": sim.state["tick"]}
    a["cooldown_until"] = 0
    sim.step()
    assert a["intent"]["goal"] == "save for a plough"
    mind.answer = lambda req: {"action": "wait", "target": None}
    sim.run_until(a["sleep_tick"] + 1)
    left = a["intent"]["left"]
    assert a["asleep"] and a["intent"]["goal"] == "save for a plough"
    sim.run_until(a["wake_tick"])  # the step clock stops while asleep
    assert a["intent"]["left"] == left
    prompt, _ = assemble_prompt(a, sim.state, "plan", "")
    assert "# My intentions\n- save for a plough: now work" in prompt
    sim.run_until(a["wake_tick"] + 40)
    assert any(e.kind == "intent_step" and e.actor == 3 and e.tick > a["wake_tick"] for e in sim.events)


def test_a_new_goal_drops_the_old_intent_and_a_one_off_only_pauses_it():
    sim = Sim(1, 12, StubMind())
    (a,) = awake(sim, [0])
    deliberate_now(sim, a, {"action": "work", "target": None, "goal": "first", "then": []})
    first = a["intent"]["event"]
    deliberate_now(sim, a, {"action": "rest", "target": None, "goal": "", "then": [{"do": "work", "target": None}]})
    assert a["intent"]["event"] == first and a["override"]["do"] == "rest"  # one-off: `then` is ignored
    deliberate_now(sim, a, {"action": "explore", "target": None, "goal": "second", "then": []})
    dropped = [e for e in sim.events if e.kind == "intent_dropped"]
    assert len(dropped) == 1 and dropped[0].parents == [first] and a["intent"]["goal"] == "second"
    assert a["override"] is None


def test_person_steps_without_a_valid_target_are_dropped_not_run():
    sim = Sim(1, 12, StubMind())
    (a,) = awake(sim, [0])
    deliberate_now(sim, a, {"action": "wait", "target": None, "goal": "g",
                            "then": [{"do": "seek", "target": 0}, {"do": "give_food", "target": True},
                                     {"do": "avoid", "target": 99}, {"do": "fly"}, {"do": "rest", "target": 4}]})
    set_ev = next(e for e in sim.events if e.kind == "intent_set")
    assert set_ev.data["dropped"] == 4 and set_ev.data["steps"] == [{"do": "rest", "target": None}]


def test_go_market_buys_on_arrival_so_one_deliberation_feeds_the_day():
    sim = Sim(1, 12, StubMind())
    (a,) = awake(sim, [0])
    a["inv"].update(food=0, coin=30)
    sim.state["market_food"] = 50
    deliberate_now(sim, a, {"action": "go_market", "target": None})
    sim.step()
    bought = [e for e in sim.events if e.kind == "receipt" and e.data["action"] == "buy_food"]
    assert len(bought) == 1 and bought[0].data["ok"] and a["inv"]["food"] >= 1 and a["override"] is None


def test_a_planned_shop_block_buys_without_any_deliberation():
    mind = Scripted(plan=[{"activity": "shop", "start_h": 0, "end_h": 24}])
    sim = Sim(2, 20, mind)
    sim.run(1)
    buys = [e for e in sim.events if e.kind == "receipt" and e.data["action"] == "buy_food"]
    planned = [e for e in buys if sim.events[e.parents[0]].kind == "mind_plan"]
    assert any(e.data["ok"] and e.data["food"] > 0 for e in planned)


def test_a_seek_that_never_finds_its_target_fails_and_is_an_occasion():
    sim = Sim(1, 12, StubMind())
    a, b = awake(sim, [0, 5])
    a["cognition"]["deliberation_threshold"] = 0
    b.update(asleep=True, wake_tick=10**6)  # asleep at home the whole time
    deliberate_now(sim, a, {"action": "seek", "target": 5, "goal": "see Bo", "then": []})
    for _ in range(20):
        sim.step()
    failed = [e for e in sim.events if e.kind == "intent_failed"]
    assert failed and failed[0].data["why"] == "time ran out"
    assert a["intent"] is None and any(r.kind == "mind_deliberate" and r.actor == 0
                                       for r in sim.events[failed[0].id:])


def test_occasions_below_the_threshold_never_wake_the_mind():
    sim = Sim(1, 12, StubMind())
    (a,) = awake(sim, [0])
    a["cognition"]["deliberation_threshold"] = 700
    sim._occasion(a, "gift", 699, who=1)
    assert a["occasion"] is None
    sim._occasion(a, "gift", 700, who=1)
    assert a["occasion"]["why"] == "gift"
    sim._occasion(a, "failed", 650)  # a weaker one doesn't displace it
    assert a["occasion"]["why"] == "gift"


def test_theft_gift_and_fire_are_occasions_for_the_person_they_happen_to():
    sim = Sim(1, 12, StubMind())
    a, b = awake(sim, [0, 1], place="plaza")
    for x in (a, b):
        x["cognition"]["deliberation_threshold"] = 0
    b["inv"]["food"] = 5
    b["traits"]["emotionality"] = 1000  # always notices
    sim._steal(a, b, deliberate_now(sim, a, {"action": "wait", "target": None}))
    assert b["occasion"]["why"] in ("robbed", "rob_tried", "food_missing")
    b["occasion"] = None
    a["inv"]["food"] = 3
    sim._give(a, b, "give_food", 0)
    assert b["occasion"]["why"] == "gift" and b["occasion"]["detail"] == "1 food"
    sim._player({"kind": "fire", "family": a["family"]})
    assert a["occasion"]["why"] == "fire"


def test_family_and_strong_feelings_nearby_are_noticed_once_a_day():
    sim = Sim(1, 30, StubMind())
    a = sim.state["agents"][0]
    kin = [k for k in sim.state["families"][a["family"]]["members"] if k != 0]
    stranger = next(i for i in range(30) if i not in kin and i != 0)
    awake(sim, [0, stranger] + kin, place="plaza")
    a["cognition"]["deliberation_threshold"] = 0
    occ = {"plaza": sorted([0, stranger] + kin)}
    noticed = lambda: [(o["who"], o["detail"]) for o in iter(lambda: sim._notice(a, occ), None)]  # noqa: E731
    assert sorted(noticed()) == [(k, "care about") for k in sorted(kin)]  # family counts, a stranger doesn't
    assert noticed() == []  # once a day
    sim._rel(a, stranger)["affinity"] = -700
    sim.state["tick"] += TICKS_PER_DAY
    assert (stranger, "can't stand") in noticed()


def test_mind_calls_stay_within_the_daily_budget():
    """Every agent's threshold at 0 and a mind that always sets a new goal: as many occasions as the
    world can make. No agent gets more than DAILY_MIND_CALLS calls in a calendar day."""
    def answer(req):
        who = req.view["people"][0] if req.view["people"] else None
        if who is None:
            return {"action": "work", "target": None, "goal": "g", "then": [{"do": "shop", "target": None}]}
        return {"action": "seek", "target": who, "goal": "g", "then": [{"do": "give_food", "target": who}]}
    sim = Sim(4, 40, Scripted(answer))
    for a in sim.state["agents"]:
        a["cognition"]["deliberation_threshold"] = 0
    sim.run(3, {TICKS_PER_DAY + 60: [{"kind": "fire", "family": 1}]})
    calls = Counter((e.actor, e.tick // TICKS_PER_DAY) for e in sim.events if e.origin == "mind")
    assert max(calls.values()) <= DAILY_MIND_CALLS
    assert max(calls.values()) >= DAILY_MIND_CALLS - 3  # the budget, not the occasions, is what binds
    assert Counter(e.kind for e in sim.events)["meet"] > 0


def test_hungry_with_coin_is_its_own_occasion_and_the_prompt_says_the_market_is_always_open():
    mind = Scripted(lambda req: {"action": "wait", "target": None})
    sim = Sim(1, 12, mind)
    (a,) = awake(sim, [0])
    a["inv"].update(food=0, coin=3 * FOOD_PRICE)
    a["needs"]["hunger"] = 990
    sim.step()
    req = next(r for r in mind.asked if r.agent_id == 0)
    assert req.view["occasion"]["why"] == "hungry_coin"
    assert "Why you stop to think: you are hungry with no food, and carry enough coin to buy some." in req.prompt
    assert "which is open at all hours" in req.prompt and "go_market takes you to the market" in req.prompt


def test_authorship_tags_intent_steps_and_keeps_precedence():
    def answer(req):
        if req.view["occasion"]["why"] in ("hungry", "hungry_coin"):
            return {"action": "go_market", "target": None}
        return {"action": "explore", "target": None, "goal": "look around", "then": [{"do": "rest", "target": None}]}
    sim = Sim(1, 30, Scripted(answer))
    sim.run(2, {200: [{"kind": "fire", "family": 2}]})
    sim.finish()
    au_sim = au.replay_authorship(sim.events)
    authors = Counter(r[2] for r in au_sim.records)
    assert authors["intent"] > 0 and authors["deliberate"] > 0 and authors["plan"] > 0
    assert au_sim.state_hash() == sim.state_hash()


def test_a_null_twin_of_an_engine_7_log_is_byte_identical():
    sim = Sim(1, 40, StubMind())
    sim.run(3, {200: [{"kind": "fire", "family": 2}]})
    sim.finish()
    assert any(e.kind == "intent_set" for e in sim.events)
    twin, mind = run_twin(sim.events, StubMind())
    assert [e.to_json() for e in twin.events] == [e.to_json() for e in sim.events]
    assert mind.counts.get("live", 0) == 0


def test_a_visit_is_scored_by_where_the_agent_actually_went():
    """PR #19 review: the effective-share memo keyed places by activity label, so every visit counted as
    the agent's own home. Now each tick is scored from the place the agent is actually at."""
    from asiimov.params import generate_population
    pop = generate_population(4, 40)
    for p in pop["agents"]:
        p["cognition"]["deliberation_threshold"] = 0
    def answer(req):
        who = req.view["people"][-1] if req.view["people"] else None
        return {"action": "seek", "target": who, "goal": "g", "then": []} if who is not None else None
    sim = Sim(4, 40, Scripted(answer), genesis=pop)
    sim.run(2)
    sim.finish()
    au_sim = au.replay_authorship(sim.events)
    homes = {a["id"]: a["home"] for a in sim.state["agents"]}
    visits = [(t, x) for t, x, author, act, _ in au_sim.records if act == "visit"]
    away = [k for k in visits if au_sim.effect[k][0][0] != homes[k[1]]]
    assert away, "some visits must leave home"
    assert all(au_sim.effect[k][0] != au_sim.effect[k][1] for k in away if au_sim.effect[k][1][0] == homes[k[1]])


# ------------------------------------------------------------------ engine 9 (T17): what a failed buy tells the mind
def test_a_failed_buy_says_whether_stock_or_coin_was_missing():
    sim = Sim(1, 12, StubMind())
    a, b = awake(sim, [0, 1], place="market")
    for x in (a, b):
        x["cognition"]["deliberation_threshold"] = 0
        x["inv"]["food"] = 0
    a["inv"]["coin"], b["inv"]["coin"] = 30, 1
    sim.state["market_food"] = 5
    sim._buy(b, 0)
    sim.state["market_food"] = 0
    sim._buy(a, 0)
    fails = {e.actor: e.data["why"] for e in sim.events if e.kind == "receipt" and not e.data["ok"]}
    assert fails == {0: "no stock", 1: "no coin"}
    assert sim._occasion_text(b, b["occasion"]).endswith("you could not buy food (not enough coin).")
    assert sim._occasion_text(a, a["occasion"]).endswith("you could not buy food (no food for sale).")


def test_a_shop_step_that_finds_no_food_says_so_on_the_blocked_line():
    sim = Sim(1, 12, StubMind())
    (a,) = awake(sim, [0])
    a["cognition"]["deliberation_threshold"] = 0
    a["inv"].update(food=0, coin=30)
    sim.state["market_food"] = 0
    deliberate_now(sim, a, {"action": "go_market", "target": None, "goal": "feed the family",
                            "then": [{"do": "rest", "target": None}]})
    sim._act(a)
    assert a["occasion"]["why"] == "blocked"
    assert sim._occasion_text(a, a["occasion"]).endswith("give up on what you set out to do: feed the family (no food for sale).")


def test_the_market_belief_says_what_the_agent_last_saw_there_today():
    sim = Sim(1, 12, StubMind())
    (a,) = awake(sim, [0], place="market")
    a.update(activity="shop", plan=[{"activity": "shop", "start_h": 0, "end_h": 24}])  # stays, no arrival buy
    sim.state["tick"] = 60  # 10:00
    sim.state["market_food"] = 0
    sim.step()  # a is at the market as the tick ends, and it is empty
    assert a["market_seen"] == {"tick": 60, "food": 0}
    assert sim._market_belief(a) == " When you were at the market at 10:00, it had no food."
    view = sim._view(a, {})
    view["occasion"] = {"why": "hungry_coin", "who": None, "detail": ""}
    assert "it had no food." in sim._situation(a, dict(view), "deliberate")
    assert "it had no food." not in sim._situation(a, dict(view), "plan")  # deliberations only, as A/B'd (T16c)
    a["market_seen"] = {"tick": 70, "food": 4}  # seen stocked again later: the belief is gone
    sim.state["tick"] = 80
    assert sim._market_belief(a) == ""
    a["market_seen"] = {"tick": 70, "food": 0}
    sim.state["tick"] = TICKS_PER_DAY + 10  # a new day
    assert sim._market_belief(a) == ""


def test_diagnostics_count_both_engine_9_supply_failures():
    from asiimov.diagnostics import SUPPLY_FAILS
    assert {"no stock", "no coin", "no stock or no coin"} == SUPPLY_FAILS
