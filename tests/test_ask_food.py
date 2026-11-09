"""Engine 10 (T19, plan/t19-ask-food-spec.md in the shared folder): a deliberation offers only actions that
can work and says why the rest are withheld, and anyone here can be asked for food or sold it."""
import json

from asiimov.memory import E9_ACTIONS, SCHEMA
from asiimov.minds import MindReply, StubMind
from asiimov.sim import (ACTION_CLAUSES, FOOD_PRICE, action_clauses, NO_COIN_LINE, OCCASION_TICKS, SHARE_BASE, SHARE_KIN,
                         TICKS_PER_DAY, Sim)
from test_intents import Scripted, awake, deliberate_now


def deliberation(sim, a, why="hungry_coin"):
    """File a deliberation for `a` as _maybe_deliberate would, and return its request."""
    view = sim._view(a, {a["place"]: [o["id"] for o in sim.state["agents"] if o["place"] == a["place"]]})
    view["occasion"] = {"why": why, "who": None, "detail": ""}
    sim._request(a, "deliberate", view, [], lambda *x: None)
    req = sim.pending.pop(a["id"])[0]
    return req


def offered_line(prompt):
    line = next(x for x in prompt.splitlines() if x.startswith("For kind=deliberate"))
    return line.split('"action":"')[1].split('"')[0].split("|")


def test_only_actions_that_can_work_are_offered_and_the_prompt_says_why():
    sim = Sim(1, 12, StubMind())
    a, = awake(sim, [0])
    a["inv"].update(coin=0, food=0)
    req = deliberation(sim, a)
    off = req.view["offer"]
    assert not {"buy_food", "go_market", "buy_from"} & set(off) and "ask_food" in off and "steal" in off
    assert offered_line(req.prompt) == off  # the SCHEMA line lists exactly the offer
    assert NO_COIN_LINE in req.prompt
    for x, clause in action_clauses(sim._price()).items():  # the live price (B5 year-1)
        assert (clause in req.prompt) == (x in off)
    assert "(these words)" in req.prompt and "or shop" not in req.prompt

    a["inv"]["coin"] = 20  # coin, away from the market, no belief: go there, but buy_food only at the stall
    req = deliberation(sim, a)
    assert {"go_market", "buy_from"} <= set(req.view["offer"]) and "buy_food" not in req.view["offer"]
    assert NO_COIN_LINE not in req.prompt and "(these words, or shop)" in req.prompt

    sim.state["tick"] = 60
    a["market_seen"] = {"tick": 50, "food": 0}  # today's belief: it was empty
    req = deliberation(sim, a)
    assert "go_market" not in req.view["offer"] and "it had no food." in req.prompt

    a.update(place="market", market_seen=None)
    sim.state["market_food"] = 0  # at the stall, which shows nothing
    req = deliberation(sim, a)
    assert not {"go_market", "buy_food"} & set(req.view["offer"]) and "The market has 0 food" in req.prompt
    sim.state["market_food"] = 5
    req = deliberation(sim, a)
    assert {"go_market", "buy_food"} <= set(req.view["offer"])


def test_plan_and_reflect_prompts_keep_the_engine_9_schema():
    sim = Sim(1, 12, StubMind())
    sim.run(1)
    for kind in ("plan", "reflect"):
        a = sim.state["agents"][0]
        sim._request(a, kind, sim._view(a, {}), [], lambda *x: None)
        req = sim.pending.pop(a["id"])[0]
        assert req.prompt.startswith(SCHEMA) and f'"action":"{E9_ACTIONS}"' in req.prompt


def test_a_deliberation_logs_what_it_withheld():
    sim = Sim(1, 30, StubMind())
    sim.run(2)
    ds = [e for e in sim.events if e.kind == "mind_deliberate"]
    assert ds and all(isinstance(e.data["withheld"], list) for e in ds)
    assert any("go_market" in e.data["withheld"] for e in ds) and any(not e.data["withheld"] or
                                                                        "buy_food" in e.data["withheld"] for e in ds)


def share_world(roll_under):
    """Two strangers at the plaza; b carries food. The seeded draw is forced above or below the limit."""
    sim = Sim(1, 12, StubMind())
    fam = sim.state["families"][sim.state["agents"][0]["family"]]["members"]
    other = next(i for i in range(12) if i not in fam)
    a, b = awake(sim, [0, other], place="plaza")
    a["inv"].update(food=0, coin=10)
    b["inv"].update(food=3, coin=0)
    b["traits"]["agreeableness"] = 600
    for x in (a, b):  # both stay at the plaza
        x.update(activity="socialize", plan=[{"activity": "socialize", "start_h": 0, "end_h": 24}])
    sim.rng.draw = lambda stream, lo, hi: (0, 0 if roll_under else 999)
    sim.rng.at = lambda stream, index, lo, hi: 0 if roll_under else 999  # the keyed form (engine 11)
    return sim, a, b


def test_ask_food_and_buy_from_move_food_and_coin_on_a_yes():
    sim, a, b = share_world(True)
    deliberate_now(sim, a, {"action": "ask_food", "target": b["id"]})
    r = [e for e in sim.events if e.kind == "receipt"][-1]
    assert r.data["ok"] and r.data["why"] == "gave" and r.origin == "chance" and r.data["to"] == b["id"]
    assert r.data["limit"] == SHARE_BASE["ask_food"] + 600 // 4 and r.data["kin"] is False
    assert (a["inv"]["food"], b["inv"]["food"]) == (1, 2) and a["inv"]["coin"] == 10
    deliberate_now(sim, a, {"action": "buy_from", "target": b["id"]})
    r = [e for e in sim.events if e.kind == "receipt"][-1]
    assert r.data["ok"] and r.data["why"] == "sold" and r.data["limit"] > SHARE_BASE["ask_food"] + 600 // 4
    assert (a["inv"]["food"], b["inv"]["food"]) == (2, 1)
    p = sim._price()  # the live price (B5 year-1)
    assert (a["inv"]["coin"], b["inv"]["coin"]) == (10 - p, p) and f"for {p} coin" in a["recent"][-1]
    assert any("bought 1 food from" in n for n in a["recent"]) and any("sold 1 food to" in n for n in b["recent"])
    deliberate_now(sim, a, {"action": "ask_food", "target": b["id"]})  # b is down to 1: nothing to spare, no draw
    r = [e for e in sim.events if e.kind == "receipt"][-1]
    assert not r.data["ok"] and r.data["why"] == "nothing to spare" and "roll" not in r.data


def test_family_and_liked_people_say_yes_more_often():
    sim, a, b = share_world(True)
    kin = next(i for i in sim.state["families"][a["family"]]["members"] if i != a["id"])
    k = sim.state["agents"][kin]
    k.update(asleep=False, place="plaza", sleep_tick=10**6)
    k["inv"]["food"], k["traits"]["agreeableness"] = 3, 600
    deliberate_now(sim, a, {"action": "ask_food", "target": kin})  # family: ask_kin's odds, whichever word
    assert [e for e in sim.events if e.kind == "receipt"][-1].data["limit"] == 400 + 600 // 2
    k["inv"]["food"] = 3
    deliberate_now(sim, a, {"action": "buy_from", "target": kin})
    assert [e for e in sim.events if e.kind == "receipt"][-1].data["limit"] == SHARE_BASE["buy_from"] + 150 + SHARE_KIN
    sim._rel(b, a["id"])["affinity"] = 400
    deliberate_now(sim, a, {"action": "ask_food", "target": b["id"]})
    assert [e for e in sim.events if e.kind == "receipt"][-1].data["limit"] == SHARE_BASE["ask_food"] + 150 + 100


def test_a_refusal_reaches_the_mind_after_its_cooldown_even_when_hungry():
    seen = []

    def answer(req):
        seen.append(req.view["occasion"]["why"])
        return {"action": "ask_food", "target": b["id"]} if len(seen) == 1 else {"action": "wait", "target": None}
    sim, a, b = share_world(False)
    sim.mind = Scripted(answer)
    a["cognition"]["deliberation_threshold"] = 1000  # a refusal still gets through
    a["needs"]["hunger"] = 900
    for _ in range(OCCASION_TICKS + 2):
        sim.step()
    r = next(e for e in sim.events if e.kind == "receipt" and e.data["action"] == "ask_food")
    assert not r.data["ok"] and r.data["why"] == "refused"
    assert seen[:2] == ["hungry_coin", "refused"]  # not hunger again: the refusal is why it stops to think
    assert any("refused to give me food" in n for n in a["recent"])
    assert not any(e.kind == "receipt" and e.data["why"] == "refused" and e.actor != a["id"] for e in sim.events)


def test_ask_food_as_a_step_acts_where_the_agent_stands():
    sim, a, b = share_world(True)
    deliberate_now(sim, a, {"action": "wait", "target": None, "goal": "eat",
                            "then": [{"do": "buy_from", "target": b["id"]}]})
    for _ in range(2):
        sim.step()
    assert any(e.kind == "receipt" and e.data["action"] == "buy_from" and e.data["ok"] for e in sim.events)
    b["place"] = "woods"
    deliberate_now(sim, a, {"action": "ask_food", "target": b["id"]})
    assert [e for e in sim.events if e.kind == "receipt"][-1].data["why"] == "target not here"


def test_the_prompt_the_reply_schema_and_withheld_agree_on_what_is_offered():
    """A person action with nobody to name is not offered (PR #22 review: 7.6% of famine deliberations listed
    steal, ask_food, buy_from or ask_kin while the reply schema dropped them)."""
    from asiimov.bench import collect_prompts
    from asiimov.llm import schema_for
    # seed 7 from pay-1: seed 3's 41 deliberations there all have someone present
    reqs = [r for r in collect_prompts(agents=30, days=2, seed=7) if r.kind == "deliberate"]
    assert any(not r.view["present"] for r in reqs)
    for req in reqs:
        grammar = {x for b in schema_for(req)["anyOf"] for x in b["properties"]["action"]["enum"]}
        assert set(req.view["offer"]) == grammar
        assert offered_line(req.prompt) == req.view["offer"]
        if not req.view["present"]:
            assert not {"steal", "ask_food", "buy_from"} & grammar


def test_a_no_moves_an_intent_on_unless_the_ask_was_its_last_step():
    sim, a, b = share_world(False)  # every draw says no
    sim.mind = Scripted(lambda req: {"action": "wait", "target": None})
    deliberate_now(sim, a, {"action": "wait", "target": None, "goal": "find food",
                            "then": [{"do": "ask_food", "target": b["id"]}, {"do": "rest", "target": None}]})
    sim.step()
    it = a["intent"]
    assert it is not None and it["i"] == 1 and it["steps"][1]["do"] == "rest"
    step = [e for e in sim.events if e.kind == "intent_step"][-1]
    assert step.data == {"i": 0, "do": "ask_food", "why": "refused"}
    assert sim.mind.asked[-1].view["occasion"]["why"] == "refused"  # the mind still hears of it and may drop it
    b["inv"]["food"] = 1  # nothing to spare, as the last step: the intent fails as before
    deliberate_now(sim, a, {"action": "wait", "target": None, "goal": "find food",
                            "then": [{"do": "buy_from", "target": b["id"]}]})
    sim.step()
    assert a["intent"] is None
    failed = [e for e in sim.events if e.kind == "intent_failed"][-1]
    assert failed.data["why"] == "nothing to spare"


def test_a_caught_steal_still_ends_the_intent():
    sim, a, b = share_world(False)
    sim.rng.draw = lambda stream, lo, hi: (0, 999)  # the steal fails and is noticed
    deliberate_now(sim, a, {"action": "wait", "target": None, "goal": "eat",
                            "then": [{"do": "steal", "target": b["id"]}, {"do": "rest", "target": None}]})
    sim.step()
    assert a["intent"] is None and [e for e in sim.events if e.kind == "intent_failed"][-1].data["why"] == "caught"
