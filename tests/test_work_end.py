"""D2 (T26, plan/d2-work-end-spec.md): the end-of-work-block occasion, its restricted offer and receipts, the
plan shift behind a "work" answer, the stub's rule and the occasion-off control arm (`<arm>-work_end`)."""
import json

import pytest

from asiimov import scorecard as sc
from asiimov.events import read_jsonl, write_jsonl
from asiimov.minds import MindReply, StubMind
from asiimov.replay import replay
from asiimov.sim import DAILY_MIND_CALLS, DEFAULT_STEP_TICKS, DELIBERATE_ACTIONS, OCCASIONS, TICKS_PER_DAY, WORKPLACE, Sim

TH, _ = sc.load_thresholds()
PROTO = dict(TH["protocol"], agents=30, days=1)


class Scripted(StubMind):
    """Deliberates with `answer(req)` (a dict, or {"raw": text} for an unparseable reply; None: like the stub)."""

    def __init__(self, answer=None):
        self.answer, self.asked = answer, []

    def decide(self, req):
        if req.kind == "deliberate":
            self.asked.append(req)
            out = self.answer(req) if self.answer else None
            if out is not None:
                raw = out["raw"] if "raw" in out else json.dumps({"reason": "x", "goal": "", "then": [], **out})
                return MindReply(raw, "scripted", {}, "stop")
        return super().decide(req)


def by_role(sim, role):
    return next(a for a in sim.state["agents"] if a["role"] == role)


def alone_at_work(sim, a, plan, place=None):
    """Only `a` is awake, at its workplace (or `place`), with `plan`, bedtime far off, the mind free."""
    for o in sim.state["agents"]:
        o.update(asleep=True, wake_tick=10**6, sleep_tick=None)
    a.update(asleep=False, wake_tick=0, sleep_tick=10**6, plan=plan, plan_event=0, habit="home", activity="home",
             place=place or WORKPLACE[a["role"]], route=None, cooldown_until=0)
    return a


def run_ticks(sim, n):
    for _ in range(n):
        sim.step()


def asks(sim):
    return [e for e in sim.events if e.kind == "mind_deliberate" and e.data.get("occasion") == "work_end"]


def receipt_of(sim, ask):
    return next(e for e in sim.events if e.kind == "receipt" and e.parents == [ask.id])


WORK_THEN_SHOP = [{"activity": "work", "start_h": 0, "end_h": 1}, {"activity": "shop", "start_h": 1, "end_h": 2},
                  {"activity": "socialize", "start_h": 2, "end_h": 5}, {"activity": "home", "start_h": 5, "end_h": 23}]


def test_a_working_adult_is_asked_once_at_the_last_tick_of_a_planned_work_block():
    mind = Scripted(lambda req: {"action": "wait", "target": None})
    sim = Sim(1, 30, mind)
    a = alone_at_work(sim, by_role(sim, "crafter"), [dict(b) for b in WORK_THEN_SHOP])
    run_ticks(sim, 12)  # hours 0 and 1
    (ask,) = asks(sim)
    assert ask.tick == 5 and ask.parents == [0]  # the block's last tick, citing the plan that authored it
    req = mind.asked[0]
    assert req.view["offer"] == ["work", "wait"]
    assert ask.data["withheld"] == [x for x in DELIBERATE_ACTIONS if x not in ("work", "wait")]
    hours = sim._work_hours(a)
    assert (f"Why you stop to think: you have done the hours you planned at the workshop; another unit of work there"
            f" takes about {hours} hours and pays 6 coin; your plan says shop next. Answering work keeps you at the"
            f" workshop for about 3 more hours and moves the rest of your plan later; answering wait ends your work"
            f" for today and your plan goes on.") in req.prompt
    assert ("Your action happens now; target is null. work goes to your work for a few hours. wait does nothing for"
            " now. Leave goal empty and then []." in req.prompt)
    assert "To pursue something over hours or days" not in req.prompt.split("# Now")[-1]
    for word in ("conscientious", "diligent", "plan and your work", "hedon", "achievement"):
        assert word not in req.prompt.split("# Now")[-1]  # facts only in the situation (L11)
    r = receipt_of(sim, ask)
    assert (r.data["action"], r.data["ok"], r.data["why"]) == ("wait", True, "stopped as planned")
    assert a["override"] is None and a["plan"] == WORK_THEN_SHOP  # the plan goes on
    assert any(e.kind == "activity" and e.data["activity"] == "shop" and e.tick == 6 for e in sim.events)
    assert "stopped work when the planned block ended" in " ".join(a["recent"])


def test_work_goes_on_for_three_hours_and_the_shop_block_runs_after_it():
    mind = Scripted(lambda req: {"action": "work", "target": None})
    sim = Sim(1, 30, mind)
    a = alone_at_work(sim, by_role(sim, "crafter"), [dict(b) for b in WORK_THEN_SHOP])
    a["inv"].update(food=0, coin=30)
    run_ticks(sim, 6)
    (ask,) = asks(sim)
    r = receipt_of(sim, ask)
    assert r.data == {"action": "work", "ok": True, "why": "went on working", "plan_shift_h": 3}
    assert a["override"] == {"do": "work", "target": None, "event": ask.id, "left": DEFAULT_STEP_TICKS}
    assert a["plan"] == [{"activity": "work", "start_h": 0, "end_h": 1}, {"activity": "shop", "start_h": 4, "end_h": 5},
                         {"activity": "socialize", "start_h": 5, "end_h": 8}, {"activity": "home", "start_h": 8, "end_h": 24}]
    assert "worked on past the end of the planned block" in " ".join(a["recent"])
    n0 = len(sim.events)
    run_ticks(sim, 18)  # hours 1 to 3: the override
    assert a["activity"] == "work" and a["place"] == "workshop" and a["override"] is None  # done at the 18th tick
    assert not any(e.kind == "activity" for e in sim.events[n0:])  # never left the workshop
    assert sum(1 for e in sim.events[n0:] if e.kind == "produce") >= 1
    assert not asks(sim)[1:]  # once per block
    run_ticks(sim, 6)  # hour 4: the shifted shop block
    shop = [e for e in sim.events if e.kind == "activity" and e.data["activity"] == "shop"]
    assert shop and shop[0].tick == 24
    run_ticks(sim, 12)
    assert any(e.kind == "receipt" and e.data["action"] == "buy_food" and e.data["ok"] for e in sim.events)


def test_the_shift_clips_the_evening_at_midnight_and_leaves_the_block_before_untouched():
    mind = Scripted(lambda req: {"action": "work", "target": None})
    sim = Sim(1, 30, mind)
    a = alone_at_work(sim, by_role(sim, "trader"), [{"activity": "home", "start_h": 0, "end_h": 1},
                                                     {"activity": "work", "start_h": 1, "end_h": 2},
                                                     {"activity": "rest", "start_h": 2, "end_h": 22},
                                                     {"activity": "socialize", "start_h": 22, "end_h": 24}])
    run_ticks(sim, 12)
    (ask,) = asks(sim)
    assert ask.tick == 11
    assert a["plan"] == [{"activity": "home", "start_h": 0, "end_h": 1}, {"activity": "work", "start_h": 1, "end_h": 2},
                         {"activity": "rest", "start_h": 5, "end_h": 24}]  # the late block fell off the day


@pytest.mark.parametrize("case", ["child", "elder", "spent_cap", "intent", "bedtime", "cooldown", "cap", "away"])
def test_the_occasion_does_not_fire_when_the_spec_says_not(case):
    class Spent(Sim):
        def _food_yield(self, a):
            return 0

    sim = (Spent if case == "spent_cap" else Sim)(1, 40, Scripted(lambda req: {"action": "work", "target": None}))
    role = {"child": "child", "elder": "elder", "spent_cap": "farmer"}.get(case, "crafter")
    a = alone_at_work(sim, by_role(sim, role), [{"activity": "work", "start_h": 0, "end_h": 1},
                                                  {"activity": "home", "start_h": 1, "end_h": 23}])
    if case == "away":  # still on the way to the workshop at the block's last tick
        a.update(place=None, route={"from": [0, 0], "to": "workshop", "depart": 0, "arrive": 8, "event": 0})
    if case == "intent":
        a["intent"] = {"goal": "keep at it", "steps": [{"do": "work", "target": None}], "i": 0, "event": 0,
                       "tick": 0, "left": 40}
    if case == "bedtime":
        a["sleep_tick"] = 10  # 5 ticks after the block's last tick: under an hour
    if case == "cooldown":
        a["cooldown_until"] = 6
    if case == "cap":
        a["calls"] = {"day": 0, "n": DAILY_MIND_CALLS - 2}
    run_ticks(sim, 6)
    assert not asks(sim) and a["occasion"] is None
    if case == "bedtime":  # the boundary: a full hour left does ask
        sim2 = Sim(1, 40, Scripted(lambda req: {"action": "work", "target": None}))
        b = alone_at_work(sim2, by_role(sim2, "crafter"), [{"activity": "work", "start_h": 0, "end_h": 1}])
        b["sleep_tick"] = 11
        run_ticks(sim2, 6)
        assert len(asks(sim2)) == 1


def test_hunger_at_the_same_tick_wins_and_the_block_ends_as_planned():
    mind = Scripted(lambda req: {"action": "wait", "target": None})
    sim = Sim(1, 30, mind)
    a = alone_at_work(sim, by_role(sim, "crafter"), [dict(b) for b in WORK_THEN_SHOP])
    a["inv"].update(food=0, coin=30)
    a["needs"]["hunger"] = 1000
    run_ticks(sim, 6)
    asked = [e for e in sim.events if e.kind == "mind_deliberate"]
    assert len(asked) == 1 and asked[0].data["occasion"] == "hungry_coin" and not asks(sim)
    assert a["occasion"] is None and a["plan"] == WORK_THEN_SHOP
    run_ticks(sim, 12)  # the cooldown passes; the block's end is not asked late
    assert not asks(sim)


@pytest.mark.parametrize("reply,action", [({"action": "socialize", "target": None}, "socialize"),
                                          ({"raw": "not json at all"}, "None")])
def test_anything_else_counts_as_stopping_and_wakes_no_failed_occasion(reply, action):
    sim = Sim(1, 30, Scripted(lambda req: reply))
    a = alone_at_work(sim, by_role(sim, "crafter"), [dict(b) for b in WORK_THEN_SHOP])
    a["cognition"]["deliberation_threshold"] = 0  # any occasion would reach this mind
    run_ticks(sim, 6)
    (ask,) = asks(sim)
    r = receipt_of(sim, ask)
    assert (r.data["action"], r.data["ok"], r.data["why"]) == (action, False, "unparseable or unknown action")
    assert a["override"] is None and a["occasion"] is None and a["plan"] == WORK_THEN_SHOP


@pytest.mark.parametrize("action", ["work", "wait"])
def test_a_goal_and_steps_in_the_answer_are_ignored_so_nothing_displaces_the_plan(action):
    """PR #34 review: the offer limited the action, but goal and then still built an intent that took over the
    plan, shop block included. The schema pins them empty and the world ignores them here."""
    reply = {"action": action, "target": None, "goal": "a long evening", "then": [{"do": "socialize", "target": None},
                                                                                   {"do": "explore", "target": None}]}
    sim = Sim(1, 30, Scripted(lambda req: reply))
    a = alone_at_work(sim, by_role(sim, "crafter"), [dict(b) for b in WORK_THEN_SHOP])
    run_ticks(sim, 6)
    (ask,) = asks(sim)
    assert a["intent"] is None and not any(e.kind == "intent_set" for e in sim.events)
    r = receipt_of(sim, ask)
    if action == "work":
        assert r.data["why"] == "went on working" and a["override"]["do"] == "work" and a["plan"][1]["start_h"] == 4
    else:
        assert r.data["why"] == "stopped as planned" and a["override"] is None and a["plan"] == WORK_THEN_SHOP
    run_ticks(sim, 30)
    social = [e.tick for e in sim.events if e.kind == "activity" and e.data["activity"] == "socialize"]
    assert social and social[0] == (30 if action == "work" else 12)  # the plan's own block, shifted or not
    assert not any(e.kind == "activity" and e.data["activity"] == "explore" for e in sim.events)


def test_the_stub_answers_by_one_rule_on_its_own_params():
    def ask(c, coin):
        v = {"traits": {"conscientiousness": c}, "coin": coin, "occasion": {"why": "work_end"}}
        return StubMind()._deliberate(v)["action"]
    assert ask(700, 50) == "work" and ask(651, 0) == "work"
    assert ask(500, 5) == "work" and ask(350, 9) == "work"
    assert ask(500, 10) == "wait" and ask(650, 40) == "wait"
    assert ask(300, 0) == "wait" and ask(349, 5) == "wait"


def test_every_deliberation_records_its_occasion_and_a_stub_day_replays_bit_for_bit(tmp_path):
    sim = Sim(1, 40, StubMind())
    sim.run(1)
    sim.finish()
    delibs = [e for e in sim.events if e.kind == "mind_deliberate"]
    assert delibs and all(e.data["occasion"] in OCCASIONS for e in delibs)
    whys = {receipt_of(sim, e).data["why"] for e in asks(sim)}
    assert len(asks(sim)) >= 10 and whys <= {"went on working", "stopped as planned"} and len(whys) == 2
    path = tmp_path / "events.jsonl"
    write_jsonl(sim.events, str(path))
    r = replay(read_jsonl(str(path)))
    assert r.state_hash() == sim.state_hash() and len(r.events) == len(sim.events)


def test_the_stub_is_asked_on_most_working_adult_days():
    sim = Sim(1, 100, StubMind())
    sim.run(2)
    adults = [a["id"] for a in sim.state["agents"] if a["role"] not in ("child", "elder")]
    day1 = {e.actor for e in asks(sim) if e.tick // TICKS_PER_DAY == 1}
    assert len(day1) / len(adults) >= 0.6, len(day1) / len(adults)
    assert not {e.actor for e in asks(sim)} - set(adults)


def test_the_control_arm_turns_the_occasion_off_and_says_so_in_its_log():
    assert sc.parse_arm("full-work_end") == ("full", frozenset({"no_work_end"}))
    assert sc.parse_arm("no_text-anchors-work_end") == sc.parse_arm("no_text-work_end-anchors") == (
        "no_text", frozenset({"unanchored", "no_work_end"}))
    assert sc.configuration("full-work_end") == ("no_work_end", "full")
    assert sc.configuration("full-anchors-work_end") == ("unanchored+no_work_end", "full")
    for bad in ("full-work_end-work_end", "full-bogus", "full-work_end+framing", "-work_end"):
        with pytest.raises(ValueError):
            sc.parse_arm(bad)
    control = sc.run_arm("full-work_end", StubMind(), PROTO)
    full = sc.run_arm("full", StubMind(), PROTO)
    assert control[0].data["work_end"] is False and "work_end" not in full[0].data
    assert not [e for e in control if e.kind == "mind_deliberate" and e.data["occasion"] == "work_end"]
    assert [e for e in full if e.kind == "mind_deliberate" and e.data["occasion"] == "work_end"]
    # the same world up to the first ask: the control differs from the full arm only by what the asks change
    first = next(e.id for e in full if e.kind == "mind_deliberate" and e.data["occasion"] == "work_end")
    assert [e.to_json() for e in full[1:first]] == [e.to_json() for e in control[1:first]]
