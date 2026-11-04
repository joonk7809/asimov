"""Engine 4 memory contract: a new day's plan prompt shows no old plan, core memories are
experiences rather than value labels, and the copy diagnostics measure both (lessons L11, L12)."""
import json

from asiimov.bench import collect_prompts
from asiimov.diagnostics import diagnose
from asiimov.events import Event
from asiimov.memory import REFLECT_ASK, is_bare_value
from asiimov.minds import StubMind
from asiimov.sim import Sim


def _goals(prompt: str) -> str:
    return prompt.split("# Today's plan\n", 1)[1].split("\n\n", 1)[0]


def test_wake_prompt_shows_no_plan_from_an_earlier_day():
    reqs = collect_prompts(agents=20, days=3, seed=3)
    plans = [r for r in reqs if r.kind == "plan"]
    assert len({r.tick // 144 for r in plans}) == 3  # later days are covered, not just the first
    assert all(_goals(r.prompt).strip() == "" for r in plans)
    # During the day the plan is still shown: deliberations and the evening reflection see it.
    assert any(_goals(r.prompt).strip() for r in reqs if r.kind == "reflect")


def test_reflect_ask_only_in_reflect_prompts():
    reqs = collect_prompts(agents=20, days=2, seed=3)
    assert all((REFLECT_ASK in r.prompt) == (r.kind == "reflect") for r in reqs)


def test_bare_value_words():
    for text in ("power", "Hedonism, power.", "self_direction", "Self-direction and achievement"):
        assert is_bare_value(text), text
    for text in ("Ana (#12) refused me food at the market", "power struggle with Bo", "stole bread", "42", "  "):
        assert not is_bare_value(text), text


def test_bare_value_core_entries_are_rejected_and_counted():
    sim = Sim(3, 10, StubMind())
    a = sim.state["agents"][0]
    a["plan"] = [{"activity": "work", "start_h": 8, "end_h": 12}]
    sim._fall_asleep(a, 0, {"core": ["power, hedonism", "Bo (#3) gave me bread at the plaza"], "summary": ""})
    assert a["core"] == ["Bo (#3) gave me bread at the plaza"]
    sleep = next(e for e in reversed(sim.events) if e.kind == "sleep")
    assert sleep.data == {"promoted": 1, "rejected": 1}
    assert a["plan"] == []


def _genesis(names):
    return Event(0, 0, "genesis", "rule", None, {"population": {"agents": [{"name": n} for n in names]}}, [])


def _mind(i, actor, kind, out):
    return Event(i, i, f"mind_{kind}", "mind", actor, {"kind": kind}, [], {"raw_output": json.dumps(out)})


def test_diagnostics_on_a_log_with_known_answers():
    day1 = {"plan": [{"activity": "work", "start_h": 8, "end_h": 16}, {"activity": "home", "start_h": 16, "end_h": 22}]}
    day2 = {"plan": [{"activity": "explore", "start_h": 8, "end_h": 22}]}
    evs = [_genesis(["Bo Kaorri", "Ana Nelen"]), _mind(0, 0, "plan", day1), _mind(1, 1, "plan", day2),   # first plans: not counted
           _mind(2, 0, "plan", day1), _mind(3, 1, "plan", day1),   # agent 0 copies, agent 1 doesn't
           _mind(4, 0, "reflect", {"core": ["power", "Bo stole my bread at the market"], "summary": ""}),
           _mind(5, 1, "reflect", {"core": ["", "hedonism"], "summary": ""}),
           _mind(6, 1, "reflect", {"core": ["hedonism is what drives me", "#4 helped me"], "summary": ""})]
    assert diagnose(evs) == {"plans_after_first": 2, "plan_copy_rate": 0.5, "core_entries": 5,
                             "value_word_share": 0.4, "grounded_share": 0.4}
