"""Engine 4 memory contract: a new day's plan prompt shows no old plan, core memories are
experiences rather than value labels, and the copy diagnostics measure both (lessons L11, L12)."""
import json

from asiimov.bench import collect_prompts
from asiimov.diagnostics import diagnose
from asiimov.events import Event
from asiimov.memory import REFLECT_ASK, is_bare_value, named_people
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
    assert "add no details" in REFLECT_ASK
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
    friend = sim.state["agents"][5]
    a["recent"] = [f"09:00 talked with {friend['name']} at the plaza"]
    a["plan"] = [{"activity": "work", "start_h": 8, "end_h": 12}]
    sim._fall_asleep(a, 0, {"core": ["power, hedonism", f"{friend['name']} told me about the plaza"], "summary": ""})
    assert a["core"] == [f"{friend['name']} told me about the plaza"]
    sleep = next(e for e in reversed(sim.events) if e.kind == "sleep")
    assert sleep.data == {"promoted": 1, "rejected": 1, "unknown_person": 0}
    assert a["plan"] == []


def test_core_entries_naming_someone_outside_the_record_are_rejected():
    """F20: names a person the agent never met, or a #id that doesn't exist, is invented."""
    sim = Sim(3, 30, StubMind())
    agents = sim.state["agents"]
    a = agents[0]
    kin = next(m for m in sim.state["families"][a["family"]]["members"] if m != a["id"])
    stranger = next(o for o in agents if o["id"] not in sim.state["families"][a["family"]]["members"]
                    and str(o["id"]) not in a["relationships"]
                    and sum(p["name"].split()[0] == o["name"].split()[0] for p in agents) == 1)
    first = stranger["name"].split()[0]
    cases = {f"{stranger['name']} showed me a map at the woods": 1, f"{first} waved at me in the market": 1,
             f"#{stranger['id']} sold me bread": 1, "#999 sold me bread": 1,
             f"{agents[kin]['name']} gave me food at home": 0, f"#{kin} and I ate at home": 0,
             f"{a['name']} rested at home all day": 0,
             f"{agents[kin]['name']} talked with {a['name'].split()[0]} at the plaza": 0}
    for text, unknown in cases.items():
        a["asleep"] = False
        sim._fall_asleep(a, 0, {"core": [text], "summary": ""})
        sleep = next(e for e in reversed(sim.events) if e.kind == "sleep")
        assert sleep.data["unknown_person"] == unknown, text
        assert (a["core"][-1:] == [text]) == (not unknown), text
    # A longer invented name isn't read as the real full name it contains, only as its first name.
    first_of_kin = agents[kin]["name"].split()[0]
    same_first = {o["id"] for o in agents if o["name"].split()[0] == first_of_kin and o["id"] != a["id"]}
    assert named_people(f"{agents[kin]['name']}son lent me a cart", agents, a["id"]) == [same_first]


def _genesis(names):
    agents = [{"id": i, "name": n, "role": "farmer"} for i, n in enumerate(names)]
    return Event(0, 0, "genesis", "rule", None, {"population": {"agents": agents}}, [])


def _mind(i, actor, kind, out):
    return Event(i, i, f"mind_{kind}", "mind", actor, {"kind": kind}, [], {"raw_output": json.dumps(out)})


def test_diagnostics_on_a_log_with_known_answers():
    day1 = {"plan": [{"activity": "work", "start_h": 8, "end_h": 16}, {"activity": "home", "start_h": 16, "end_h": 22}]}
    day2 = {"plan": [{"activity": "explore", "start_h": 8, "end_h": 22}]}
    evs = [_genesis(["Bo Kaorri", "Ana Nelen"]), _mind(0, 0, "plan", day1), _mind(1, 1, "plan", day2),   # first plans: not counted
           _mind(2, 0, "plan", day1), _mind(3, 1, "plan", day1),   # agent 0 copies, agent 1 doesn't
           _mind(4, 0, "reflect", {"core": ["power", "Ana stole my bread at the market"], "summary": ""}),
           _mind(5, 1, "reflect", {"core": ["", "hedonism"], "summary": ""}),
           _mind(6, 1, "reflect", {"core": ["hedonism is what drives me", "#4 helped me"], "summary": ""}),
           _mind(7, 1, "reflect", {"core": ["Ana Nelen rested at home", "#1 walked to the woods"], "summary": ""})]
    # Agent 1 is Ana Nelen: naming herself or her own #1 is not naming someone.
    d = diagnose(evs)
    assert {k: d[k] for k in ("plans_after_first", "plan_copy_rate", "core_entries", "value_word_share",
                              "person_share", "place_share")} == {
        "plans_after_first": 2, "plan_copy_rate": 0.5, "core_entries": 7,
        "value_word_share": 0.286, "person_share": 0.286, "place_share": 0.429}


def test_full_names_are_unique_per_world():
    """The memory guard and every name the minds read resolve a full name to one person."""
    from asiimov.params import generate_population
    from asiimov.scorecard import labeled_population
    for seed in range(1, 21):
        for pop in (generate_population(seed, 100), labeled_population(seed, 100)):
            names = [a["name"] for a in pop["agents"]]
            assert len(set(names)) == len(names), seed


def test_author_naming_themself_is_not_someone_else():
    """Two Annas: Anna Dean writing her own full or first name names nobody (the reflect A/B's 54
    false 'unknown' entries were all this)."""
    agents = [{"id": 0, "name": "Anna Dean"}, {"id": 1, "name": "Anna Vosavo"}, {"id": 2, "name": "Omar Luriro"}]
    assert named_people("Omar Luriro and Anna Dean talked at the plaza", agents, 0) == [{2}]
    assert named_people("Omar Luriro talked with Anna at the woods", agents, 0) == [{2}]
    assert named_people("Anna Vosavo talked with me", agents, 0) == [{1}]
    assert named_people("Anna waved at Omar Luriro", agents, 2) == [{0, 1}]


def test_named_people_does_not_depend_on_hash_seed():
    """Overlapping names ("Anna Dean" + "Dean Kofi") must resolve the same in every process: the
    result decides which memories are kept, so it is world state."""
    import os
    import subprocess
    import sys
    code = ("from asiimov.memory import named_people;"
            "a=[{'id':0,'name':'Anna Dean'},{'id':1,'name':'Dean Kofi'},{'id':2,'name':'Mia Orne'},"
            "{'id':3,'name':'Lia Orne'},{'id':4,'name':'Mia Lenro'},{'id':5,'name':'Lia Lenro'}];"
            "print([sorted(g) for g in named_people('talked with Anna Dean Kofi, then Mia and Lia', a, 3)])")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(seed)}).stdout for seed in range(1, 9)}
    assert len(outs) == 1, outs


def test_food_diagnostic_counts_meals_by_role_and_food_by_day():
    from asiimov.diagnostics import food
    g = Event(0, 0, "genesis", "rule", None, {"population": {"agents": [
        {"id": 0, "name": "A", "role": "farmer"}, {"id": 1, "name": "B", "role": "crafter"},
        {"id": 2, "name": "C", "role": "crafter"}]}}, [])
    ev = [g, Event(1, 10, "produce", "rule", 0, {"food": 3}, []), Event(2, 10, "sell", "rule", 0, {"food": 2}, []),
          Event(3, 20, "eat", "rule", 0, {}, []), Event(4, 150, "eat", "rule", 1, {}, []),
          Event(5, 160, "eat", "rule", 2, {"household": 0}, []),
          Event(6, 170, "mind_deliberate", "mind", 1, {"kind": "deliberate"}, []),
          Event(7, 171, "receipt", "rule", 1, {"action": "buy_food", "ok": True, "why": "bought", "food": 2}, []),
          Event(8, 180, "receipt", "rule", 2, {"action": "buy_food", "ok": False, "why": "no stock or no coin"}, []),
          Event(9, 288, "end", "rule", None, {}, [])]
    day0 = {"produced": 3, "sold": 2, "bought": 0, "eaten": 1, "from_household": 0, "deliberations": 0, "spoiled": 0}
    day1 = {"produced": 0, "sold": 0, "bought": 2, "eaten": 2, "from_household": 1, "deliberations": 1, "spoiled": 0}
    assert food(ev) == {"meals_per_agent_day_by_role": {"crafter": 0.5, "farmer": 0.5},
                        "buy_attempts": 2, "buy_failed_share": 0.5,
                        "buy_failed_by_why": {"no stock or no coin": 1}, "food_by_day": [day0, day1]}
