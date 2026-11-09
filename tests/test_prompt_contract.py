"""Engine 6 prompt contract (T14a): economic facts in the situation, relationships clipped by whole
lines and ranked, crowds cut to fit. The reflection voice ask (c) was dropped: it failed steering's
bars (plan/t14a-signoff.md, results/raw/t14a-ab-4b-c2)."""
import re

from asiimov.bench import collect_prompts
from asiimov.memory import SCHEMA, SECTION_BUDGETS, assemble_prompt, rank_relationships, situation_budget
from asiimov.minds import StubMind
from asiimov.sim import FOOD_PRICE, WORK_WAGE, Sim


def _section(prompt: str, header: str) -> str:
    return prompt.split(header, 1)[1].split("\n\n", 1)[0] if header in prompt else ""


def test_every_prompt_states_what_work_pays_and_where_food_is_sold():
    reqs = collect_prompts(agents=30, days=2, seed=3)
    assert {r.kind for r in reqs} == {"plan", "deliberate", "reflect"}
    for r in reqs:
        now = r.prompt.split("# Now (", 1)[1]
        # the live price (B5 year-1): 3 to 6 by the stall's stock
        assert re.search(r"Food costs [3-6] coin and is sold only at the market, which is open at all hours\.", now)
        assert not re.search(r"\bclosed\b|opening hours|\bcloses\b", now)  # no prompt implies market hours (T15)
        assert ("pays" in now or "yields" in now or "don't work for pay" in now), now
    assert "coin" not in SCHEMA.split("For kind=")[0]  # the facts are not in the shared SCHEMA (L11)


def test_the_economy_line_matches_the_role_and_the_world():
    sim = Sim(1, 60, StubMind())
    by_role = {a["role"]: a for a in sim.state["agents"]}
    assert f"pays {WORK_WAGE} coin about every" in sim._economy(by_role["crafter"])
    assert "workshop" in sim._economy(by_role["crafter"])
    assert f"yields {sim._food_yield(by_role['farmer'])} food" in sim._economy(by_role["farmer"])  # the year's day
    assert f"sell at {sim._price() - 1} coin each. Food costs {sim._price()} coin" in sim._economy(by_role["farmer"])
    assert "don't work for pay" in sim._economy(by_role["child"])
    sim._food_yield = lambda a: 1  # an arm that changes the yield changes what the prompt says
    assert "yields 1 food" in sim._economy(by_role["farmer"])


def test_relationships_keep_the_header_and_whole_lines_strongest_first():
    sim = Sim(1, 100, StubMind())
    a = sim.state["agents"][50]
    for i in range(100):
        if i != 50:
            sim._rel(a, i)["affinity"] = 20
    sim._rel(a, 99)["affinity"] = -400  # a grudge against a high id
    sim.state["tick"] = 7
    sim._met(a, 98)  # the most recent contact among equals
    prompt, _ = assemble_prompt(a, sim.state, "plan", "x")
    sec = _section(prompt, "# People I know\n")
    lines = sec.split("\n")
    assert "# People I know\n" in prompt and len(sec) <= SECTION_BUDGETS["relationships"] * 4
    assert all(line.startswith("- ") and ": affinity " in line for line in lines)
    assert lines[0].startswith(f"- {sim.state['agents'][99]['name']} (#99)")
    assert lines[1].startswith(f"- {sim.state['agents'][98]['name']} (#98)")
    order = [int(k) for k, _ in rank_relationships(a)]
    assert order[2:5] == [0, 1, 2]  # then by id


def test_every_warm_relationship_carries_its_last_contact():
    sim = Sim(3, 30, StubMind())
    sim.run(1)
    rels = [r for a in sim.state["agents"] for r in a["relationships"].values()]
    assert rels and all(r.get("last", -1) >= 0 for r in rels if r["affinity"] > 0)



def test_a_crowd_is_cut_to_fit_and_the_schema_offers_only_the_people_shown():
    from collections import defaultdict
    from asiimov.llm import schema_for
    sim = Sim(1, 120, StubMind())
    sim.state["tick"] = 60
    for o in sim.state["agents"]:
        o["place"] = "market"
    occ = defaultdict(list)
    for o in sim.state["agents"]:
        occ[o["place"]].append(o["id"])
    a = sim.state["agents"][40]
    a["sleep_tick"] = 130
    kin = [k for k in sim.state["families"][a["family"]]["members"] if k != a["id"]]
    sim._rel(a, 119)["affinity"] = -500  # a grudge stays in view
    for kind in ("deliberate", "plan", "reflect"):
        sim.pending.clear()
        sim._request(a, kind, sim._view(a, occ), [], lambda *x: None)
        req = sim.pending[a["id"]][0]
        now = req.prompt.split(f"# Now ({kind})\n", 1)[1]
        assert now.startswith("Day 0, 10:00.") and "others." in now  # the head survives, the cut is said
        assert len(now) + len(f"# Now ({kind})\n") <= situation_budget(kind) * 4
        shown = {p["id"] for p in req.view["present"]}
        assert shown and all(f"#{i} (" in now for i in shown) and 119 in shown and set(kin) <= shown
        listed = now.split("Here: ", 1)[1].split(" and ", 1)[0]
        assert {int(x.split(" ")[0][1:]) for x in listed.split(", ")} == shown
        if kind == "deliberate":
            steal = next(b for b in schema_for(req)["anyOf"]
                         if b["properties"]["action"]["enum"] == ["steal"])
            assert set(steal["properties"]["target"]["enum"]) == shown
            seek = next(b for b in schema_for(req)["anyOf"] if b["properties"]["action"]["enum"] == ["seek"])
            assert set(seek["properties"]["target"]["enum"]) == set(req.view["people"])
            assert shown | set(kin) | {119} <= set(req.view["people"])
