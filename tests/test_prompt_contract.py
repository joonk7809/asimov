"""Engine 6 prompt contract (T14a): economic facts in the situation, relationships clipped by whole
lines and ranked, the reflection summary asked in the agent's own voice."""
from asiimov.bench import collect_prompts
from asiimov.memory import REFLECT_VOICE, SCHEMA, SECTION_BUDGETS, assemble_prompt, rank_relationships
from asiimov.minds import StubMind
from asiimov.sim import FOOD_PRICE, WORK_WAGE, Sim


def _section(prompt: str, header: str) -> str:
    return prompt.split(header, 1)[1].split("\n\n", 1)[0] if header in prompt else ""


def test_every_prompt_states_what_work_pays_and_where_food_is_sold():
    reqs = collect_prompts(agents=30, days=2, seed=3)
    assert {r.kind for r in reqs} == {"plan", "deliberate", "reflect"}
    for r in reqs:
        now = r.prompt.split("# Now (", 1)[1]
        assert f"Food costs {FOOD_PRICE} coin and is sold only at the market." in now
        assert ("pays" in now or "yields" in now or "don't work for pay" in now), now
    assert "coin" not in SCHEMA.split("For kind=")[0]  # the facts are not in the shared SCHEMA (L11)


def test_the_economy_line_matches_the_role_and_the_world():
    sim = Sim(1, 60, StubMind())
    by_role = {a["role"]: a for a in sim.state["agents"]}
    assert f"pays {WORK_WAGE} coin about every" in sim._economy(by_role["crafter"])
    assert "workshop" in sim._economy(by_role["crafter"])
    assert "yields 3 food" in sim._economy(by_role["farmer"])
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


def test_the_summary_voice_ask_is_only_in_reflect_prompts():
    reqs = collect_prompts(agents=20, days=2, seed=3)
    assert all((REFLECT_VOICE in r.prompt) == (r.kind == "reflect") for r in reqs)
