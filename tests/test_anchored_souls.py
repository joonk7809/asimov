"""T25: the soul ends with the behavior anchors (plan/t25-anchored-souls-spec.md)."""
from asiimov import scorecard as sc
from asiimov.memory import SECTION_BUDGETS, anchors, render_soul, unanchored
from asiimov.minds import StubMind
from asiimov.params import TRAITS

# scorecard.anchors as it was before T25, and T24b's openness rewording (results/tools/t24b_ab.py,
# OPEN_V2): the canonical block must be exactly what T24b's A/B showed the minds.
OLD = {"honesty": ("If a lie or taking something gets you what you need, you do it without much guilt.",
                   "You keep your word most of the time, but hunger or fear can bend you.",
                   "You would rather go without than lie or take what isn't yours."),
       "agreeableness": ("You hold grudges and make people pay for slights.",
                         "You treat people as they treat you.", "You let things go and help even those who wronged you."),
       "extraversion": ("You avoid company unless you need something.",
                        "You enjoy a little company, then want quiet.", "You seek people out whenever you can."),
       "conscientiousness": ("You drop plans when something better comes up.",
                             "You mostly do what you planned.", "You stick to your plan and your work."),
       "openness": ("You keep to familiar places and routines.",
                    "You try something new now and then.", "You go looking for new places and people."),
       "emotionality": ("Danger and scarcity barely worry you.",
                        "Scarcity makes you careful.", "Scarcity frightens you and you act to protect yourself first.")}
OPEN_V2 = {"- You keep to familiar places and routines.":
           "- You keep to familiar places and routines, and you don't go wandering off to explore.",
           "- You go looking for new places and people.":
           "- You go looking for new places and people, and you like to explore."}


def t24b_anchors_v2(agent):
    t = agent["traits"]
    band = lambda v: 0 if v < 350 else 2 if v > 650 else 1  # noqa: E731
    text = "\n".join(["# How you tend to act"] + ["- " + OLD[k][band(t[k])] for k in OLD])
    for old, new in OPEN_V2.items():
        if "\n" + old + "\n" in text:
            text = text.replace("\n" + old + "\n", "\n" + new + "\n", 1)
    return text


def test_the_block_is_t24b_anchors_v2_byte_for_byte_for_every_agent():
    for seed in (1, 2, 3):
        for a in sc.labeled_population(seed, 100)["agents"]:
            soul = render_soul(a)
            assert anchors(a) == t24b_anchors_v2(a)
            assert soul.endswith("\n\n" + anchors(a)) and soul.count("# How you tend to act") == 1
            assert unanchored(soul) + "\n\n" + anchors(a) == soul
    for tr in TRAITS:  # every band of every trait, so no line is only reached by luck of the population
        for v in (0, 500, 1000):
            a = {"traits": {k: 500 for k in TRAITS} | {tr: v}}
            assert anchors(a) == t24b_anchors_v2(a)


def test_every_soul_fits_its_budget_uncut():
    """The soul section is clipped to its budget (memory._clip); an anchored soul must fit whole."""
    sizes = [len(render_soul(a)) for seed in range(1, 6) for a in sc.labeled_population(seed, 100)["agents"]]
    assert max(sizes) <= SECTION_BUDGETS["soul"] * 4, max(sizes)


def test_the_fiction_arm_soul_holds_the_block_once_and_ablations_keep_their_meaning():
    seen = {}

    class Spy(StubMind):
        def decide(self, req):
            seen.setdefault(self.arm, {}).setdefault(req.agent_id, req.prompt)
            return super().decide(req)
    pop = sc.labeled_population(1, 30)
    souls = {a["id"]: render_soul(a) for a in pop["agents"]}
    for arm in ("fiction", "no_text", "flat_params"):
        spy = Spy()
        spy.arm = arm
        genesis = sc.flatten_params(pop) if arm == "flat_params" else pop
        sc.ArmSim(1, 30, spy, genesis, arm, souls).run(1)
    flat = sc.flatten_params(pop)["agents"]
    for a in pop["agents"]:
        assert seen["fiction"][a["id"]].count("# How you tend to act") == 1
        assert "# How you tend to act" not in seen["no_text"][a["id"]]
        assert anchors(a) in seen["flat_params"][a["id"]]  # the real params' anchors, not the median's
    assert any(anchors(a) != anchors(f) for a, f in zip(pop["agents"], flat))


def test_the_judges_sheet_lines_hold_no_anchor():
    """Review of PR #33: test 11's judge sees the three sheet lines only, never the anchors block."""
    from asiimov.memory import ANCHORS, ANCHORS_HEADING, sheet_lines
    from asiimov.sim import Sim
    for seed in (1, 2, 3):
        for a in Sim(seed, 100, StubMind()).state["agents"]:
            s = sheet_lines(a)
            lines = s.split("\n")
            assert len(lines) == 3 and lines[0].startswith("You are ")
            assert lines[1].startswith("What matters most") and lines[2].startswith("What you care little")
            assert ANCHORS_HEADING not in s
            assert not any(x in s for band in ANCHORS.values() for x in band)
