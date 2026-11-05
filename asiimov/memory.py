"""Per-agent memory files and prompt assembly.

Rule: anything code computes on lives in structured state; anything only the mind reads
is rendered as markdown. Files are rendered from state, so they are versioned by the
event log for free.
"""
import hashlib
import re

from .params import VALUES

SECTION_BUDGETS = {  # approximate tokens (chars / 4)
    "schema": 300, "soul": 400, "core": 300, "goals": 200, "relationships": 150,
    "beliefs": 150, "recent": 400, "episodes": 300, "situation": 300,
    # A deliberation's situation also says why the mind was asked and what its actions do (engine 7).
    "situation_deliberate": 450,
}

SCHEMA = """# How you act
You are one person living in a village. Reply with JSON only.
For kind=plan: {"plan":[{"activity":"work|socialize|home|rest|explore|shop","start_h":H,"end_h":H}]}
For kind=deliberate: {"action":"buy_food|go_market|ask_kin|steal|give_food|give_coin|seek|avoid|work|socialize|home|rest|explore|wait","target":ID_or_null,"reason":"...","goal":"..." or "","then":[{"do":"...","target":ID_or_null}]}
For kind=reflect: {"core":["..."] or [], "summary":"..."}
You only know what is written below. Claims count only if the world confirms them."""


# Words that only name a value. A core memory made of nothing else ("power", "hedonism, power")
# is a label, not an experience: 92% of plain Qwen3-4B core entries were one (engine 3, t1e-4b).
VALUE_WORDS = frozenset(w for v in VALUES for w in v.split("_")) | {"and", "or"}
REFLECT_ASK = (" Your day is over. For core, write at most 2 things that happened to you today that you"
               " want to remember, each saying who did what and where, or [] if nothing stood out."
               " Not a value or a trait. Use only what is written under Today and Earlier; add no details"
               " that are not there.")


def is_bare_value(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and all(w in VALUE_WORDS for w in words)


def named_people(text: str, agents: list, self_id: int) -> list[set[int]]:
    """The people a text names, other than its author: one set of candidate ids per mention. A #id
    or full name is one person; a bare first name is everyone who has it."""
    low, groups = text.lower(), []
    for i in map(int, re.findall(r"#(\d+)", text)):
        if i != self_id:
            groups.append({i} if i < len(agents) else set())
    me = agents[self_id]["name"].lower()
    low = re.sub(rf"\b{re.escape(me)}\b", " ", low)  # the author naming themself
    # Longest first, ties by name: set order follows PYTHONHASHSEED, and the result feeds world state.
    for name in sorted({o["name"].lower() for o in agents} - {me}, key=lambda n: (-len(n), n)):
        full = re.compile(rf"\b{re.escape(name)}\b")
        if full.search(low):  # names are unique from engine 4; older worlds can repeat one
            groups.append({o["id"] for o in agents if o["name"].lower() == name})
            low = full.sub(" ", low)
    words = set(re.findall(r"[a-z]+", low)) - {me.split()[0]}  # a bare own first name is the author too
    for first in sorted({o["name"].split()[0].lower() for o in agents} & words):
        ids = {o["id"] for o in agents if o["name"].split()[0].lower() == first and o["id"] != self_id}
        if ids:
            groups.append(ids)
    return groups


def known_people(agent: dict, state: dict) -> set[int]:
    """Everyone the agent's own record mentions: kin, relationships, today's notes and episodes."""
    agents = state["agents"]
    known = set(state["families"][agent["family"]]["members"]) | {int(k) for k in agent["relationships"]}
    for text in list(agent["recent"]) + [e["text"] for e in agent["episodes"]]:
        known |= {o["id"] for o in agents if re.search(rf"\b{re.escape(o['name'])}\b", text)}
    return known


def tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _clip(text: str, budget_tokens: int) -> str:
    limit = budget_tokens * 4
    return text if len(text) <= limit else text[-limit:]


def _fit_count(header: str, lines: list[str], budget_tokens: int) -> int:
    size = len(header)
    for n, line in enumerate(lines):
        if size + 1 + len(line) > budget_tokens * 4:
            return n
        size += 1 + len(line)
    return len(lines)


def _fit_lines(header: str, lines: list[str], budget_tokens: int) -> str:
    """The header and as many whole lines, in the given order, as fit the budget. Clipping the tail of
    the text instead cut the header and every early line mid-word (71% of 4B probe prompts at engine 5)."""
    return "\n".join([header] + lines[:_fit_count(header, lines, budget_tokens)])


def situation_budget(kind: str) -> int:
    return SECTION_BUDGETS.get(f"situation_{kind}", SECTION_BUDGETS["situation"])


def rank_relationships(agent: dict) -> list[tuple[str, dict]]:
    """Strongest feeling first, then the most recent contact, then id: the people a mind keeps in view."""
    return sorted(agent["relationships"].items(),
                  key=lambda kv: (-abs(kv[1]["affinity"]), -kv[1].get("last", -1), int(kv[0])))


def _level(v: int, low: str, mid: str, high: str) -> str:
    return low if v < 350 else high if v > 650 else mid


def render_soul(agent: dict) -> str:
    t = agent["traits"]
    r = agent["rhythm"]
    wake = 390 + r["chronotype_min"]
    lines = [
        f"# {agent['name']}",
        f"{agent['age']} years old, {agent['role']}.",
        f"You wake around {wake // 60:02d}:{wake % 60:02d} and need about {r['sleep_need_min'] // 60} hours of sleep.",
        f"You are {_level(t['honesty'], 'willing to bend the truth when it pays', 'mostly straight with people', 'scrupulously honest')}, "
        f"{_level(t['extraversion'], 'happiest alone', 'sociable in small doses', 'restless without company')}, "
        f"{_level(t['agreeableness'], 'slow to forgive', 'fair-minded', 'quick to forgive')}, "
        f"{_level(t['conscientiousness'], 'loose with plans', 'steady', 'disciplined')}, "
        f"{_level(t['openness'], 'set in your ways', 'curious at times', 'drawn to anything new')}, and "
        f"{_level(t['emotionality'], 'hard to frighten', 'cautious', 'anxious about danger')}.",
        f"What matters most to you: {', '.join(v.replace('_', ' ') for v in agent['values']['top'])}.",
        f"What you care little for: {', '.join(v.replace('_', ' ') for v in agent['values']['bottom'])}.",
    ]
    return "\n".join(lines)


def _relationship_lines(agent: dict, state: dict) -> list[tuple[int, str]]:
    names = {a["id"]: a["name"] for a in state["agents"]}
    return [(int(other), f"- {names[int(other)]} (#{other}): affinity {rel['affinity']}, {rel.get('note', '')}".rstrip(", "))
            for other, rel in rank_relationships(agent)]


def shown_relationships(agent: dict, state: dict) -> list[int]:
    """The ids the prompt's "People I know" section shows (the lines that fit its budget)."""
    rel = _relationship_lines(agent, state)
    return [i for i, _ in rel[:_fit_count("# People I know", [line for _, line in rel], SECTION_BUDGETS["relationships"])]]


def render_intent(agent: dict, state: dict) -> str:
    """The agent's live intent: its goal, the step under way and the steps still to come. The reason
    the mind gave is not shown back (L12)."""
    it = agent.get("intent")
    if not it:
        return ""
    names = {a["id"]: a["name"] for a in state["agents"]}
    desc = [f"{s['do'].replace('_', ' ')}" + (f" {names[s['target']]} (#{s['target']})" if s["target"] is not None else "")
            for s in it["steps"][it["i"]:]]
    return "\n\n# My intentions\n" + f"- {it['goal']}: now {desc[0]}" + (f"; then {', '.join(desc[1:])}" if desc[1:] else "")


def render_files(agent: dict, state: dict) -> dict[str, str]:
    """Render the agent's markdown map from state."""
    rel_lines = [line for _, line in _relationship_lines(agent, state)]
    belief_lines = [f"- {k}: {v}" for k, v in sorted(agent["beliefs"].items())]
    return {
        "soul.md": agent["soul"],
        "core.md": "# Core memories\n" + "\n".join(f"- {c}" for c in agent["core"]),
        "goals.md": "# Today's plan\n" + "\n".join(
            f"- {b['activity']} {b['start_h']:02d}:00-{b['end_h']:02d}:00" for b in agent["plan"])
            + render_intent(agent, state),
        "relationships.md": "# People I know\n" + "\n".join(rel_lines),
        "beliefs.md": "# What I believe\n" + "\n".join(belief_lines),
        "recent.md": "# Today\n" + "\n".join(f"- {r}" for r in agent["recent"]),
        "episodes.md": "# Earlier\n" + "\n".join(f"- [{e['importance']}] {e['text']}" for e in agent["episodes"]),
    }


def assemble_prompt(agent: dict, state: dict, kind: str, situation: str, edit=None) -> tuple[str, dict]:
    """Stable sections first (prefix-cacheable), changing tail last.

    `edit(sections) -> sections` may drop or replace sections before clipping; the world never
    passes one, so its prompts are unchanged. asiimov.probe uses it for input ablations."""
    files = render_files(agent, state)
    top_episodes = sorted(agent["episodes"], key=lambda e: (-e["importance"], -e["tick"]))[:8]
    sections = {
        "schema": SCHEMA,
        "soul": files["soul.md"],
        "core": files["core.md"],
        "goals": files["goals.md"],
        "relationships": _fit_lines("# People I know", [x for x in files["relationships.md"].split("\n")[1:] if x],
                                    SECTION_BUDGETS["relationships"]),
        "beliefs": files["beliefs.md"],
        "episodes": "# Earlier\n" + "\n".join(f"- {e['text']}" for e in top_episodes),
        "recent": files["recent.md"],
        "situation": f"# Now ({kind})\n{situation}",
    }
    if edit is not None:
        sections = edit(dict(sections))
    budgets = {**SECTION_BUDGETS, "situation": situation_budget(kind)}
    parts = [_clip(text, budgets[name]) for name, text in sections.items()]
    prompt = "\n\n".join(parts)
    counts = {name: tokens(p) for name, p in zip(sections, parts)}
    return prompt, counts


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]
