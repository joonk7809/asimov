"""Per-agent memory files and prompt assembly.

Rule: anything code computes on lives in structured state; anything only the mind reads
is rendered as markdown. Files are rendered from state, so they are versioned by the
event log for free.
"""
import hashlib

SECTION_BUDGETS = {  # approximate tokens (chars / 4)
    "schema": 300, "soul": 400, "core": 300, "goals": 200, "relationships": 150,
    "beliefs": 150, "recent": 400, "episodes": 300, "situation": 300,
}

SCHEMA = """# How you act
You are one person living in a village. Reply with JSON only.
For kind=plan: {"plan":[{"activity":"work|socialize|home|rest|explore","start_h":H,"end_h":H}]}
For kind=deliberate: {"action":"buy_food|go_market|ask_kin|steal|wait","target":ID_or_null,"reason":"..."}
For kind=reflect: {"core":["..."] or [], "summary":"..."}
You only know what is written below. Claims count only if the world confirms them."""


def tokens(text: str) -> int:
    return (len(text) + 3) // 4


def _clip(text: str, budget_tokens: int) -> str:
    limit = budget_tokens * 4
    return text if len(text) <= limit else text[-limit:]


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


def render_files(agent: dict, state: dict) -> dict[str, str]:
    """Render the agent's markdown map from state."""
    names = {a["id"]: a["name"] for a in state["agents"]}
    rel_lines = [
        f"- {names[int(other)]} (#{other}): affinity {rel['affinity']}, {rel.get('note', '')}".rstrip(", ")
        for other, rel in sorted(agent["relationships"].items(), key=lambda kv: int(kv[0]))
    ]
    belief_lines = [f"- {k}: {v}" for k, v in sorted(agent["beliefs"].items())]
    return {
        "soul.md": agent["soul"],
        "core.md": "# Core memories\n" + "\n".join(f"- {c}" for c in agent["core"]),
        "goals.md": "# Today's plan\n" + "\n".join(
            f"- {b['activity']} {b['start_h']:02d}:00-{b['end_h']:02d}:00" for b in agent["plan"]),
        "relationships.md": "# People I know\n" + "\n".join(rel_lines),
        "beliefs.md": "# What I believe\n" + "\n".join(belief_lines),
        "recent.md": "# Today\n" + "\n".join(f"- {r}" for r in agent["recent"]),
        "episodes.md": "# Earlier\n" + "\n".join(f"- [{e['importance']}] {e['text']}" for e in agent["episodes"]),
    }


def assemble_prompt(agent: dict, state: dict, kind: str, situation: str) -> tuple[str, dict]:
    """Stable sections first (prefix-cacheable), changing tail last."""
    files = render_files(agent, state)
    top_episodes = sorted(agent["episodes"], key=lambda e: (-e["importance"], -e["tick"]))[:8]
    sections = {
        "schema": SCHEMA,
        "soul": files["soul.md"],
        "core": files["core.md"],
        "goals": files["goals.md"],
        "relationships": files["relationships.md"],
        "beliefs": files["beliefs.md"],
        "episodes": "# Earlier\n" + "\n".join(f"- {e['text']}" for e in top_episodes),
        "recent": files["recent.md"],
        "situation": f"# Now ({kind})\n{situation}",
    }
    parts = [_clip(text, SECTION_BUDGETS[name]) for name, text in sections.items()]
    prompt = "\n\n".join(parts)
    counts = {name: tokens(p) for name, p in zip(sections, parts)}
    return prompt, counts


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]
