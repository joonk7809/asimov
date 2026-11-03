"""Run metrics: cost, provenance, consequence depth and the variety tests."""
from collections import Counter, defaultdict

from .events import Event
from .sim import TICKS_PER_DAY


def icc1(groups: dict[int, list[float]], min_k: int = 2) -> float:
    """One-way ICC(1): share of variance that is between agents (character) vs within (noise).
    Uses agents with at least min_k observations, truncated to a common k."""
    groups = {k: v for k, v in groups.items() if len(v) >= max(2, min_k)}
    if len(groups) < 2:
        return float("nan")
    k = min(len(v) for v in groups.values())
    data = {g: v[:k] for g, v in groups.items()}
    n = len(data)
    grand = sum(sum(v) for v in data.values()) / (n * k)
    msb = k * sum((sum(v) / k - grand) ** 2 for v in data.values()) / (n - 1)
    msw = sum(sum((x - sum(v) / k) ** 2 for x in v) for v in data.values()) / (n * (k - 1))
    denom = msb + (k - 1) * msw
    return (msb - msw) / denom if denom else float("nan")


def summarize(events: list[Event], activity: dict, n_agents: int) -> dict:
    by_id = {e.id: e for e in events}
    origins = Counter(e.origin for e in events)
    kinds = Counter(e.kind for e in events)
    days = (events[-1].tick + 1) // TICKS_PER_DAY or 1

    # Cost
    mind = [e for e in events if e.origin == "mind"]
    tok_in = sum(e.mind["prompt_tokens"] for e in mind)
    tok_out = sum(e.mind["output_tokens"] for e in mind)

    # Consequence depth and mind ancestry (walk parents)
    depth: dict[int, int] = {}
    has_mind: dict[int, int | None] = {}  # hops to nearest mind ancestor
    for e in events:
        ps = [p for p in e.parents if p in by_id]
        depth[e.id] = 1 + max((depth[p] for p in ps), default=0)
        if e.origin == "mind":
            has_mind[e.id] = 0
        else:
            hops = [has_mind[p] + 1 for p in ps if has_mind.get(p) is not None]
            has_mind[e.id] = min(hops) if hops else None
    # Reach of each player intervention: descendant events and distinct agents involved
    reach = {}
    for pe in (e for e in events if e.origin == "player"):
        desc = {pe.id}
        for e in events:
            if e.id > pe.id and any(p in desc for p in e.parents):
                desc.add(e.id)
        actors = {by_id[d].actor for d in desc if by_id[d].actor is not None}
        reach[f"{pe.kind}#{pe.id}"] = {"descendant_events": len(desc) - 1, "agents": len(actors)}
    deepest = max(events, key=lambda e: depth[e.id])
    chain, cur = [], deepest
    while cur is not None:
        chain.append(f"{cur.kind}[{cur.origin}]#{cur.id}")
        ps = [p for p in cur.parents if p in by_id]
        cur = by_id[max(ps, key=lambda p: depth[p])] if ps else None

    # Variety: wake-time and work-hours ICC across days
    wake = defaultdict(list)
    for e in events:
        if e.kind == "wake":
            wake[e.actor].append((e.tick % TICKS_PER_DAY) * 10)
    work = defaultdict(lambda: defaultdict(int))
    # activity events give transitions; count work ticks from plan receipts is coarse, so use mind plans
    import json
    for e in mind:
        if e.data["kind"] == "plan":
            try:
                plan = json.loads(e.mind["raw_output"]).get("plan", [])
            except Exception:
                plan = []
            plan = plan if isinstance(plan, list) else []
            hrs = sum(b["end_h"] - b["start_h"] for b in plan if isinstance(b, dict)
                      and b.get("activity") == "work" and isinstance(b.get("start_h"), int)
                      and isinstance(b.get("end_h"), int) and b["end_h"] > b["start_h"])
            work[e.actor][e.tick // TICKS_PER_DAY] += hrs
    work_groups = {a: list(d.values()) for a, d in work.items()}

    # Lockstep: max share of awake agents doing the same activity in any hour (excluding night)
    lock = []
    for (day, hour), counts in activity.items():
        total = sum(counts.values())
        if total >= 10 and 8 <= hour <= 20:
            lock.append(max(counts.values()) / total)

    return {
        "days": days, "agents": n_agents, "events": len(events),
        "events_by_origin": dict(origins), "events_by_kind": dict(kinds),
        "mind_calls_per_agent_day": round(len(mind) / (n_agents * days), 2),
        "tokens_per_agent_day": {"in": tok_in // (n_agents * days), "out": tok_out // (n_agents * days)},
        "share_events_with_mind_ancestor_within_3": round(
            sum(1 for v in has_mind.values() if v is not None and v <= 3) / len(events), 3),
        "max_causal_depth": depth[deepest.id],
        "player_reach": reach,
        "deepest_chain": list(reversed(chain)),
        "variety": {
            "wake_icc": round(icc1(wake), 3),
            "work_hours_icc": round(icc1(work_groups), 3),
            "wake_range_min": (min(min(v) for v in wake.values()), max(max(v) for v in wake.values())) if wake else None,
            "max_lockstep_share_daytime": round(max(lock), 3) if lock else None,
            "mean_lockstep_share_daytime": round(sum(lock) / len(lock), 3) if lock else None,
        },
    }
