"""Copy diagnostics: how much of a mind's output repeats what its prompt already handed it (lesson L12).

  plan_copy_rate    share of plans (after an agent's first) whose accepted blocks equal that agent's
                    previous plan. Engine 3 showed yesterday's plan at wake as "Today's plan"; 86% of
                    t1e-4b full plans were copies.
  value_word_share  share of non-empty core entries the mind wrote that only name values ("power",
                    "hedonism, power"). Counted on what the mind wrote, before engine 4 rejects them.
  person_share      share of those entries that name another villager (a name word or #id, not one's
                    own): the signal that a memory records something that happened with someone. A
                    label wrapped in a sentence
                    ("hedonism is what drives me") escapes value_word_share but not this.
  place_share       share that name a place in a sentence of 3+ words (a lone "market" doesn't count).
                    Weaker: to-do lines ("buy food before I go home") count here too.
  meals_per_agent_day_by_role  eat events per agent per simulated day, by role. A fed agent eats about
                    3 a day; under the engine-4 economy most non-farmers fell to about 0.5 from day 2.
  food_by_day       food produced, sold to the market and eaten, per day.

Reads any engine's log without replaying it, so old and new runs compare directly.

    python -m asiimov.diagnostics results/raw/t1e-4b/full/events.jsonl.gz [more logs...]
"""
import argparse
from collections import Counter
import json
import re

from .memory import is_bare_value
from .sim import TICKS_PER_DAY, WORKPLACE, valid_blocks

# A failed buy at the market: one reason up to engine 8, split into its two causes from engine 9.
SUPPLY_FAILS = frozenset({"no stock or no coin", "no stock", "no coin"})
PLACES = frozenset(WORKPLACE.values()) | {"market", "plaza", "woods", "home"}


def _reply(e) -> dict | None:
    try:
        out = json.loads(e.mind["raw_output"])
    except (ValueError, RecursionError):
        return None
    return out if isinstance(out, dict) else None


def names_person(text: str, others: frozenset, own_id: int | None) -> bool:
    """Names another villager: a #id other than one's own, or a word of someone else's name
    (one's own name words, surname included, don't count)."""
    if any(int(i) != own_id for i in re.findall(r"#(\d+)", text)):
        return True
    return bool(set(re.findall(r"[a-z]+", text.lower())) & others)


def names_place(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return len(words) >= 3 and bool(set(words) & PLACES)


def food(events: list) -> dict:
    pop = events[0].data.get("population", {}) if events and events[0].kind == "genesis" else {}
    roles = {a["id"]: a["role"] for a in pop.get("agents", ())}
    days = events[-1].tick // TICKS_PER_DAY + (events[-1].tick % TICKS_PER_DAY > 0) if events else 0
    meals = {r: 0 for r in sorted(set(roles.values()))}
    by_day = [{"produced": 0, "sold": 0, "bought": 0, "eaten": 0, "from_household": 0, "deliberations": 0,
               "spoiled": 0}
              for _ in range(days)]
    buys = failed = 0
    why = Counter()
    for e in events:
        d = e.tick // TICKS_PER_DAY
        if e.kind == "eat":
            meals[roles[e.actor]] += 1
            by_day[d]["eaten"] += 1
            by_day[d]["from_household"] += "household" in e.data
        elif e.kind == "receipt" and e.data["action"] == "buy_food":
            buys += 1
            failed += not e.data["ok"]
            why[e.data["why"]] += not e.data["ok"]
            by_day[d]["bought"] += e.data.get("food", 1) if e.data["ok"] else 0
        elif e.kind == "spoil":
            by_day[d]["spoiled"] += e.data["market_food_lost"]
        elif e.origin == "mind" and e.data["kind"] == "deliberate":
            by_day[d]["deliberations"] += 1
        elif e.kind == "produce" and "food" in e.data:
            by_day[d]["produced"] += e.data["food"]
        elif e.kind == "sell":
            by_day[d]["sold"] += e.data["food"]
    count = {r: sum(1 for x in roles.values() if x == r) for r in meals}
    return {"meals_per_agent_day_by_role": {r: round(meals[r] / (count[r] * days), 2) for r in meals if days},
            "buy_attempts": buys, "buy_failed_share": round(failed / buys, 3) if buys else None,
            "buy_failed_by_why": {w: n for w, n in sorted(why.items()) if n},
            "food_by_day": by_day}


def effort(events: list) -> dict:
    """Whether the hungry are the minds' doing: working-age agents (not children or elders) who
    completed no unit of work count as idle. Reported next to the role averages of food()."""
    pop = events[0].data.get("population", {}) if events and events[0].kind == "genesis" else {}
    ags = [a for a in pop.get("agents", ()) if a.get("role") not in (None, "child", "elder")]
    days = events[-1].tick // TICKS_PER_DAY + (events[-1].tick % TICKS_PER_DAY > 0) if events else 0
    earned = {e.actor for e in events if e.kind == "produce"}
    meals, seen = Counter(), set()
    buys = supply_fails = futile = not_at_market = 0
    buy_days, fail_days = set(), set()  # (agent, day) pairs: the acceptance bar counts a day once
    for e in events:
        if e.kind == "eat":
            meals[e.actor] += 1
        elif e.kind == "receipt":
            if not e.data["ok"]:  # the same failure again, the same day
                key = (e.actor, e.tick // TICKS_PER_DAY, e.data["action"], e.data["why"])
                futile += key in seen
                seen.add(key)
            if e.data["action"] == "buy_food":
                not_at_market += e.data["why"] == "not at market"
                if e.actor in earned:
                    buys += 1
                    supply_fails += e.data["why"] in SUPPLY_FAILS
                    buy_days.add((e.actor, e.tick // TICKS_PER_DAY))
                    if e.data["why"] in SUPPLY_FAILS:
                        fail_days.add((e.actor, e.tick // TICKS_PER_DAY))
    working = {}
    for r in sorted({a["role"] for a in ags}):
        w = [a["id"] for a in ags if a["role"] == r and a["id"] in earned]
        if w and days:
            working[r] = round(sum(meals[i] for i in w) / (len(w) * days), 2)
    by_c = sorted(ags, key=lambda a: (a.get("traits", {}).get("conscientiousness", 0), a["id"]))
    quart = [by_c[len(by_c) * q // 4:len(by_c) * (q + 1) // 4] for q in range(4)]
    return {"meals_per_agent_day_working_by_role": working,
            "idle_share": round(sum(a["id"] not in earned for a in ags) / len(ags), 3) if ags else None,
            "idle_share_by_conscientiousness_quartile": [
                round(sum(a["id"] not in earned for a in q) / len(q), 3) if q else None for q in quart],
            # every attempt, repeats included; the frozen T1i bar reads the agent-day share
            "earner_supply_failed_attempt_share": round(supply_fails / buys, 3) if buys else None,
            "earner_supply_failed_agent_day_share": round(len(fail_days) / len(buy_days), 3) if buy_days else None,
            "futile_repeats": futile, "not_at_market": not_at_market}


def diagnose(events: list) -> dict:
    pop = events[0].data.get("population", {}) if events and events[0].kind == "genesis" else {}
    own = {i: frozenset(re.findall(r"[a-z]+", a["name"].lower())) for i, a in enumerate(pop.get("agents", ()))}
    everyone = frozenset().union(*own.values())
    last: dict[int, list] = {}
    plans = copies = entries = bare = person = place = 0
    for e in events:
        if e.origin != "mind" or e.data.get("retried"):  # the retry (T21) supersedes it
            continue
        out = _reply(e)
        if e.data["kind"] == "plan":
            blocks = valid_blocks(out)[0]
            if e.actor in last:
                plans += 1
                copies += blocks == last[e.actor]
            last[e.actor] = blocks
        elif e.data["kind"] == "reflect" and out and isinstance(out.get("core"), list):
            for c in out["core"][:2]:  # the entries the world considers, as in Sim._fall_asleep
                if isinstance(c, str) and c.strip():
                    entries += 1
                    bare += is_bare_value(c)
                    person += names_person(c, everyone - own.get(e.actor, frozenset()), e.actor)
                    place += names_place(c)
    return {"plans_after_first": plans, "plan_copy_rate": round(copies / plans, 3) if plans else None,
            "core_entries": entries, "value_word_share": round(bare / entries, 3) if entries else None,
            "person_share": round(person / entries, 3) if entries else None,
            "place_share": round(place / entries, 3) if entries else None, **food(events), **effort(events)}


def main():
    from .probe import read_log  # probe imports scorecard, which imports this module
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    for path in ap.parse_args().logs:
        print(path, json.dumps(diagnose(read_log(path))), flush=True)


if __name__ == "__main__":
    main()
