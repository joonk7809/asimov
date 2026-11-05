"""Intent-loop readings from an event log (engine 7, T15). Reads the log only; no replay.

Definitions, fixed before any engine-7 model run was read (plan/t15-intent-loop-spec.md, shared folder):

  intents             intent_set events.
  end of an intent    the first intent_done, intent_failed or intent_dropped event citing its intent_set.
                      An intent with none is open: still live when the log stops.
  spans_days_share    share of intents alive on more than one calendar day: the day of its end (for an
                      open intent, the day of the log's last event) is later than the day it was set.
  completed_share     intent_done / (intent_done + intent_failed + intent_dropped), over ended intents.
  mind_calls          mind events per (agent, calendar day); the bar reads the maximum.
  person_directed     interactions a mind chose, between two people: meet events (seek), gifts that went
                      through (give_food / give_coin receipts with ok), and ask_kin and steal attempts that
                      reached their target (receipts carrying a chance roll). Counted per unordered pair.

python -m asiimov.intents results/raw/<run>/full/events.jsonl.gz [more logs...]
"""
import argparse
import json
from collections import Counter

from .sim import TICKS_PER_DAY

ENDS = ("intent_done", "intent_failed", "intent_dropped")


def summarize(events: list) -> dict:
    sets = {e.id: e for e in events if e.kind == "intent_set"}
    end = {}
    for e in events:
        if e.kind in ENDS and e.parents and e.parents[0] in sets and e.parents[0] not in end:
            end[e.parents[0]] = e
    last_day = events[-1].tick // TICKS_PER_DAY if events else 0
    spans = sum((end[i].tick if i in end else events[-1].tick) // TICKS_PER_DAY > s.tick // TICKS_PER_DAY
                for i, s in sets.items())
    ends = Counter(e.kind for e in end.values())
    ended = sum(ends.values())
    steps = [len(s.data["steps"]) for s in sets.values()]
    calls = Counter((e.actor, e.tick // TICKS_PER_DAY) for e in events if e.origin == "mind")
    pairs = Counter()
    by_kind = Counter()
    for e in events:
        other = None
        if e.kind == "meet":
            other, k = e.data["with"], "seek"
        elif e.kind == "receipt" and e.data["action"] in ("give_food", "give_coin") and e.data["ok"]:
            other, k = e.data["to"], e.data["action"]
        elif e.kind == "receipt" and e.data["action"] in ("ask_kin", "steal") and "roll" in e.data:
            other, k = e.data.get("victim", e.data.get("to")), e.data["action"]
        if other is not None:
            pairs[tuple(sorted((e.actor, other)))] += 1
            by_kind[k] += 1
    per_pair = sorted(pairs.values())
    return {
        "intents": len(sets), "open_at_end": len(sets) - ended, "last_day": last_day,
        "spans_days_share": round(spans / len(sets), 3) if sets else None,
        "ended": dict(sorted(ends.items())),
        "completed_share": round(ends["intent_done"] / ended, 3) if ended else None,
        "steps_per_intent_mean": round(sum(steps) / len(steps), 2) if steps else None,
        "mind_calls_per_agent_day_max": max(calls.values()) if calls else 0,
        "mind_calls_per_agent_day_mean": round(sum(calls.values()) / len(calls), 2) if calls else 0,
        "person_directed": {"interactions": sum(pairs.values()), "by_kind": dict(sorted(by_kind.items())),
                            "pairs": len(pairs),
                            "per_pair_mean": round(sum(per_pair) / len(per_pair), 2) if per_pair else None,
                            "pairs_with_4_or_more": sum(n >= 4 for n in per_pair)},
    }


def main():
    from .probe import read_log
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("logs", nargs="+")
    for path in ap.parse_args().logs:
        print(path, json.dumps(summarize(read_log(path))), flush=True)


if __name__ == "__main__":
    main()
