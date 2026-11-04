"""Copy diagnostics: how much of a mind's output repeats what its prompt already handed it (lesson L12).

  plan_copy_rate    share of plans (after an agent's first) whose accepted blocks equal that agent's
                    previous plan. Engine 3 showed yesterday's plan at wake as "Today's plan"; 86% of
                    t1e-4b full plans were copies.
  value_word_share  share of non-empty core entries the mind wrote that only name values ("power",
                    "hedonism, power"). Counted on what the mind wrote, before engine 4 rejects them.

Reads any engine's log without replaying it, so old and new runs compare directly.

    python -m asiimov.diagnostics results/raw/t1e-4b/full/events.jsonl.gz [more logs...]
"""
import argparse
import json

from .memory import is_bare_value
from .sim import valid_blocks


def _reply(e) -> dict | None:
    try:
        out = json.loads(e.mind["raw_output"])
    except (ValueError, RecursionError):
        return None
    return out if isinstance(out, dict) else None


def diagnose(events: list) -> dict:
    last: dict[int, list] = {}
    plans = copies = entries = bare = 0
    for e in events:
        if e.origin != "mind":
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
    return {"plans_after_first": plans, "plan_copy_rate": round(copies / plans, 3) if plans else None,
            "core_entries": entries, "value_word_share": round(bare / entries, 3) if entries else None}


def main():
    from .probe import read_log  # probe imports scorecard, which imports this module
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    for path in ap.parse_args().logs:
        print(path, json.dumps(diagnose(read_log(path))), flush=True)


if __name__ == "__main__":
    main()
