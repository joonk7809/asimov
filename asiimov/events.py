"""Append-only world event log with provenance.

origin is one of:
  rule   - recomputable from state
  chance - recomputable from the seed and a draw index (genesis included)
  mind   - an LLM (or stand-in) output; NOT recomputable, replay uses the recording
  player - external input; replay uses the recording
"""
import json
from dataclasses import asdict, dataclass, field

ORIGINS = ("rule", "chance", "mind", "player")


@dataclass
class Event:
    id: int
    tick: int
    kind: str
    origin: str
    actor: int | None
    data: dict
    parents: list[int] = field(default_factory=list)
    mind: dict | None = None  # model, prompt_hash, prompt_tokens, sampling, raw_output

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def write_jsonl(events: list[Event], path: str) -> None:
    with open(path, "w") as f:
        for e in events:
            f.write(e.to_json() + "\n")


def read_jsonl(path: str) -> list[Event]:
    with open(path) as f:
        return [Event(**json.loads(line)) for line in f if line.strip()]
