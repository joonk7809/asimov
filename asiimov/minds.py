"""Minds: anything that turns a prompt (plus, for stand-ins, a perceived view) into raw output.

The world never trusts a mind's output: it parses, validates and returns receipts.
StubMind is a deterministic, parameter-driven stand-in used until the GPU model is up.
It is NOT the product's mind; it exists to exercise the pipeline and as a no-LLM baseline.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class MindRequest:
    kind: str          # plan | deliberate | reflect
    agent_id: int
    tick: int
    prompt: str
    prompt_hash: str
    view: dict         # perceived facts only (own state, people present, kin)


@dataclass
class MindReply:
    raw_output: str
    model: str
    sampling: dict


class StubMind:
    model = "stub-rules-v0"

    def decide(self, req: MindRequest) -> MindReply:
        v = req.view
        if req.kind == "plan":
            out = self._plan(v)
        elif req.kind == "deliberate":
            out = self._deliberate(v)
        else:
            out = self._reflect(v)
        return MindReply(json.dumps(out, sort_keys=True), self.model, {"temperature": 0})

    def _plan(self, v: dict) -> dict:
        t = v["traits"]
        hour = v["hour"]
        if v["role"] in ("child", "elder"):
            blocks = [{"activity": "socialize", "start_h": min(hour + 2, 22), "end_h": min(hour + 5, 23)}]
        else:
            hours = 4 + t["conscientiousness"] * 5 // 1000
            start = hour + 1 + (1 if t["conscientiousness"] < 300 else 0)
            end = min(start + hours, 21)
            blocks = [{"activity": "work", "start_h": start, "end_h": end}]
            if t["extraversion"] > 550:
                blocks.append({"activity": "socialize", "start_h": end,
                               "end_h": min(end + 1 + t["extraversion"] // 400, 23)})
        return {"plan": blocks}

    def _deliberate(self, v: dict) -> dict:
        t = v["traits"]
        if v["coin"] >= v["food_price"] and v["at_market"] and v.get("market_food", 0) > 0:
            return {"action": "buy_food", "target": None, "reason": "hungry, have coin"}
        if v["coin"] >= v["food_price"] and not v["at_market"]:
            return {"action": "go_market", "target": None, "reason": "need food"}
        holders = [p for p in v["present"] if p["has_food"] and not p["kin"]]
        if t["honesty"] < 350 and holders:
            return {"action": "steal", "target": holders[0]["id"], "reason": "hungry and nobody is watching"}
        kin = [p for p in v["kin"] if p["has_food"]]
        if kin:
            return {"action": "ask_kin", "target": kin[0]["id"], "reason": "family helps"}
        return {"action": "wait", "target": None, "reason": "nothing to do"}

    def _reflect(self, v: dict) -> dict:
        marked = [r for r in v["recent"] if any(w in r for w in ("stole", "robbed", "fire", "caught"))]
        return {"core": marked[:2], "summary": f"{len(v['recent'])} things happened today"}


class NoisyMind(StubMind):
    """Stand-in for a sampled LLM: same policy, but with OS randomness. Not reproducible."""
    model = "noisy-stub-v0"

    def decide(self, req: MindRequest) -> MindReply:
        reply = super().decide(req)
        out = json.loads(reply.raw_output)
        roll = os.urandom(1)[0]
        if "plan" in out and out["plan"] and roll < 128:
            out["plan"][0]["end_h"] = max(out["plan"][0]["start_h"] + 1, out["plan"][0]["end_h"] - 1)
        elif out.get("action") == "steal" and roll < 128:
            out = {"action": "wait", "target": None, "reason": "lost nerve"}
        return MindReply(json.dumps(out, sort_keys=True), self.model, {"temperature": 0.7})


class ReplayMind:
    """Serves recorded outputs by (tick, agent) and refuses any request that doesn't match the record.

    Keyed, not sequential: requests are asked in agent order but committed in the world's
    acting order, and an agent asks at most once per tick.
    """

    def __init__(self, mind_events: list):
        self.record = {}
        for ev in mind_events:
            if (ev.tick, ev.actor) in self.record:
                raise ReplayDivergence(f"two mind events for agent {ev.actor} at tick {ev.tick}")
            self.record[(ev.tick, ev.actor)] = ev

    def decide(self, req: MindRequest) -> MindReply:
        ev = self.record.pop((req.tick, req.agent_id), None)
        if ev is None:
            raise ReplayDivergence(f"unrecorded mind request at tick {req.tick} for agent {req.agent_id}")
        expected = (ev.data["kind"], ev.mind["prompt_hash"])
        got = (req.kind, req.prompt_hash)
        if expected != got:
            raise ReplayDivergence(f"mind request mismatch at tick {req.tick} for agent {req.agent_id}: "
                                   f"expected {expected}, got {got}")
        return MindReply(ev.mind["raw_output"], ev.mind["model"], ev.mind["sampling"])


def dispatch(mind, reqs: list[MindRequest], workers: int = 1) -> list[MindReply]:
    """Ask the mind every request of a tick; replies come back in request order, however they arrive.

    workers > 1 calls the mind from that many threads, so the mind must be thread-safe (LLMMind is).
    """
    if workers <= 1 or len(reqs) <= 1:
        return [mind.decide(r) for r in reqs]
    with ThreadPoolExecutor(min(workers, len(reqs))) as ex:
        return list(ex.map(mind.decide, reqs))


class ForbiddenMind:
    def decide(self, req: MindRequest) -> MindReply:
        raise AssertionError("a mind was called during replay")


class ReplayDivergence(Exception):
    pass
