"""SANDBOX (B5): the 8B base-rate pilot for the stakes asks (lesson L24). Never merges.

pack:  run stub worlds with the stakes levers and keep every ask (dues, stock, sick) as it would reach a mind:
       the full prompt, and the same prompt with the soul and core memories removed (no_text). Sampled to
       --per asks per channel, spread over seeds and days. $0, deterministic.
ask:   send every packed prompt once per arm to an OpenAI-compatible server (the pod's vLLM), at the canonical
       sampling (T 0.7, top_p 0.9, structured JSON: action from the ask's two options, reason). Appends each
       answer as it arrives, so a rerun skips what is done.
read:  per channel and arm, each option's share with its Wilson 95% interval; the choice against each trait
       (point-biserial r) in full and no_text; the share of reasons that cite a plan or schedule (D2's failure)
       or any held-out cue word (L25). Its rules are fixed in plan/b5-stakes-design.md before any output.

python -m asiimov.stakes_pilot pack --out runs/b5-pilot --per 150
python -m asiimov.stakes_pilot ask --out runs/b5-pilot --url $LLM_URL --model $LLM_MODEL --key $LLM_KEY
python -m asiimov.stakes_pilot read --out runs/b5-pilot
"""
import argparse
import json
import math
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .llm import LLMMind
from .memory import assemble_prompt, prompt_hash
from .minds import MindRequest
from .params import TRAITS
from .stakes import ASKS, LEVERS, StakesSim, StakesStub
from .sim import TICKS_PER_DAY

PACK_LEVERS = tuple(x for x in LEVERS if x != "lean")  # the package as designed; lean is a calibration
PLAN_WORDS = re.compile(r"\b(plan|plans|planned|schedule|scheduled)\b", re.I)


class PackSim(StakesSim):
    """Keeps each stakes ask's prompts. The world runs exactly as StakesSim; nothing here changes it."""

    def __init__(self, *a, **k):
        self.kept = []
        super().__init__(*a, **k)

    def _request(self, a, kind, view, parents, apply):
        super()._request(a, kind, view, parents, apply)
        why = (view.get("occasion") or {}).get("why")
        if kind != "deliberate" or why not in ASKS:
            return
        req = self.pending[a["id"]][0]
        saved = a["soul"], a["core"]
        a["soul"], a["core"] = "", []
        try:
            situation = self._situation(a, json.loads(json.dumps(req.view)), "deliberate")
            nt, _ = assemble_prompt(a, self.state, "deliberate", situation, actions=req.view["offer"])
        finally:
            a["soul"], a["core"] = saved
        self.kept.append({"what": why, "agent": a["id"], "tick": self.state["tick"], "day": self.day(),
                          "coin": a["inv"]["coin"], "food": a["inv"]["food"], "role": a["role"],
                          "traits": a["traits"], "values_top": a["values"]["top"], "options": list(ASKS[why]),
                          "full": req.prompt, "no_text": nt})


def pack(out: str, per: int, seeds: list[int], days: int):
    kept = []
    for seed in seeds:
        sim = PackSim(seed, 100, StakesStub(), levers=PACK_LEVERS)
        sim.run_until(days * TICKS_PER_DAY)
        for k in sim.kept:
            k["seed"] = seed
        kept += sim.kept
    chosen = []
    for what in ASKS:
        xs = [k for k in kept if k["what"] == what]
        step = max(1, len(xs) / per)  # evenly spaced over seeds and days, deterministic
        chosen += [xs[int(i * step)] for i in range(min(per, len(xs)))]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "pack.jsonl"), "w") as f:
        for i, k in enumerate(chosen):
            f.write(json.dumps({"i": i, **k}) + "\n")
    counts = {w: sum(1 for k in chosen if k["what"] == w) for w in ASKS}
    print(json.dumps({"available": {w: sum(1 for k in kept if k["what"] == w) for w in ASKS}, "packed": counts}))


def schema(options):
    return {"type": "object", "properties": {"action": {"type": "string", "enum": options},
                                             "target": {"type": "null"}, "reason": {"type": "string"},
                                             "goal": {"type": "string", "enum": [""]},
                                             "then": {"type": "array", "maxItems": 0}},
            "required": ["action", "target", "reason", "goal", "then"], "additionalProperties": False}


def ask(out: str, url: str, model: str, key: str | None, workers: int, arms=("full", "no_text")):
    items = [json.loads(x) for x in open(os.path.join(out, "pack.jsonl"))]
    path = os.path.join(out, "answers.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["i"], r["arm"]) for r in map(json.loads, open(path))}
    mind = LLMMind(url, model, api_key=key, structured=False)
    lock = threading.Lock()

    def one(item, arm):
        prompt = item[arm]
        req = MindRequest("deliberate", item["agent"], item["tick"], prompt, prompt_hash(prompt), {})
        body = mind.body(req)
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "deliberate", "schema": schema(item["options"]), "strict": True}}
        resp = mind.post("/v1/chat/completions", body)
        ch = resp["choices"][0]
        rec = {"i": item["i"], "arm": arm, "raw": (ch["message"].get("content") or "").strip(),
               "finish": ch.get("finish_reason"), "model": resp.get("model"), "seed": body["seed"]}
        with lock, open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    jobs = [(it, arm) for it in items for arm in arms if (it["i"], arm) not in done]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: one(*j), jobs))
    print(f"asked {len(jobs)}; {len(done) + len(jobs)} answers in {path}")


def wilson(k: int, n: int) -> list[float]:
    if not n:
        return [0.0, 1.0]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def corr(xs, ys) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return None if not sx or not sy else round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy), 3)


def read(out: str, cues_path: str | None):
    items = {x["i"]: x for x in map(json.loads, open(os.path.join(out, "pack.jsonl")))}
    ans = [json.loads(x) for x in open(os.path.join(out, "answers.jsonl"))]
    cues = []
    if cues_path:
        c = json.load(open(cues_path))
        cues = sorted({w for lv in c["traits"].values() for lst in lv for w in lst}
                      | {w for lst in c["values"].values() for w in lst})
    res = {}
    for what in ASKS:
        for arm in ("full", "no_text"):
            rows = []
            for r in ans:
                it = items[r["i"]]
                if it["what"] != what or r["arm"] != arm:
                    continue
                try:
                    out_ = json.loads(r["raw"])
                except ValueError:
                    out_ = {}
                rows.append((it, out_.get("action"), str(out_.get("reason", ""))))
            n = len(rows)
            first = ASKS[what][0]
            k = sum(1 for _, a, _ in rows if a == first)
            parsed = sum(1 for _, a, _ in rows if a in ASKS[what])
            y = [1 if a == first else 0 for _, a, _ in rows]
            low = [r_.lower() for _, _, r_ in rows]
            res[f"{what}/{arm}"] = {
                "n": n, "parse_share": round(parsed / n, 3) if n else None,
                f"{first}_share": round(k / n, 3) if n else None, "wilson95": wilson(k, n),
                "r_with_trait": {t: corr([it["traits"][t] for it, _, _ in rows], y) for t in TRAITS},
                "r_with_coin": corr([it["coin"] for it, _, _ in rows], y),
                "plan_word_share": round(sum(1 for x in low if PLAN_WORDS.search(x)) / n, 3) if n else None,
                "cue_word_share": round(sum(1 for x in low if any(re.search(r"\b" + re.escape(w), x) for w in cues))
                                        / n, 3) if n and cues else None,
            }
    print(json.dumps(res, indent=1))
    with open(os.path.join(out, "read.json"), "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out", required=True)
    p.add_argument("--per", type=int, default=150)
    p.add_argument("--seeds", default="1,2,3,4,5,6")
    p.add_argument("--days", type=int, default=7)
    q = sub.add_parser("ask")
    q.add_argument("--out", required=True)
    q.add_argument("--url", required=True)
    q.add_argument("--model", required=True)
    q.add_argument("--key")
    q.add_argument("--workers", type=int, default=32)
    r = sub.add_parser("read")
    r.add_argument("--out", required=True)
    r.add_argument("--cues")
    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.out, args.per, [int(s) for s in args.seeds.split(",")], args.days)
    elif args.cmd == "ask":
        ask(args.out, args.url, args.model, args.key, args.workers)
    else:
        read(args.out, args.cues)


if __name__ == "__main__":
    main()
