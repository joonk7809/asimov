"""Decision-level counterfactual probe: how much each prompt input moves the decisions a mind makes.

The scorecard's soul ablation is diluted because code runs most of each day. This measures the
soul (and memory, and everything but the situation) at the decisions the mind actually makes:

1. pack:    replay a recorded log, verifying every event, to sampled decision points, and rebuild
            each point's prompt under five input sets (VARIANTS). The full variant must equal the
            recorded prompt, and replay already checks that against the recorded prompt_hash.
2. ask:     re-ask every variant K times (full 2K times, as two independent halves) with a fresh
            seed per sample. Each answer is appended as it arrives, so an abort loses nothing and
            a rerun skips what's done.
3. analyze: per decision point, the excess cross-disagreement between a variant's answers and full's,
            beyond each side's own sampling noise (temperature 0.7 makes full disagree with itself).
            Zero means the input changes nothing. The same statistic between the two halves of full's
            answers is the null: it must sit at zero, or the probe is miscalibrated.

python -m asiimov.probe pack --log events.jsonl.gz --arm full --points 500 --out runs/probe/full
python -m asiimov.probe pack --log fiction.jsonl.gz --arm fiction --prompt-token-offset 56 --out runs/probe/fiction
python -m asiimov.probe ask --out runs/probe/full --pod POD_ID --k 5 --workers 64
python -m asiimov.probe analyze --out runs/probe/full

The world is never changed: variants exist only as prompts sent outside the world.
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .events import Event
from .memory import assemble_prompt, prompt_hash, render_soul, unanchored
from .minds import MindRequest, ReplayDivergence, ReplayMind
from .scorecard import FICTION_HEADER, SCHEMA_HEAD, ArmSim, check_famine_version, parse_arm
from .sim import ACTIVITIES, ENGINE_VERSION, Sim

VARIANTS = ("full", "no_soul", "swapped_soul", "no_memory", "situation_only")
MEMORY_SECTIONS = ("core", "relationships", "beliefs", "episodes", "recent")
KINDS = ("plan", "deliberate", "reflect")
# Simulated (n=167 points of one kind, base action share uniform in 0.2-0.8, uniform planted shift):
# 80% power to detect a shift of about 0.16 in an action's share at k=5, and about 0.11 at k=10.
# For one action moving share by d, the deliberate excess is d**2.
POWER = {5: 0.16, 10: 0.11}
ARMS = ("full", "famine", "fiction")  # arms whose logs replay exactly (prompts are a function of state)


def _open(path, mode="rt"):
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def read_log(path: str) -> list[Event]:
    with _open(path) as f:
        return [Event(**json.loads(line)) for line in f if line.strip()]


# ---------------------------------------------------------------------------- sampling
def sample_points(events: list[Event], n: int, seed: int) -> list[tuple[int, int, str]]:
    """About n decision points, split evenly across kinds (a kind with fewer gives its share to the
    others), and within a kind spread across agents round-robin so no one agent dominates."""
    by_kind = defaultdict(lambda: defaultdict(list))
    for e in events:
        if e.origin == "mind" and not e.data.get("retried"):  # one point per decision: the final attempt
            by_kind[e.data["kind"]][e.actor].append(e.tick)
    rng = random.Random(seed)
    queues = {}
    for kind in KINDS:
        agents = sorted(by_kind[kind])
        rng.shuffle(agents)
        lists = {a: rng.sample(by_kind[kind][a], len(by_kind[kind][a])) for a in agents}
        q = []
        while any(lists.values()):
            for a in agents:
                if lists[a]:
                    q.append((lists[a].pop(), a, kind))
        queues[kind] = q
    picked, want = [], n
    kinds = [k for k in KINDS if queues[k]]
    while kinds and want > 0:
        share = max(1, want // len(kinds))
        for k in list(kinds):
            take, queues[k] = queues[k][:share], queues[k][share:]
            picked += take
            want -= len(take)
            if not queues[k]:
                kinds.remove(k)
            if want <= 0:
                break
    return sorted(picked)


# ---------------------------------------------------------------------------- variants
def donor_for(agent: dict, agents: list[dict]) -> dict:
    """The agent whose traits differ most (L1), lowest id on ties: the strongest soul swap available."""
    t = agent["traits"]
    return max((o for o in agents if o["id"] != agent["id"]),
               key=lambda o: (sum(abs(o["traits"][k] - t[k]) for k in t), -o["id"]))


def _soul_text(person: dict, arm: str) -> str:
    """The soul as `arm` shows it. Since T25 the anchors are part of every soul (T25)."""
    soul = render_soul(person)
    return unanchored(soul) if "unanchored" in parse_arm(arm)[1] else soul


def edit_for(variant: str, agent: dict, arm: str = "full", donor: dict | None = None):
    """The assemble_prompt `edit` for one variant, or None for full. Usable for a whole run too (a Sim
    subclass passing it on every request), not only for one re-asked decision. A swapped soul keeps
    the agent's name, age, role and rhythm and takes the donor's traits and values, as text only."""
    if variant == "full":
        return None
    if variant == "no_soul":
        return lambda s: {k: v for k, v in s.items() if k != "soul"}
    if variant == "swapped_soul":
        swapped = _soul_text({**agent, "traits": donor["traits"], "values": donor["values"]}, arm)
        return lambda s: {**s, "soul": swapped}
    if variant == "no_memory":
        return lambda s: {k: v for k, v in s.items() if k not in MEMORY_SECTIONS}
    if variant == "situation_only":
        return lambda s: {k: s[k] for k in ("schema", "situation")}
    raise ValueError(f"unknown variant {variant}")


def variant_prompt(agent: dict, state: dict, kind: str, situation: str, variant: str, arm: str,
                   donor: dict | None = None, actions=None) -> str:
    """The prompt the world would have sent, with one input set changed. `actions`: what a deliberation
    offered (its request view's "offer")."""
    soul = _soul_text(agent, arm)
    prompt, _ = assemble_prompt({**agent, "soul": soul}, state, kind, situation, edit_for(variant, agent, arm, donor),
                                actions)
    if arm == "fiction":
        assert prompt.startswith(SCHEMA_HEAD)
        prompt = FICTION_HEADER + prompt[len(SCHEMA_HEAD):]
    return prompt


class ProbeSim(ArmSim):
    """Replays a log, verifying every event, and captures the variant prompts at the sampled points."""

    def __init__(self, events: list[Event], arm: str, targets: set, prompt_token_offset: int = 0):
        """prompt_token_offset is added to every recorded mind prompt_tokens before comparing. The only
        known use is 56 for fiction logs written before 1d02a44, which under-recorded by that constant."""
        genesis = events[0].data
        if genesis.get("engine") != ENGINE_VERSION:
            raise ReplayDivergence(f"log was written by engine {genesis.get('engine')}, this is {ENGINE_VERSION}")
        check_famine_version(genesis, arm)
        if prompt_token_offset:
            events = [Event(**{**e.__dict__, "mind": {**e.mind, "prompt_tokens": e.mind["prompt_tokens"]
                                                      + prompt_token_offset}}) if e.mind else e for e in events]
        self.arm, self.souls, self.targets, self.captured = arm, {}, set(targets), {}
        self._last_situation = None
        self.player = defaultdict(list)
        for e in events:
            if e.origin == "player":
                self.player[e.tick].append({"kind": e.kind, **e.data})
        mind = ReplayMind([e for e in events if e.origin == "mind"])
        Sim.__init__(self, genesis["seed"], len(genesis["population"]["agents"]), mind,
                     genesis=genesis["population"], expected=events, order=genesis["order"])

    def _situation(self, *args, **kwargs):  # signature-agnostic: only remembers what the world built
        self._last_situation = super()._situation(*args, **kwargs)
        return self._last_situation

    def _request(self, a, kind, view, parents, apply):
        self._last_situation = None
        super()._request(a, kind, view, parents, apply)
        key = (self.state["tick"], a["id"])
        if key not in self.targets:
            return
        req = self.pending[a["id"]][0]
        agents = self.state["agents"]
        donor = donor_for(a, agents)
        prompts = {v: variant_prompt(a, self.state, kind, self._last_situation, v, self.arm, donor,
                                     req.view.get("offer"))
                   for v in VARIANTS}
        if prompts["full"] != req.prompt:
            raise ReplayDivergence(f"variant hook does not rebuild the recorded prompt at {key}")
        if len(set(prompts.values())) != len(VARIANTS):
            raise AssertionError(f"two variants share a prompt at {key}")
        self.captured[key] = {"tick": key[0], "agent": a["id"], "kind": kind, "donor": donor["id"],
                              "traits": dict(a["traits"]), "role": a["role"], "view": req.view,
                              "prompts": prompts}


def build_pack(events: list[Event], arm: str, n: int, seed: int, prompt_token_offset: int = 0) -> list[dict]:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}: other arms' prompts aren't a function of the log")
    points = sample_points(events, n, seed)
    recorded = {(e.tick, e.actor): e for e in events if e.origin == "mind" and not e.data.get("retried")
                and e.data["kind"] in KINDS}  # text-1: a listener's `decide` shares its tick with its own request
    sim = ProbeSim(events, arm, {(t, a) for t, a, _ in points}, prompt_token_offset)
    sim.run_until(max(t for t, _, _ in points) + 1, sim.player)
    pack = []
    for i, (t, a, kind) in enumerate(points):
        c = sim.captured[(t, a)]
        assert c["kind"] == kind
        ev = recorded[(t, a)]
        c.update({"id": i, "recorded_output": ev.mind["raw_output"], "sampling": ev.mind["sampling"],
                  "model": ev.mind["model"]})
        pack.append(c)
    return pack


# ---------------------------------------------------------------------------- asking
def jobs(pack: list[dict], k: int) -> list[tuple[int, str, int]]:
    """Full is asked 2k times (reps 0..2k-1: two halves for the noise floor), every other variant k."""
    return [(p["id"], v, r) for p in pack for v in VARIANTS for r in range(2 * k if v == "full" else k)]


def sample_seed(point: dict, variant: str, rep: int) -> int:
    """A seed per sample. The world seeds from the prompt hash, so re-asking the recorded prompt with that
    seed would repeat one sample and zero the noise floor while every variant (new prompt, new seed) moved."""
    h = hashlib.sha256(f"probe:{point['tick']}:{point['agent']}:{variant}:{rep}".encode()).hexdigest()
    return int(h[:8], 16)


class AskIncomplete(RuntimeError):
    pass


def ask(pack: list[dict], mind, k: int, answers_path: str, workers: int = 1, max_failures: int = 50) -> int:
    """Ask every job not yet in answers_path; append each answer as it arrives. Returns answers written.

    A failed job never discards the others: every answer that arrives is written. Once max_failures jobs
    have failed (a dead pod, say) the queue is cancelled. Either way AskIncomplete is raised at the end
    with the gap count, and a rerun asks only the gaps."""
    done = set()
    if os.path.exists(answers_path):
        with open(answers_path, "rb+") as f:  # drop a line a crash cut off, so the next append starts clean
            blob = f.read()
            if blob and not blob.endswith(b"\n"):
                f.truncate(blob.rfind(b"\n") + 1)
        with open(answers_path) as f:
            for line in f:
                try:
                    a = json.loads(line)
                except ValueError:  # a line cut off by a crash: ask that job again
                    continue
                done.add((a["point"], a["variant"], a["rep"]))
    by_id = {p["id"]: p for p in pack}
    todo = [j for j in jobs(pack, k) if j not in done]
    lock = threading.Lock()
    stop = threading.Event()
    written, failures = 0, []

    def one(job):
        if stop.is_set():  # too many failures: skip the rest quickly (cancelled futures never
            return None    # reach as_completed, so the queue drains through this check instead)
        pid, variant, rep = job
        p = by_id[pid]
        prompt = p["prompts"][variant]
        req = MindRequest(p["kind"], p["agent"], p["tick"], prompt, prompt_hash(prompt),
                          json.loads(json.dumps(p["view"])), seed=sample_seed(p, variant, rep))
        try:
            reply = mind.decide(req)
        except Exception as e:  # noqa: BLE001 - counted here, in the worker, so a fast failure stops the queue
            with lock:
                failures.append(e)
                if len(failures) >= max_failures:
                    stop.set()
            return None
        return {"point": pid, "variant": variant, "rep": rep, "seed": req.seed, "raw_output": reply.raw_output,
                "model": reply.model}

    with open(answers_path, "a") as f:
        if workers <= 1:
            results = (one(j) for j in todo)
        else:
            ex = ThreadPoolExecutor(workers)
            results = (fut.result() for fut in as_completed([ex.submit(one, j) for j in todo]))
        try:
            for rec in results:  # written by this thread only, as each answer arrives
                if rec is not None:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                    f.flush()
                    written += 1
        finally:  # on Ctrl-C or a write error too: queued jobs must not keep calling the model
            stop.set()
            if workers > 1:
                ex.shutdown()
    gaps = len(todo) - written
    if gaps:
        raise AskIncomplete(f"{gaps} of {len(todo)} jobs unanswered ({len(failures)} failed, first: "
                            f"{failures[0]!r}); answers so far are saved, rerun ask to fill the gaps")
    return written


# ---------------------------------------------------------------------------- analysis
def _parse(raw: str):
    try:
        out = json.loads(raw)
    except (ValueError, RecursionError):
        return None
    return out if isinstance(out, dict) else None


def decision(kind: str, raw: str):
    """What the world would act on, as a comparable value. Validity follows Sim's own checks."""
    out = _parse(raw)
    if kind == "deliberate":
        if out is None:
            return ("invalid", None)
        t = out.get("target")
        return (str(out.get("action")), t if type(t) is int else None)
    if kind == "plan":
        hours = [None] * 24
        blocks = out.get("plan") if out else None
        for b in blocks if isinstance(blocks, list) else []:
            if (isinstance(b, dict) and b.get("activity") in ACTIVITIES and type(b.get("start_h")) is int
                    and type(b.get("end_h")) is int and 0 <= b["start_h"] < b["end_h"] <= 24):
                for h in range(b["start_h"], b["end_h"]):
                    hours[h] = b["activity"]  # later blocks win, as in Sim._act
        return tuple(hours)
    core = out.get("core") if out else None
    words = set()
    for c in core[:2] if isinstance(core, list) else []:
        if isinstance(c, str):
            words |= {w for w in re.findall(r"[a-z]+", c.lower()) if len(w) > 2}
    return frozenset(words)


def distance(kind: str, x, y) -> float:
    """deliberate: different action or target; plan: share of hours planned differently;
    reflect: Jaccard distance between the words of the promoted core memories."""
    if kind == "deliberate":
        return float(x != y)
    if kind == "plan":
        return sum(a != b for a, b in zip(x, y)) / 24
    if not x and not y:
        return 0.0
    return 1 - len(x & y) / len(x | y)


def excess(kind: str, xs: list, ys: list) -> float:
    """Mean cross-distance minus the mean of each side's own pairwise distance (half the energy
    distance). Zero in expectation when both sets come from the same answer distribution."""
    cross = sum(distance(kind, x, y) for x in xs for y in ys) / (len(xs) * len(ys))
    return cross - (_within(kind, xs) + _within(kind, ys)) / 2


def _within(kind: str, xs: list) -> float:
    pairs = [(i, j) for i in range(len(xs)) for j in range(i + 1, len(xs))]
    return sum(distance(kind, xs[i], xs[j]) for i, j in pairs) / len(pairs)


def _ci(vals: list[float], seed: int, n_boot: int = 2000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def analyze(pack: list[dict], answers: list[dict], k: int, seed: int = 0) -> dict:
    got = defaultdict(dict)
    for a in answers:
        got[(a["point"], a["variant"])][a["rep"]] = a["raw_output"]
    per_kind = defaultdict(lambda: defaultdict(list))  # kind -> stat -> per-point values
    actions = defaultdict(Counter)  # deliberate only: variant -> action counts
    incomplete = 0
    for p in pack:
        kind = p["kind"]
        dec = {}
        for v in VARIANTS:
            reps = got[(p["id"], v)]
            need = 2 * k if v == "full" else k
            if any(r not in reps for r in range(need)):
                break
            dec[v] = [decision(kind, reps[r]) for r in range(need)]
        else:
            full = dec["full"]
            per_kind[kind]["noise_within_full"].append(_within(kind, full))
            per_kind[kind]["null"].append(excess(kind, full[:k], full[k:]))
            for v in VARIANTS[1:]:
                per_kind[kind][v].append(excess(kind, full, dec[v]))
            if kind == "deliberate":
                for v in VARIANTS:
                    actions[v].update(d[0] for d in dec[v][:k])
            continue
        incomplete += 1
    out = {"k": k, "points_incomplete": incomplete, "kinds": {}}
    for kind in KINDS:
        stats = per_kind.get(kind)
        if not stats:
            continue
        n = len(stats["null"])
        row = {"points": n,
               "noise_within_full": round(sum(stats["noise_within_full"]) / n, 4),
               "null_excess": _summary(stats["null"], seed)}
        for v in VARIANTS[1:]:
            row[v] = _summary(stats[v], seed)
        out["kinds"][kind] = row
    if actions:
        out["deliberate_actions"] = {v: {a: round(c / sum(actions[v].values()), 3) for a, c in
                                         sorted(actions[v].items())} for v in VARIANTS}
    return out


def _summary(vals: list[float], seed: int) -> dict:
    lo, hi = _ci(vals, seed)
    return {"mean": round(sum(vals) / len(vals), 4), "ci95": [round(lo, 4), round(hi, 4)]}


def render_md(report: dict, meta: dict) -> str:
    lines = [f"# Decision probe: {meta.get('label', '')}", "",
             f"Log sha256 {meta.get('log_sha256', '?')[:16]}, arm {meta.get('arm')}, K={report['k']} "
             f"(full asked 2K), model {meta.get('model')}. Points with missing answers: {report['points_incomplete']}.",
             "",
             "Each cell is the input's effect on decisions beyond sampling noise: how much more the variant's answers "
             "disagree with full's than each set disagrees with itself. The null column is the same statistic "
             "between two halves of full's answers and should be about 0. "
             "Units are the kind's distance (deliberate: share of answer pairs with a different action or target; "
             "plan: share of hours planned differently; reflect: Jaccard distance of promoted core-memory words). "
             "0 means no effect; 95% bootstrap CI over decision points (points cluster by agent, so it is "
             "somewhat narrow; with 12 cells, one marginal exclusion of 0 is weak evidence).", "",
             "swapped_soul uses the most-different donor and swaps text only (params unchanged): an upper bound "
             "on a random swap, and not scorecard v2 test 8.",
             "| kind | points | noise (full vs full) | null excess | " + " | ".join(VARIANTS[1:]) + " | 80% power |",
             "|---|---|---|---|" + "---|" * len(VARIANTS)]
    k = report["k"]
    for kind, row in report["kinds"].items():
        cells = [f"{row[v]['mean']:+.3f} [{row[v]['ci95'][0]:+.3f}, {row[v]['ci95'][1]:+.3f}]" for v in VARIANTS[1:]]
        n = row["null_excess"]
        power = (f"action-share shift {POWER[k]} (excess about {POWER[k] ** 2:.3f}) at 167 points"
                 if kind == "deliberate" and k in POWER else "not simulated")
        lines.append(f"| {kind} | {row['points']} | {row['noise_within_full']:.3f} | "
                     f"{n['mean']:+.3f} [{n['ci95'][0]:+.3f}, {n['ci95'][1]:+.3f}] | " + " | ".join(cells)
                     + f" | {power} |")
    lines += ["", "A CI that includes 0 means the effect is below what this probe detects, not that there is none."]
    if meta.get("engine") == "1" and "plan" in report["kinds"]:
        lines.append("Engine 1 plans were one short block (F2), so the plan row is diluted; the plan row that "
                     "counts comes from an engine 2 log.")
    if "deliberate_actions" in report:
        lines += ["", "Deliberate action shares (first K answers per variant):", ""]
        for v, shares in report["deliberate_actions"].items():
            lines.append(f"- {v}: " + ", ".join(f"{a} {s:.2f}" for a, s in shares.items()))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- CLI
def _write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--log", required=True)
    p.add_argument("--arm", choices=ARMS, default="full")
    p.add_argument("--points", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", default="")
    p.add_argument("--prompt-token-offset", type=int, default=0,
                   help="56 for fiction logs written before 1d02a44 (see ProbeSim)")
    p.add_argument("--out", required=True)
    a = sub.add_parser("ask")
    a.add_argument("--out", required=True)
    a.add_argument("--k", type=int, default=10, help="answers per variant; full gets 2k (see POWER)")
    a.add_argument("--workers", type=int, default=64)
    a.add_argument("--pod")
    a.add_argument("--llm-url")
    a.add_argument("--llm-model")
    a.add_argument("--llm-key")
    z = sub.add_parser("analyze")
    z.add_argument("--out", required=True)
    args = ap.parse_args()
    meta_path = os.path.join(args.out, "meta.json")
    pack_path = os.path.join(args.out, "pack.jsonl.gz")
    answers_path = os.path.join(args.out, "answers.jsonl")

    if args.cmd == "pack":
        os.makedirs(args.out, exist_ok=True)
        with open(args.log, "rb") as f:
            raw = f.read()
        log_sha = hashlib.sha256(gzip.decompress(raw) if args.log.endswith(".gz") else raw).hexdigest()
        events = read_log(args.log)
        pack = build_pack(events, args.arm, args.points, args.seed, args.prompt_token_offset)
        with gzip.open(pack_path, "wt") as f:
            for rec in pack:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        models = sorted({r["model"] for r in pack})
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                    cwd=os.path.dirname(__file__)).stdout.strip() or None
            dirty = bool(subprocess.run(["git", "status", "--porcelain", "asiimov"], capture_output=True, text=True,
                                        cwd=os.path.dirname(__file__)).stdout.strip())
        except OSError:
            commit, dirty = None, None
        _write_json({"label": args.label, "log": args.log, "log_sha256": log_sha, "arm": args.arm,
                     "points": len(pack), "seed": args.seed,
                     "prompt_token_offset": args.prompt_token_offset, "commit": commit, "dirty": dirty, "model": models, "engine": ENGINE_VERSION,
                     "kinds": dict(Counter(r["kind"] for r in pack)), "variants": list(VARIANTS)}, meta_path)
        print(f"packed {len(pack)} points: {dict(Counter(r['kind'] for r in pack))}")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    with gzip.open(pack_path, "rt") as f:
        pack = [json.loads(line) for line in f]

    if args.cmd == "ask":
        from .run import llm_from_args
        mind, _ = llm_from_args(args)
        s = pack[0]["sampling"]  # ask exactly as the world did, except for the seed
        mind.temperature, mind.top_p, mind.max_tokens = s["temperature"], s["top_p"], s["max_tokens"]
        mind.structured = s.get("structured", True)
        if mind.model not in meta["model"]:
            raise SystemExit(f"pod serves {mind.model}, the log was made by {meta['model']}")
        if meta.get("k", args.k) != args.k:
            raise SystemExit(f"answers so far were asked with --k {meta['k']}")
        meta.update({"k": args.k, "sampling": {kk: v for kk, v in s.items() if kk != "seed"}})
        _write_json(meta, meta_path)
        try:
            n = ask(pack, mind, args.k, answers_path, args.workers)
        finally:  # record what this run cost even when it stops short
            meta.setdefault("ask_runs", []).append(mind.stats())
            _write_json(meta, meta_path)
        print(f"answered {n} jobs: {mind.stats()}")
        return

    with open(answers_path) as f:
        answers = []
        for line in f:
            try:
                answers.append(json.loads(line))
            except ValueError:
                pass
    report = analyze(pack, answers, meta["k"])
    _write_json(report, os.path.join(args.out, "report.json"))
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(render_md(report, meta))
    print(render_md(report, meta))


if __name__ == "__main__":
    main()
