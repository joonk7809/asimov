"""Fork mind and twin runs: re-run a recorded world with a change, asking the model only where it must.

A twin re-runs a base log's world with an intervention (a player action, edited souls) or with
chosen agents forced live. ForkMind serves the base log's recorded output whenever a request
matches it on (tick, agent, kind, prompt_hash), and asks the live mind only on a miss. An LLM sees
nothing but its prompt, so a matching prompt is the same question: outside the intervention's
causal cone the twin is the base world, with zero divergence by construction, and only calls
inside the cone cost money. A miss is asked exactly as a fresh run would ask it. Forced-live
agents are asked every time, with a seed salted by the twin, so a noise twin (no intervention,
the same agents live) measures their sampling band.

The twin's log records every output verbatim, so it replays exactly like any other log. A twin
that dies is resumed by re-running it with its partial log as a second record: its own answers
are served first, so only the tick it died in is asked again.

That holds only for a mind that sees nothing but its prompt. StubMind and NoisyMind also read
MindRequest.view, which shows facts the prompt doesn't, such as who carries food: a stub served
from the record can miss a theft a full stub rerun makes (3 of 40 seeded fire twins, all at the
engine 1-4 yield). StubMind is deterministic, so its twins (the $0 negative controls) are full
reruns with nothing served: `serve=False`, which the CLI sets for --mind stub.

python -m asiimov.fork --base runs/demo/events.jsonl --out runs/twin-fire2 --fire 2@1 --pod POD_ID
python -m asiimov.fork --base runs/demo/events.jsonl --out runs/noise-a --live 12,40 --salt 1 --pod POD_ID
python -m asiimov.fork --resume --out runs/twin-fire2 --pod POD_ID
python -m asiimov.fork --base runs/stub/events.jsonl --out runs/stub-fire2 --fire 2@1 --mind stub
"""
import argparse
import gzip
import hashlib
import json
import os
import threading
from collections import Counter, defaultdict
from dataclasses import replace

from .events import Event, write_jsonl
from .memory import render_soul
from .minds import MindReply, MindRequest, ReplayDivergence, mind_key, recorded_reply  # noqa: F401
from .sim import ENGINE_VERSION, TICKS_PER_DAY, Sim


class ForkLeak(RuntimeError):
    """The twin asked the live mind before anything was changed: it is not the base world's twin."""


def _index(mind_events) -> dict:
    record = {}
    for ev in mind_events:
        key = mind_key(ev.tick, ev.actor, ev.data.get("attempt", 0))
        if key in record:
            raise ReplayDivergence(f"two mind events for agent {ev.actor} at tick {ev.tick}")
        record[key] = ev
    return record


def _serve(ev: Event) -> MindReply:
    return recorded_reply(ev)


def live_seed(salt: int, req: MindRequest) -> int:
    """The sampling seed of a forced-live call. LLMMind otherwise derives it from the prompt hash,
    which would give a forced agent the base run's own seed for an unchanged prompt."""
    blob = f"fork:{salt}:{req.tick}:{req.agent_id}:{req.prompt_hash}" + (f":{req.attempt}" if req.attempt else "")
    blob = blob.encode()
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


class ForkMind:
    """Serves recorded outputs where a request matches them, asks `live` on a miss.

    base:        the base log's events (only mind events are used).
    live_agents: agent ids asked live from tick `live_from` on, even when their prompt matches.
    salt:        separates the samples of twins that share a base (each noise twin gets its own).
    calm_until:  a live call before this tick raises ForkLeak. Set it to the first changed tick, so
                 a twin built wrong (another arm, another engine's prompts) stops before it pays.
    prior:       a crashed run of this same twin; its answers are served first, forced or not.
    serve:       False asks `live` for everything not in `prior`: a full rerun, for minds that read
                 more than their prompt (see the module docstring).
    Thread-safe, so dispatch can call it from many workers.
    """

    def __init__(self, base, live, live_agents=(), live_from: int = 0, salt: int = 0,
                 calm_until: int = 0, prior=(), serve: bool = True):
        self.base = _index(e for e in base if e.origin == "mind") if serve else {}
        self.prior = _index(e for e in prior if e.origin == "mind")
        self.live = live
        self.model = getattr(live, "model", None)
        self.live_agents, self.live_from = frozenset(live_agents), live_from
        self.salt, self.calm_until = salt, calm_until
        self.counts = Counter()  # base | prior | miss | forced
        self._lock = threading.Lock()

    def forced(self, req: MindRequest) -> bool:
        return req.agent_id in self.live_agents and req.tick >= self.live_from

    def decide(self, req: MindRequest) -> MindReply:
        key, want = mind_key(req.tick, req.agent_id, req.attempt), (req.kind, req.prompt_hash)
        for source, record in (("prior", self.prior), ("base", self.base)):
            ev = record.get(key)
            if source == "base" and self.forced(req):
                break
            if ev is not None and (ev.data["kind"], ev.mind["prompt_hash"]) == want:
                self._count(source)
                return _serve(ev)
        source = "forced" if self.forced(req) else "miss"
        if req.tick < self.calm_until and self.base:
            raise ForkLeak(f"{source} at tick {req.tick} for agent {req.agent_id}, before the first change "
                           f"at tick {self.calm_until}: the twin's prompts don't match the base log")
        self._count(source)
        # A miss is a new prompt, asked as the base run would have asked it. A forced call repeats
        # a prompt the base already answered, so it needs its own seed to be a new sample.
        return self.live.decide(replace(req, seed=live_seed(self.salt, req)) if source == "forced" else req)

    def _count(self, source: str):
        with self._lock:
            self.counts[source] += 1


# ---------------------------------------------------------------------------- reading twins
def _agent_streams(events) -> dict[int, list]:
    """Each agent's own events without ids and parents, which shift whenever any event is added."""
    out = defaultdict(list)
    for e in events:
        if e.actor is not None and e.kind != "end":
            out[e.actor].append((e.tick, e.kind, json.dumps(e.data, sort_keys=True),
                                 e.mind["raw_output"] if e.mind else None))
    return out


def divergence(base, twin) -> dict[int, int]:
    """Agent id -> the first tick its own events differ between the two logs (diverged agents only)."""
    a, b = _agent_streams(base), _agent_streams(twin)
    out = {}
    for aid in sorted(set(a) | set(b)):
        xs, ys = a.get(aid, []), b.get(aid, [])
        for i in range(max(len(xs), len(ys))):
            x = xs[i] if i < len(xs) else None
            y = ys[i] if i < len(ys) else None
            if x != y:
                out[aid] = min(e[0] for e in (x, y) if e is not None)
                break
    return out


def live_calls(base, twin, live_agents=(), live_from: int = 0) -> list[tuple[int, int, str]]:
    """The twin's mind events that the live mind answered: (tick, agent, 'miss' | 'forced').
    Derived from the two logs and the twin's config, so a resumed twin needs no extra record."""
    record = _index(e for e in base if e.origin == "mind")
    live_agents = frozenset(live_agents)
    out = []
    for e in twin:
        if e.origin != "mind":
            continue
        if e.actor in live_agents and e.tick >= live_from:
            out.append((e.tick, e.actor, "forced"))
            continue
        b = record.get(mind_key(e.tick, e.actor, e.data.get("attempt", 0)))
        if b is None or (b.data["kind"], b.mind["prompt_hash"]) != (e.data["kind"], e.mind["prompt_hash"]):
            out.append((e.tick, e.actor, "miss"))
    return out


# ---------------------------------------------------------------------------- running twins
def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def read_log(path: str) -> list[Event]:
    with _open(path) as f:
        return [Event(**json.loads(line)) for line in f if line.strip()]


def end_tick(events) -> int:
    """The first tick the base log didn't run (scorecard arm logs end on a checkpoint, not `end`)."""
    last = events[-1]
    return last.tick if last.kind == "end" else last.tick + 1


def base_player(events) -> dict[int, list]:
    player = defaultdict(list)
    for e in events:
        if e.origin == "player":
            player[e.tick].append({"kind": e.kind, **e.data})
    return player


def build_sim(base, mind, arm: str | None = None, genesis: dict | None = None, workers: int = 1) -> Sim:
    """A world like the base log's, from its genesis (or an edited copy), driven by `mind`.
    Scorecard arm logs need their arm: its prompts differ from the plain world's."""
    g = base[0]
    if g.kind != "genesis" or g.data.get("engine") != ENGINE_VERSION:
        raise ReplayDivergence(f"base log was written by engine {g.data.get('engine')}, this is {ENGINE_VERSION}")
    pop = genesis if genesis is not None else g.data["population"]
    n = len(pop["agents"])
    if arm is None:
        return Sim(g.data["seed"], n, mind, genesis=pop, order=g.data["order"], workers=workers)
    from .scorecard import ArmSim, check_famine_version, labeled_population, parse_arm
    check_famine_version(g.data, arm)
    if g.data["order"] != "id":
        raise ValueError("scorecard arms run in id order")
    flat = parse_arm(arm)[0] == "flat_params"
    real = labeled_population(g.data["seed"], n) if flat else pop  # flat_params prompts keep the real souls
    return ArmSim(g.data["seed"], n, mind, pop, arm, {a["id"]: render_soul(a) for a in real["agents"]}, workers)


def run_twin(base, live, out: str | None = None, player: dict | None = None, live_agents=(), live_from: int = 0,
             salt: int = 0, arm: str | None = None, genesis: dict | None = None, until: int | None = None,
             workers: int = 1, prior=(), serve: bool = True, drop=(), souls: dict | None = None) -> tuple[Sim, ForkMind]:
    """Run the base world again with `player` actions added (tick -> actions), the base's own player
    actions in `drop` ((tick, action) pairs) removed, `genesis` edits and `live_agents` forced live,
    to `until` (default: where the base stops). A dropped action is a change at its tick: the
    counterfactual twin of the base's own fire drops it. `souls` (agent id -> text) replaces those
    agents' soul text from tick 0 and nothing else: their params stay, so only a mind reading the
    prompt can follow it (test 8, Steering's ruling 2026-11-09). A twin with souls replays only with them. Returns the finished sim
    and its mind (whose counts say how each call was answered).

    Nothing may differ from the base before the first change: the player actions' first tick,
    live_from when agents are forced live, or tick 0 for an edited genesis. A live call there
    raises ForkLeak before it is paid for, and a finished twin that diverged there raises too.
    On a crash the events so far go to out/events.partial.jsonl; pass them back as `prior`.
    """
    player = {t: list(acts) for t, acts in (player or {}).items()}
    merged = base_player(base)
    for t, act in drop:
        if act not in merged.get(t, []):
            raise ValueError(f"drop: the base has no player action {act} at tick {t}")
        merged[t].remove(act)
    for t, acts in player.items():
        merged[t] = merged[t] + acts
    edited = genesis is not None and genesis != base[0].data["population"]
    changes = ([t for t in player if player[t]] + [t for t, _ in drop] + ([live_from] if live_agents else [])
               + ([0] if edited or souls else []))
    calm = min(changes, default=end_tick(base))
    models = {e.mind["model"] for e in base if e.origin == "mind"}
    if not serve and models - {getattr(live, "model", None)}:
        raise ForkLeak(f"a full rerun needs the base log's own mind; the base was run with {sorted(models)}")
    mind = ForkMind(base, live, live_agents, live_from, salt, calm, prior, serve)
    sim = build_sim(base, mind, arm, genesis, workers)
    for aid, text in (souls or {}).items():
        sim.state["agents"][aid]["soul"] = text
    stop = end_tick(base) if until is None else until
    if out:
        os.makedirs(out, exist_ok=True)  # before the run, so an unwritable --out fails before paying
    try:
        sim.run_until(stop, merged)
    except BaseException:
        if out:
            write_jsonl(sim.events, os.path.join(out, "events.partial.jsonl"))
            print(f"twin aborted at tick {sim.state['tick']}; wrote events.partial.jsonl; "
                  f"continue it with: python -m asiimov.fork --resume --out {out}", flush=True)
        raise
    sim.finish()
    early = {a: t for a, t in divergence(base, sim.events).items() if t < calm}
    if early:
        raise ForkLeak(f"agents {sorted(early)[:10]} diverged before the first change at tick {calm}")
    return sim, mind


def replay_twin(events, arm: str | None = None, souls_path: str | None = None,
                souls_sha256: str | None = None) -> Sim:
    """Replay a twin's log without a model, checking every event. A souls= twin's genesis is the base's, so
    its swap lives beside the log in a souls file, whose sha256 the twin's summary records: it replays only
    with that file, and a file whose sha256 doesn't match is refused (Steering, 2026-11-09 10:47Z)."""
    from .minds import ReplayMind
    souls = {}
    if souls_path is not None:
        with open(souls_path, "rb") as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != souls_sha256:
            raise ReplayDivergence(f"{souls_path} has sha256 {hashlib.sha256(blob).hexdigest()}, the twin recorded {souls_sha256}")
        souls = json.loads(blob)
    sim = build_sim(events, ReplayMind([e for e in events if e.origin == "mind"]), arm)
    for aid, text in souls.items():
        sim.state["agents"][int(aid)]["soul"] = text
    sim.expected = events
    sim.run_until(end_tick(events), base_player(events))
    if events[-1].kind == "end":
        sim.finish()
    if len(sim.events) != len(events):
        raise ReplayDivergence(f"replay produced {len(sim.events)} events, log has {len(events)}")
    return sim


def summary(base, twin, live_agents=(), live_from: int = 0) -> dict:
    div = divergence(base, twin)
    calls = live_calls(base, twin, live_agents, live_from)
    n_mind = sum(1 for e in twin if e.origin == "mind")
    return {"twin_mind_calls": n_mind, "live_calls": len(calls),
            "live_by_reason": dict(Counter(r for _, _, r in calls)),
            "agents_live": sorted({a for _, a, _ in calls}),
            "agents_diverged": len(div), "first_divergence_tick": min(div.values(), default=None),
            "diverged": {str(a): t for a, t in sorted(div.items())}}


def main():
    from .minds import NoisyMind, StubMind
    from .run import llm_from_args
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="base log (events.jsonl or .jsonl.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true", help="continue a crashed twin in --out with its own config")
    ap.add_argument("--arm", help="the scorecard arm the base log was run as (e.g. fiction)")
    ap.add_argument("--fire", action="append", default=[], help="add a fire as FAMILY@DAY (repeatable)")
    ap.add_argument("--live", default="", help="comma-separated agent ids forced live")
    ap.add_argument("--live-from-day", type=int, default=0)
    ap.add_argument("--salt", type=int, default=0, help="sampling salt; give each noise twin its own")
    ap.add_argument("--days", type=int, help="run to this day (default: where the base stops)")
    ap.add_argument("--mind", choices=["stub", "noisy", "llm"], default=None,
                    help="default llm; on --resume, the twin's. stub twins are full reruns (nothing served)")
    ap.add_argument("--pod")
    ap.add_argument("--llm-url")
    ap.add_argument("--llm-model")
    ap.add_argument("--llm-key")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=None, help="default 64 for llm, else 1; never changes the log")
    args = ap.parse_args()
    cfg_path = os.path.join(args.out, "fork.json")
    if args.resume:
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        if not args.base:
            raise SystemExit("--base is required")
        if os.path.exists(cfg_path):
            raise SystemExit(f"{args.out} already holds a twin; pass --resume or pick another directory")
        player = defaultdict(list)
        for spec in args.fire:
            fam, day = (int(x) for x in spec.split("@"))
            player[day * TICKS_PER_DAY + 60].append({"kind": "fire", "family": fam})
        with open(args.base, "rb") as f:
            base_sha = hashlib.sha256(f.read()).hexdigest()
        cfg = {"base": os.path.abspath(args.base), "base_sha256": base_sha, "arm": args.arm,
               "player": [[t, a] for t in sorted(player) for a in player[t]],
               "live": [int(x) for x in args.live.split(",") if x.strip()],
               "live_from": args.live_from_day * TICKS_PER_DAY, "salt": args.salt,
               "until": args.days * TICKS_PER_DAY if args.days else None, "mind": args.mind or "llm"}
    args.mind = args.mind or cfg["mind"]
    with open(cfg["base"], "rb") as f:
        if hashlib.sha256(f.read()).hexdigest() != cfg["base_sha256"]:
            raise SystemExit(f"{cfg['base']} changed since the twin started")
    if args.mind != cfg["mind"]:
        raise SystemExit(f"this twin was run with --mind {cfg['mind']}; refusing {args.mind}")
    base = read_log(cfg["base"])
    if args.mind == "llm":
        live = llm_from_args(args)[0]
    else:
        live = StubMind() if args.mind == "stub" else NoisyMind()
    model = getattr(live, "model", None)
    if cfg.setdefault("model", model) != model:
        raise SystemExit(f"this twin was run with model {cfg['model']!r}; refusing {model!r}")
    os.makedirs(args.out, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    partial = os.path.join(args.out, "events.partial.jsonl")
    prior = read_log(partial) if args.resume and os.path.exists(partial) else []
    player = defaultdict(list)
    for t, act in cfg["player"]:
        player[t].append(act)
    sim, mind = run_twin(base, live, args.out, player, cfg["live"], cfg["live_from"], cfg["salt"], cfg["arm"],
                         until=cfg["until"], workers=args.workers or (64 if args.mind == "llm" else 1), prior=prior,
                         serve=args.mind != "stub")
    write_jsonl(sim.events, os.path.join(args.out, "events.jsonl"))
    if os.path.exists(partial):
        os.remove(partial)
    report = summary(base, sim.events, cfg["live"], cfg["live_from"])
    report["served"] = dict(mind.counts)
    if args.mind == "llm":
        report["llm"] = live.stats()
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "diverged"}, indent=2))


if __name__ == "__main__":
    main()
