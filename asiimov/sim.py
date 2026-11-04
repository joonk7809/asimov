"""Deterministic, event-sourced village simulation.

One tick = 10 sim-minutes, 144 ticks per day. A tick has two phases:
1. Rules, agents in id order. An agent that needs its mind (waking, going to sleep,
   hungry with no food) files one request built from the world as it stands, then waits.
2. Commit. The tick's requests go to the mind together (concurrently when workers > 1),
   then each reply is validated and applied in the order `_order` decides. A reply
   takes effect on the tick it was asked, and its mind_* event carries that tick.
Arrival order never matters, so a log is the same whatever the server's latency, and
mind outputs are recorded so a replay reproduces the world bit-for-bit without a model.
A complete log starts with genesis (which names the engine version) and ends with `end`,
whose tick is the first tick not run. `snapshot()`/`restore()` let a world continue
in another process; asiimov.persist uses them to resume runs after a crash.
"""
import hashlib
import json
from collections import defaultdict

from .events import Event, canonical
from .memory import REFLECT_ASK, assemble_prompt, is_bare_value, prompt_hash, render_soul, tokens
from .minds import MindRequest, ReplayDivergence, dispatch
from .params import generate_population
from .rng import Rng

TICKS_PER_DAY = 144
VILLAGE_WAKE_MIN = 390  # 06:30
FOOD_PRICE = 3
FARMER_SELL_PRICE = 2
WORKPLACE = {"farmer": "field", "crafter": "workshop", "trader": "market",
             "healer": "plaza", "guard": "plaza", "child": "plaza", "elder": "plaza"}
ACTIVITIES = ("work", "socialize", "home", "rest", "explore")
ACTIVITY_PLACE = {"socialize": "plaza", "shop": "market", "explore": "woods"}  # work: WORKPLACE; else home
# Bump on any change that alters the log for the same inputs: old logs then refuse to replay
# with a clear message instead of diverging. tests/test_world.py pins a golden log hash per version.
ENGINE_VERSION = "4"
ORDERS = ("id", "shuffle")  # id: low ids go first (the original behavior); shuffle: seeded, per tick
IMPORTANT_WORDS = {"stole": 8, "robbed": 9, "caught": 8, "fire": 9, "refused": 5, "gave": 4, "bought": 2}


def _clamp(v, lo=0, hi=1000):
    return max(lo, min(hi, v))


def _is_int(v) -> bool:
    # JSON true/false parse to bool, which Python treats as int; a mind's hour or id never is one.
    return type(v) is int


def valid_blocks(out: dict | None) -> tuple[list[dict], int]:
    """The plan blocks the world accepts from a plan reply, and how many it drops."""
    blocks, dropped = [], 0
    raw_plan = out.get("plan") if out else None
    for b in raw_plan if isinstance(raw_plan, list) else []:
        ok = (isinstance(b, dict) and b.get("activity") in ACTIVITIES
              and _is_int(b.get("start_h")) and _is_int(b.get("end_h"))
              and 0 <= b["start_h"] < b["end_h"] <= 24)
        if ok:
            blocks.append({"activity": b["activity"], "start_h": b["start_h"], "end_h": b["end_h"]})
        else:
            dropped += 1
    return blocks, dropped


class Sim:
    def __init__(self, seed: int, n_agents: int, mind, genesis: dict | None = None,
                 expected: list[Event] | None = None, order: str = "id", workers: int = 1):
        self._setup(seed, mind, expected, order, workers)
        pop = genesis if genesis is not None else generate_population(seed, n_agents)
        self.state = {"tick": 0, "market_food": 10, "families": pop["families"], "agents": []}
        for p in pop["agents"]:
            a = json.loads(json.dumps(p))
            fam = pop["families"][p["family"]]
            a.update({
                "needs": {"hunger": 200, "fatigue": 600, "loneliness": 300},
                "asleep": True, "slept_at": -TICKS_PER_DAY, "wake_tick": None, "sleep_tick": None,
                "rhythm_event": None,
                "inv": {"food": p["start"]["food"], "coin": p["start"]["coin"]},
                "work_progress": 0, "plan": [], "plan_event": None, "override": None,
                "activity": "sleep", "place": fam["home"], "home": fam["home"],
                "soul": render_soul(p), "core": [], "recent": [], "episodes": [],
                "relationships": {}, "beliefs": {}, "cooldown_until": 0,
                "last_food_loss_event": None, "belief_causes": {}, "habit": None,
            })
            self.state["agents"].append(a)
        self._emit("genesis", "chance", None,
                   {"engine": ENGINE_VERSION, "seed": seed, "order": order, "population": pop}, [])
        for a in self.state["agents"]:
            self._schedule_wake(a, 0, first=True)

    def _setup(self, seed, mind, expected, order, workers):
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}")
        self.seed = seed
        self.rng = Rng(seed)
        self.mind = mind
        self.order = order
        self.workers = workers  # concurrent mind calls per tick; never changes the log
        self.pending: dict[int, tuple] = {}  # agent id -> (request, parents, prompt tokens, apply)
        self.expected = expected
        self.events: list[Event] = []
        self.metrics = {"activity": defaultdict(lambda: defaultdict(int))}

    # ------------------------------------------------------------------ persistence
    def snapshot(self) -> dict:
        """Everything needed to continue this world from the next tick, as JSON. Taken between ticks.
        Serialize it without sort_keys, so a rule that iterates a dict sees the same order after restore."""
        assert not self.pending, "snapshots are taken between ticks"
        return {"engine": ENGINE_VERSION, "seed": self.seed, "order": self.order, "tick": self.state["tick"],
                "n_events": len(self.events), "hash": self.state_hash(), "state": self.state,
                "rng": self.rng.counters,
                "activity": [[d, h, act, n] for (d, h), c in self.metrics["activity"].items() for act, n in c.items()]}

    @classmethod
    def restore(cls, snap: dict, events: list[Event], mind, workers: int = 1,
                expected: list[Event] | None = None) -> "Sim":
        """Continue a world from `snapshot()` and the log it was taken after (its first n_events)."""
        if snap["engine"] != ENGINE_VERSION:
            raise ValueError(f"snapshot is from engine {snap['engine']}, this is {ENGINE_VERSION}")
        if len(events) < snap["n_events"]:
            raise ValueError(f"snapshot follows {snap['n_events']} events, the log has {len(events)}")
        sim = cls.__new__(cls)
        sim._setup(snap["seed"], mind, expected, snap["order"], workers)
        sim.state = json.loads(json.dumps(snap["state"]))
        sim.rng.counters = dict(snap["rng"])
        sim.events = list(events[:snap["n_events"]])
        for d, h, act, n in snap["activity"]:
            sim.metrics["activity"][(d, h)][act] = n
        if sim.state_hash() != snap["hash"]:
            raise ValueError("snapshot state does not match its own hash")
        return sim

    def finish(self) -> int:
        """Mark where the recording stops. Its tick is the first tick not run; its hash is the final state."""
        return self._emit("end", "rule", None, {"hash": self.state_hash()})

    # ------------------------------------------------------------------ plumbing
    def _emit(self, kind, origin, actor, data, parents=(), mind=None) -> int:
        ev = Event(len(self.events), self.state["tick"], kind, origin, actor,
                   json.loads(json.dumps(data)), list(parents), mind)
        if self.expected is not None:
            if ev.id >= len(self.expected):
                raise ReplayDivergence(f"extra event {ev.id}: {ev.to_json()[:200]}")
            exp = self.expected[ev.id].to_json()
            if exp != ev.to_json():
                raise ReplayDivergence(f"event {ev.id} differs\n expected {exp[:400]}\n got      {ev.to_json()[:400]}")
        self.events.append(ev)
        return ev.id

    def state_hash(self) -> str:
        blob = canonical({"state": self.state, "rng": self.rng.counters})
        return hashlib.sha256(blob.encode()).hexdigest()

    def _note(self, aid: int, text: str):
        t = self.state["tick"]
        m = (t % TICKS_PER_DAY) * 10
        self.state["agents"][aid]["recent"].append(f"{m // 60:02d}:{m % 60:02d} {text}")

    def _rel(self, a: dict, other: int) -> dict:
        return a["relationships"].setdefault(str(other), {"affinity": 0, "note": ""})

    def _request(self, a: dict, kind: str, view: dict, parents, apply):
        """File a mind request for this tick. `apply(eid, out)` runs in the commit phase."""
        assert a["id"] not in self.pending, "one mind request per agent per tick"
        prompt, counts = assemble_prompt(a, self.state, kind, self._situation(a, view, kind))
        # The mind gets a copy: whatever it does to the view must not reach world state.
        view = json.loads(json.dumps(view))
        req = MindRequest(kind, a["id"], self.state["tick"], prompt, prompt_hash(prompt), view)
        self.pending[a["id"]] = (req, list(parents), sum(counts.values()), apply)

    def _order(self, ids) -> list[int]:
        """Who goes first when agents contest the same food, target or listener. Decided only here."""
        ids = sorted(ids)
        if self.order == "shuffle":
            for i in range(len(ids) - 1, 0, -1):  # Fisher-Yates on the seeded "order" stream
                _, j = self.rng.draw("order", 0, i)
                ids[i], ids[j] = ids[j], ids[i]
        return ids

    def _commit(self):
        """Ask every pending request at once, then validate and apply replies in `_order`."""
        aids = sorted(self.pending)
        replies = dict(zip(aids, dispatch(self.mind, [self.pending[i][0] for i in aids], self.workers)))
        pending, self.pending = self.pending, {}
        for aid in self._order(aids):
            req, parents, prompt_tokens, apply = pending[aid]
            reply = replies[aid]
            eid = self._emit(f"mind_{req.kind}", "mind", aid, {"kind": req.kind}, parents, mind={
                "model": reply.model, "prompt_hash": req.prompt_hash, "prompt_tokens": prompt_tokens,
                "output_tokens": tokens(reply.raw_output), "sampling": reply.sampling,
                "raw_output": reply.raw_output})
            try:
                out = json.loads(reply.raw_output)
                if not isinstance(out, dict):
                    out = None
            except (ValueError, RecursionError):  # JSONDecodeError, or nesting deep enough to blow the stack
                out = None
            apply(eid, out)

    # ------------------------------------------------------------------ views
    def _hour(self):
        return (self.state["tick"] % TICKS_PER_DAY) * 10 // 60

    def _view(self, a: dict, occupancy: dict) -> dict:
        agents = self.state["agents"]
        fam = self.state["families"][a["family"]]["members"]
        present = [{"id": o, "role": agents[o]["role"], "has_food": agents[o]["inv"]["food"] > 0,
                    "kin": o in fam} for o in occupancy.get(a["place"], []) if o != a["id"]]
        kin = [{"id": o, "has_food": agents[o]["inv"]["food"] > 1} for o in fam if o != a["id"]]
        at_market = a["place"] == "market"
        view = {"role": a["role"], "traits": a["traits"], "hour": self._hour(), "coin": a["inv"]["coin"],
                "food": a["inv"]["food"], "hunger": a["needs"]["hunger"], "food_price": FOOD_PRICE,
                "at_market": at_market, "present": present, "kin": kin, "recent": list(a["recent"])}
        if at_market:
            view["market_food"] = self.state["market_food"]
        return view

    def _situation(self, a: dict, view: dict, kind: str) -> str:
        m = (self.state["tick"] % TICKS_PER_DAY) * 10
        who = ", ".join(f"#{p['id']} ({p['role']})" for p in view["present"]) or "nobody"
        s = (f"Day {self.state['tick'] // TICKS_PER_DAY}, {m // 60:02d}:{m % 60:02d}. You are at {a['place']}. "
             f"Hunger {a['needs']['hunger']}/1000. You carry {a['inv']['food']} food and {a['inv']['coin']} coin. "
             f"Here: {who}.")
        agents = self.state["agents"]
        kin = ", ".join(f"{agents[k['id']]['name']} (#{k['id']})" for k in view["kin"])
        s += f" Your family: {kin or 'none'}."
        if "market_food" in view:
            s += f" The market has {view['market_food']} food at {FOOD_PRICE} coin."
        if kind == "plan":
            day_start = self.state["tick"] - self.state["tick"] % TICKS_PER_DAY
            bed_h = min(24, (a["sleep_tick"] - day_start) * 10 // 60)
            # Here, not in the shared SCHEMA: plan wording there shifted deliberate choices (PR #8 A/B,
            # results/raw/pr8-4b-2d), so deliberate prompts keep main's SCHEMA byte for byte.
            s += (f" You just woke and will be ready for bed around {bed_h:02d}:00."
                  f" Plan your whole day from {m // 60:02d}:00 to {bed_h:02d}:00 in 2 to 6 back-to-back blocks,"
                  ' like {"plan":[{"activity":A1,"start_h":WAKE,"end_h":H1},{"activity":A2,"start_h":H1,'
                  '"end_h":H2},{"activity":A3,"start_h":H2,"end_h":BED}]}.')
        elif kind == "reflect":
            s += REFLECT_ASK
        return s

    # ------------------------------------------------------------------ tick
    def run(self, days: int, player_events: dict | None = None):
        self.run_until(self.state["tick"] + days * TICKS_PER_DAY, player_events)

    def run_until(self, end_tick: int, player_events: dict | None = None):
        player_events = player_events or {}
        while self.state["tick"] < end_tick:
            self.step(player_events.get(self.state["tick"], []))

    def step(self, player_actions=()):
        st = self.state
        tick = st["tick"]
        agents = st["agents"]
        for act in player_actions:
            self._player(act)
        occupancy = defaultdict(list)
        for a in agents:
            occupancy[a["place"]].append(a["id"])

        for a in agents:
            self._body(a, occupancy)
            if a["asleep"]:
                if tick >= a["wake_tick"]:
                    self._wake(a, occupancy)
                continue
            if tick >= a["sleep_tick"] or a["needs"]["fatigue"] >= 1000:
                self._sleep(a, occupancy)
                continue
            self._act(a)
            self._eat(a)
            self._maybe_deliberate(a, occupancy)

        self._commit()
        self._talk()
        for a in agents:
            if not a["asleep"]:
                self.metrics["activity"][(tick // TICKS_PER_DAY, self._hour())][a["activity"]] += 1
        if tick % TICKS_PER_DAY == TICKS_PER_DAY - 1:
            self._emit("checkpoint", "rule", None, {"day": tick // TICKS_PER_DAY, "hash": self.state_hash()})
        st["tick"] = tick + 1

    def _schedule_wake(self, a: dict, after: int, first: bool = False):
        """Pick the next wake tick at the agent's preferred time, after enough sleep."""
        jmax = a["rhythm"]["jitter_max_min"]
        stream = f"jitter:{a['id']}"
        idx, jitter = self.rng.draw(stream, -jmax, jmax)
        wake_min = VILLAGE_WAKE_MIN + a["rhythm"]["chronotype_min"] + jitter
        wake_tick = (after // TICKS_PER_DAY) * TICKS_PER_DAY + wake_min // 10
        min_gap = 0 if first else a["rhythm"]["sleep_need_min"] * 3 // 4 // 10
        while wake_tick < after + min_gap:
            wake_tick += TICKS_PER_DAY
        a["wake_tick"] = wake_tick
        a["rhythm_event"] = self._emit("day_rhythm", "chance", a["id"], {
            "stream": stream, "index": idx, "jitter_min": jitter, "wake_tick": wake_tick})

    def _body(self, a: dict, occupancy: dict):
        n = a["needs"]
        if a["asleep"]:
            n["hunger"] = _clamp(n["hunger"] + a["body"]["metabolism"] // 200)
            n["fatigue"] = _clamp(n["fatigue"] - max(1, 10000 // a["rhythm"]["sleep_need_min"]))
        else:
            n["hunger"] = _clamp(n["hunger"] + a["body"]["metabolism"] // 100)
            n["fatigue"] = _clamp(n["fatigue"] + 6)
            alone = len(occupancy.get(a["place"], [])) <= 1
            n["loneliness"] = _clamp(n["loneliness"] + (a["traits"]["extraversion"] // 150 if alone else -5))

    def _wake(self, a: dict, occupancy: dict):
        a["asleep"] = False
        a["sleep_tick"] = self.state["tick"] + (1440 - a["rhythm"]["sleep_need_min"]) // 10
        a["activity"] = "home"
        self._emit("wake", "rule", a["id"], {"place": a["place"]}, [a["rhythm_event"]])
        self._note(a["id"], "woke up")
        self._request(a, "plan", self._view(a, occupancy), [a["rhythm_event"]],
                      lambda eid, out: self._apply_plan(a, eid, out))

    def _apply_plan(self, a: dict, eid: int, out: dict | None):
        blocks, dropped = valid_blocks(out)
        a["plan"], a["plan_event"] = blocks, eid
        self._emit("plan_receipt", "rule", a["id"], {"ok": out is not None, "blocks": len(blocks),
                                                      "dropped": dropped}, [eid])

    def _sleep(self, a: dict, occupancy: dict):
        """The agent reflects on its day; it falls asleep when the reflection is committed."""
        self._request(a, "reflect", self._view(a, occupancy), [a["rhythm_event"]],
                      lambda eid, out: self._fall_asleep(a, eid, out))

    def _fall_asleep(self, a: dict, eid: int, out: dict | None):
        promoted, rejected = [], 0
        if out and isinstance(out.get("core"), list):
            for c in out["core"][:2]:
                if isinstance(c, str) and c.strip():
                    if is_bare_value(c):
                        rejected += 1
                    else:
                        promoted.append(c.strip()[:200])
        a["core"] = (a["core"] + promoted)[-6:]
        tick = self.state["tick"]
        for r in a["recent"]:
            imp = max([w for k, w in IMPORTANT_WORDS.items() if k in r] or [1])
            a["episodes"].append({"tick": tick, "importance": imp, "text": f"day {tick // TICKS_PER_DAY} {r}"})
        keep = 80 - a["cognition"]["forgetting"] // 20
        a["episodes"] = sorted(a["episodes"], key=lambda e: (-e["importance"], -e["tick"]))[:keep]
        a["episodes"].sort(key=lambda e: e["tick"])
        a["recent"] = []
        # Today's plan ends with today: left in place, the next wake prompt showed it as "Today's plan"
        # and 86% of 4B plans copied it (engine 3).
        a["plan"] = []
        a["asleep"], a["slept_at"] = True, tick
        a["activity"], a["place"], a["override"] = "sleep", a["home"], None
        self._emit("sleep", "rule", a["id"], {"promoted": len(promoted), "rejected": rejected}, [eid])
        self._schedule_wake(a, tick)

    def _act(self, a: dict):
        tick = self.state["tick"]
        hour = self._hour()
        if a["override"] and tick < a["override"]["until"]:
            activity, parent = a["override"]["activity"], a["override"]["event"]
        else:
            a["override"] = None
            if tick % 6 == 0 or a["habit"] is None:
                a["habit"] = self._habit(a)
            activity, parent = a["habit"], a["plan_event"]
            for b in a["plan"]:
                if b["start_h"] <= hour < b["end_h"]:
                    activity = b["activity"]
        place = self._place(a, activity)
        if activity != a["activity"] or place != a["place"]:
            if a["activity"] == "work" and a["role"] == "farmer":
                self._farmer_sells(a)
            a["activity"], a["place"] = activity, place
            self._emit("activity", "rule", a["id"], {"activity": activity, "place": place},
                       [parent] if parent is not None else [])
        if activity == "work" and a["role"] not in ("child", "elder"):
            a["work_progress"] += a["body"]["strength"] * a["body"]["stamina"] // 1000
            if a["work_progress"] >= 3000:
                a["work_progress"] -= 3000
                if a["role"] == "farmer":
                    food = self._food_yield(a)
                    if food:
                        a["inv"]["food"] += food
                        self._emit("produce", "rule", a["id"], {"food": food}, [a["plan_event"]])
                else:
                    a["inv"]["coin"] += 2
                    self._emit("produce", "rule", a["id"], {"coin": 2}, [a["plan_event"]])

    def _place(self, a: dict, activity: str) -> str:
        return WORKPLACE[a["role"]] if activity == "work" else ACTIVITY_PLACE.get(activity, a["home"])

    def _food_yield(self, a: dict) -> int:
        """Food from one completed unit of farm work. A hook so experiment arms can change it."""
        return 1

    def _habit(self, a: dict) -> str:
        """No-LLM habit layer for hours the plan doesn't cover, driven by needs and traits."""
        n, t, w = a["needs"], a["traits"], a["need_weights"]
        if n["fatigue"] > 750:
            return "rest"
        if n["loneliness"] > 600 - t["extraversion"] // 3:
            return "socialize"
        if (a["role"] not in ("child", "elder") and w["wealth"] >= w["belonging"]
                and a["inv"]["coin"] < 10 + w["wealth"] // 5):
            return "work"
        if t["openness"] > 600 and w["novelty"] >= 100:
            return "explore"
        return "home"

    def _farmer_sells(self, a: dict):
        keep = 3 + a["cognition"]["planning_horizon_days"] // 2
        surplus = a["inv"]["food"] - keep
        if surplus > 0:
            a["inv"]["food"] -= surplus
            a["inv"]["coin"] += surplus * FARMER_SELL_PRICE
            self.state["market_food"] += surplus
            self._emit("sell", "rule", a["id"], {"food": surplus}, [a["plan_event"]])

    def _eat(self, a: dict):
        if a["needs"]["hunger"] >= 500 and a["inv"]["food"] > 0:
            a["inv"]["food"] -= 1
            a["needs"]["hunger"] = _clamp(a["needs"]["hunger"] - 450)
            self._emit("eat", "rule", a["id"], {})

    def _maybe_deliberate(self, a: dict, occupancy: dict):
        tick = self.state["tick"]
        trigger = 550 + a["cognition"]["deliberation_threshold"] // 3
        if a["inv"]["food"] > 0 or a["needs"]["hunger"] < trigger or tick < a["cooldown_until"]:
            return
        a["cooldown_until"] = tick + 12
        parents = [a["last_food_loss_event"]] if a["last_food_loss_event"] is not None else []
        self._request(a, "deliberate", self._view(a, occupancy), parents,
                      lambda eid, out: self._resolve(a, out, eid))

    # ------------------------------------------------------------------ actions -> receipts
    def _receipt(self, a, eid, action, ok, why, origin="rule", extra=None):
        data = {"action": action, "ok": ok, "why": why}
        data.update(extra or {})
        return self._emit("receipt", origin, a["id"], data, [eid])

    def _resolve(self, a: dict, out: dict | None, eid: int):
        agents = self.state["agents"]
        action = (out or {}).get("action")
        target = (out or {}).get("target")
        if action == "buy_food":
            if a["place"] != "market":
                return self._receipt(a, eid, action, False, "not at market")
            if self.state["market_food"] <= 0 or a["inv"]["coin"] < FOOD_PRICE:
                return self._receipt(a, eid, action, False, "no stock or no coin")
            self.state["market_food"] -= 1
            a["inv"]["coin"] -= FOOD_PRICE
            a["inv"]["food"] += 1
            self._note(a["id"], "bought food at the market")
            return self._receipt(a, eid, action, True, "bought")
        if action == "go_market":
            a["override"] = {"activity": "shop", "until": self.state["tick"] + 6, "event": eid}
            a["cooldown_until"] = self.state["tick"] + 1
            return self._receipt(a, eid, action, True, "heading to market")
        if action in ("steal", "ask_kin"):
            if target is None:
                return self._receipt(a, eid, action, False, "no target named")
            if not _is_int(target) or not 0 <= target < len(agents) or target == a["id"]:
                return self._receipt(a, eid, action, False, "no such person")
            t = agents[target]
            if action == "steal":
                # Live places, not the tick-start occupancy: the target may already have moved this tick.
                if t["place"] != a["place"]:
                    return self._receipt(a, eid, action, False, "target not here")
                if t["inv"]["food"] <= 0:
                    return self._receipt(a, eid, action, False, "target has no food")
                return self._steal(a, t, eid)
            fam = self.state["families"][a["family"]]["members"]
            if target not in fam:
                return self._receipt(a, eid, action, False, "not kin")
            if t["inv"]["food"] <= 1:
                return self._receipt(a, eid, action, False, "kin has nothing to spare")
            stream = f"kin:{t['id']}"
            idx, roll = self.rng.draw(stream, 0, 999)
            limit = 400 + t["traits"]["agreeableness"] // 2 + self._rel(t, a["id"])["affinity"] // 4
            if roll < limit:
                t["inv"]["food"] -= 1
                a["inv"]["food"] += 1
                self._note(a["id"], f"{t['name']} gave me food")
                self._note(t["id"], f"gave food to {a['name']}")
                r = self._rel(a, t["id"])
                r["affinity"] = min(1000, r["affinity"] + 50)
                return self._receipt(a, eid, action, True, "kin gave food", "chance",
                                     {"stream": stream, "index": idx, "roll": roll, "limit": limit})
            self._note(a["id"], f"{t['name']} refused to share food")
            r = self._rel(a, t["id"])
            r["affinity"] = max(-1000, r["affinity"] - 50)
            return self._receipt(a, eid, action, False, "kin refused", "chance",
                                 {"stream": stream, "index": idx, "roll": roll, "limit": limit})
        if action == "wait":
            return self._receipt(a, eid, action, True, "waited")
        return self._receipt(a, eid, str(action), False, "unparseable or unknown action")

    def _steal(self, a: dict, t: dict, eid: int):
        stream = f"steal:{a['id']}"
        i1, roll = self.rng.draw(stream, 0, 999)
        i2, notice = self.rng.draw(stream, 0, 999)
        success = roll < 700 - t["traits"]["emotionality"] // 3
        noticed = (not success) or notice < 300 + t["traits"]["emotionality"] // 2
        extra = {"stream": stream, "index": [i1, i2], "roll": roll, "notice": notice,
                 "success": success, "noticed": noticed, "victim": t["id"]}
        if success:
            t["inv"]["food"] -= 1
            a["inv"]["food"] += 1
            self._note(a["id"], f"stole food from {t['name']}")
        else:
            self._note(a["id"], f"was caught trying to steal from {t['name']}")
        rid = self._receipt(a, eid, "steal", success, "stolen" if success else "caught", "chance", extra)
        if success:
            t["last_food_loss_event"] = rid
        if noticed:
            rel = self._rel(t, a["id"])
            rel["affinity"] = max(-1000, rel["affinity"] - 400)
            rel["note"] = "stole from me" if success else "tried to rob me"
            rel["cause"] = rid
            t["beliefs"][f"thief:{a['id']}"] = f"{a['name']} is a thief"
            t["belief_causes"][f"thief:{a['id']}"] = rid
            self._note(t["id"], f"{a['name']} {'robbed' if success else 'tried to rob'} me")
        elif success:
            t["beliefs"]["food_missing"] = "someone took my food"
        return rid

    def _talk(self):
        agents = self.state["agents"]
        by_place = defaultdict(list)
        for a in agents:
            if not a["asleep"] and a["place"] not in (a["home"],) and a["activity"] != "work":
                by_place[a["place"]].append(a["id"])
        for place in sorted(by_place):
            ids = by_place[place]
            if len(ids) < 2:
                continue
            for aid in self._order(ids):
                a = agents[aid]
                stream = f"talk:{aid}"
                idx, roll = self.rng.draw(stream, 0, 999)
                if roll >= a["traits"]["extraversion"] // 4:
                    continue
                others = [o for o in ids if o != aid]
                idx2, pick = self.rng.draw(stream, 0, len(others) - 1)
                b = agents[others[pick]]
                grudge = self._rel(b, aid)
                if grudge["affinity"] <= -300:
                    self._note(b["id"], f"turned away from {a['name']}")
                    self._emit("snub", "chance", b["id"], {"stream": stream, "index": [idx, idx2], "with": aid},
                               [grudge["cause"]] if grudge.get("cause") is not None else [])
                    continue
                # gossip: each may pass on what they believe about thieves (credulity decides uptake)
                transfers, sources = [], []
                for x, y in ((a, b), (b, a)):
                    for k in sorted(y["beliefs"]):
                        if k.startswith("thief:") and k not in x["beliefs"] and int(k.split(":")[1]) != x["id"]:
                            c_idx, c_roll = self.rng.draw(f"cred:{x['id']}", 0, 999)
                            if c_roll < x["cognition"]["credulity"]:
                                transfers.append((x, y, k))
                                if y["belief_causes"].get(k) is not None:
                                    sources.append(y["belief_causes"][k])
                tid = self._emit("talk", "chance", aid, {"stream": stream, "index": [idx, idx2], "with": b["id"],
                                                         "place": place, "gossip": [[x["id"], k] for x, _, k in transfers]},
                                 sorted(set(sources)))
                for x, y in ((a, b), (b, a)):
                    x["needs"]["loneliness"] = _clamp(x["needs"]["loneliness"] - 200)
                    r = self._rel(x, y["id"])
                    r["affinity"] = min(1000, r["affinity"] + 20)
                for x, y, k in transfers:
                    x["beliefs"][k] = y["beliefs"][k].split(" (heard")[0] + f" (heard from {y['name']})"
                    x["belief_causes"][k] = tid
                    rr = self._rel(x, int(k.split(":")[1]))
                    rr["affinity"] = max(-1000, rr["affinity"] - 150)
                    rr["cause"] = tid
                self._note(aid, f"talked with {b['name']} at the {place}")
                self._note(b["id"], f"talked with {a['name']} at the {place}")

    # ------------------------------------------------------------------ player
    def _player(self, act: dict):
        if act["kind"] == "fire":
            fam = self.state["families"][act["family"]]
            eid = self._emit("fire", "player", None, {"family": act["family"]})
            for m in fam["members"]:
                a = self.state["agents"][m]
                a["inv"]["food"] = 0
                a["last_food_loss_event"] = eid
                self._note(m, "a fire burned our food stores")
            return eid
        raise ValueError(f"unknown player action {act['kind']}")
