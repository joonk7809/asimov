"""Deterministic, event-sourced village simulation.

One tick = 10 sim-minutes, 144 ticks per day. A tick has two phases:
1. Rules, agents in id order. An agent that needs its mind (waking, going to sleep, or an
   occasion: hungry with no food, or an appraised event salient enough for this agent) files
   one request built from the world as it stands, then waits.
2. Commit. The tick's requests go to the mind together (concurrently when workers > 1),
   then each reply is validated and applied in the order `_order` decides. A reply
   takes effect on the tick it was asked, and its mind_* event carries that tick.
Arrival order never matters, so a log is the same whatever the server's latency, and
mind outputs are recorded so a replay reproduces the world bit-for-bit without a model.
Places sit on a tile map (asiimov/worldmap.py, recorded in genesis). Changing place is a walk that
takes ticks and ends with an `arrive` event; a walker is at no place, so it meets, asks, robs and
talks with nobody until it arrives (`_together`).
A complete log starts with genesis (which names the engine version) and ends with `end`,
whose tick is the first tick not run. `snapshot()`/`restore()` let a world continue
in another process; asiimov.persist uses them to resume runs after a crash.
"""
import hashlib
import json
from collections import defaultdict
from dataclasses import replace

from .events import Event, canonical
from .memory import (REFLECT_ASK, assemble_prompt, is_bare_value, known_people, named_people, prompt_hash,
                     render_soul, shown_relationships, situation_budget, tokens)
from .minds import MindRequest, ReplayDivergence, dispatch
from .params import generate_population
from .rng import Rng
from .worldmap import build_map, position, travel_ticks

TICKS_PER_DAY = 144
VILLAGE_WAKE_MIN = 390  # 06:30
FOOD_PRICE = 3
FARMER_SELL_PRICE = 2
# Food per completed unit of farm work. At 1, 33 farmers made about 100 food a day against about
# 267 meals of need (t1h-4b), so the plain world was a famine from day 2. At 3 the same work
# makes about 300.
FARM_YIELD = 3
# Coin per unit of non-farm work: what a farmer gets for the same unit's food, so a day's work buys
# a day's meals. At 2 (engine 1-4) a 4B guard earned about 3 coin a day against about 8 for meals.
WORK_WAGE = FARM_YIELD * FARMER_SELL_PRICE
MARKET_KEEP_DAYS = 2  # market stock above this many days of village need spoils each night
WORKPLACE = {"farmer": "field", "crafter": "workshop", "trader": "market",
             "healer": "plaza", "guard": "plaza", "child": "plaza", "elder": "plaza"}
# What a plan block can say. "shop" (engine 7) goes to the market and buys on arrival, so buying food
# can be planned: at engine 6 it took a hunger crisis, and 13 of 22 4B guard-days had none.
ACTIVITIES = ("work", "socialize", "home", "rest", "explore", "shop")
ACTIVITY_PLACE = {"socialize": "plaza", "shop": "market", "explore": "woods"}  # work: WORKPLACE; else home
# Intent steps (engine 7). Person steps name a target. seek, give_* and avoid move the agent; ask_kin,
# ask_food, buy_from and steal act on the tick the step starts, where the agent already is.
PERSON_STEPS = ("seek", "avoid", "give_food", "give_coin", "ask_kin", "ask_food", "buy_from", "steal")
STEP_DOS = ACTIVITIES + PERSON_STEPS
INSTANT_STEPS = ("ask_kin", "ask_food", "buy_from", "steal")
RATED_ACTS = INSTANT_STEPS + ("buy_food",)  # player acts that roll or buy: held to the deliberation cooldown
REACH_STEPS = ("seek", "give_food", "give_coin")  # go where the target is, then act when both are there
WAIT_STEPS = REACH_STEPS + ("shop",)  # steps that wait to arrive somewhere, and fail if their time runs out
STEP_TICKS = {"avoid": 24}  # how long a step lasts (activities, avoid) or may take to get done (the rest)
DEFAULT_STEP_TICKS = 18     # 3 hours
MAX_STEPS = 4
# A person who can't or won't give doesn't end an intent whose ask was not its last step: the intent moves
# on (steering, T19 world A: refusals are common by design, and ending on them cut intents spanning a day
# from 0.269 to 0.171). A caught steal or any other failure still ends it.
ASK_STEPS = ("ask_food", "buy_from", "ask_kin")
ASK_NOS = ("nothing to spare", "refused", "kin refused", "kin has nothing to spare")
GIFT_COIN = FOOD_PRICE      # give_coin hands over one meal's price
DAILY_MIND_CALLS = 15       # per agent and calendar day, plans and reflections included
# What a mind reads for a failed buy (engine 9). At engine 8 one bundled "no stock or no coin" left an
# agent holding coin unable to tell why it failed (T16b: 95% of famine agents who had failed to buy went
# back to the empty market).
BUY_FAIL_TEXT = {"no stock": "no food for sale", "no coin": "not enough coin"}
OCCASION_TICKS = 12         # an occasion the mind hasn't been asked about yet fades after 2 hours
OCCASIONS = {  # why a deliberation was asked, as its situation says it
    "hungry_coin": "you are hungry with no food, and carry enough coin to buy some",
    "hungry": "you are hungry and have no food",
    "robbed": "{who} robbed you", "rob_tried": "{who} tried to rob you",
    "food_missing": "some of your food went missing", "fire": "a fire burned your family's food",
    "gift": "{who} gave you {detail}", "refused": "{who} refused to share food with you",
    "failed": "you could not {detail}", "person": "{who}, whom you {detail}, is here",
    "intent_done": "you did what you set out to do: {detail}",
    "blocked": "you had to give up on what you set out to do: {detail}",
}
# What a deliberation may answer, in the order its prompt lists them (engine 10). Only the ones that can
# work now are offered (_offer): with impossible ones listed, 63% of 4B famine answers chose buy_food away
# from the market or facing an empty one (T16e), and cutting them sent minds to people instead (T16f).
DELIBERATE_ACTIONS = ("buy_food", "go_market", "ask_kin", "ask_food", "buy_from", "steal", "give_food", "give_coin",
                      "seek", "avoid", "work", "socialize", "home", "rest", "explore", "wait")
# One factual clause per offered action, no trait or value words (L11); T16e's wording where it exists.
# Deliberate-only wording lives in the situation, not the shared SCHEMA (L11).
ACTION_CLAUSES = {
    "buy_food": "buy_food buys food if you are at the market, carry enough coin and it has some",
    "go_market": "go_market takes you to the market, where you can buy food if you carry enough coin and it has some",
    "ask_kin": "ask_kin asks a family member here for 1 food",  # "here" since the map's place check
    "ask_food": "ask_food asks a person here for 1 food; they may say no",
    "buy_from": f"buy_from offers a person here {FOOD_PRICE} coin for 1 food; they may say no",
    "steal": "steal tries to take 1 food from a person here, and you may be caught",
    "give_food": "give_food hands them 1 food",
    "give_coin": f"give_coin hands them {GIFT_COIN} coin",
    "seek": "seek finds a person and talks with them",
    "avoid": "avoid keeps you away from them",
    "work": "work goes to your work for a few hours",
    "socialize": "socialize spends a few hours with others",
    "home": "home takes you home for a few hours",
    "rest": "rest lasts a few hours",
    "explore": "explore lasts a few hours",
    "wait": "wait does nothing for now",
}
assert set(ACTION_CLAUSES) == set(DELIBERATE_ACTIONS)
NO_COIN_LINE = " You have too little coin to buy food."  # why buy_food, go_market and buy_from are withheld


def deliberate_ask(offer) -> str:
    """The deliberate ask for the offered actions. A shop step buys like go_market, so it goes with it."""
    shop = ", or shop" if "go_market" in offer else ""
    return (" Your action happens now; target is a person's #id or null. "
            + ". ".join(ACTION_CLAUSES[x] for x in offer) + "."
            f" To pursue something over hours or days, set goal and up to 3 more steps in then (these words{shop}),"
            " done in order; for a one-off, leave goal empty and then []."
            " A new goal replaces the intention you have now; to keep it, leave goal empty.")


# Asking anyone here for food (engine 10, T19). Code decides, as for ask_kin, with a seeded draw: the
# asker is given food when the roll falls under this limit (out of 1000). The target must carry at least
# 2 food. Numbers proposed in the PR and checked by steering before any model run.
SHARE_BASE = {"ask_food": 100, "buy_from": 350}  # buy_from: the coin makes a yes likelier
SHARE_KIN = 200                                   # family are likelier to agree (buy_from; ask_food uses ask_kin's rule)
SHARE_MAX = 950
# Bump on any change that alters the log for the same inputs: old logs then refuse to replay
# with a clear message instead of diverging. tests/test_world.py pins a golden log hash per version.
ENGINE_VERSION = "pb-1"  # the player's body (plan/player-body-spec.md), on engine 15; Steering assigns the integer
ORDERS = ("id", "shuffle")  # id: low ids go first (the original behavior); shuffle: a keyed priority per tick
IMPORTANT_WORDS = {"stole": 8, "robbed": 9, "caught": 8, "fire": 9, "refused": 5, "gave": 4, "bought": 2}


RETRY_PENALTY_MILLI = 1000  # frequency_penalty 1.0 on the T21 reflect retry (plan/t21-reflect-fix-spec.md)


def _parse(reply) -> dict | None:
    """A mind reply as the engine reads it: a JSON object, or None."""
    try:
        out = json.loads(reply.raw_output)
    except (ValueError, RecursionError):  # JSONDecodeError, or nesting deep enough to blow the stack
        return None
    return out if isinstance(out, dict) else None


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
                "activity": "sleep", "place": fam["home"], "home": fam["home"], "route": None,
                "soul": render_soul(p), "core": [], "recent": [], "episodes": [],
                "relationships": {}, "beliefs": {}, "cooldown_until": 0,
                "last_food_loss_event": None, "belief_causes": {}, "habit": None,
                "intent": None, "occasion": None, "market_seen": None, "noticed": {"day": -1, "ids": []}, "calls": {"day": -1, "n": 0},
            })
            self.state["agents"].append(a)
        self.map = build_map([f["home"] for f in pop["families"]])
        self._emit("genesis", "chance", None,
                   {"engine": ENGINE_VERSION, "seed": seed, "order": order, "population": pop, "map": self.map}, [])
        for a in self.state["agents"]:
            self._schedule_wake(a, 0, first=True)

    def _setup(self, seed, mind, expected, order, workers):
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}")
        self.seed = seed
        self.rng = Rng(seed)
        self._asks = (-1, {})  # (tick, {ask key: asks so far this tick}), see _ask_roll
        self.mind = mind
        self.order = order
        self.workers = workers  # concurrent mind calls per tick; never changes the log
        self.pending: dict[int, tuple] = {}  # agent id -> (request, parents, prompt tokens, apply)
        self.player_pending: dict[int, tuple] = {}  # agent id -> a player act to apply in _commit (no mind call)
        self._acts: dict[int, list] = {}  # agent id -> this tick's queued player acts: (event, action, target)
        self.expected = expected
        self.events: list[Event] = []
        self.metrics = {"activity": defaultdict(lambda: defaultdict(int))}

    # ------------------------------------------------------------------ persistence
    def snapshot(self) -> dict:
        """Everything needed to continue this world from the next tick, as JSON. Taken between ticks.
        Serialize it without sort_keys, so a rule that iterates a dict sees the same order after restore."""
        assert not self.pending and not self.player_pending, "snapshots are taken between ticks"
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
        sim.map = build_map([f["home"] for f in sim.state["families"]])
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

    def _met(self, a: dict, other: int) -> dict:
        """The relationship, stamped with this tick as the last contact (it ranks the People section)."""
        r = self._rel(a, other)
        r["last"] = self.state["tick"]
        return r

    def _request(self, a: dict, kind: str, view: dict, parents, apply):
        """File a mind request for this tick. `apply(eid, out)` runs in the commit phase."""
        assert a["id"] not in self.pending, "one mind request per agent per tick"
        day = self.state["tick"] // TICKS_PER_DAY
        if a["calls"]["day"] != day:
            a["calls"] = {"day": day, "n": 0}
        a["calls"]["n"] += 1
        actions = None
        if kind == "deliberate":  # what it may answer: only actions that can work now
            actions = view["offer"] = self._offer(a, view)
        situation = self._situation(a, view, kind)
        if actions is not None and actions != self._offer(a, view):  # a crowd cut to nobody shown (rare)
            actions = view["offer"] = self._offer(a, view)
            situation = self._situation(a, view, kind)
        prompt, counts = assemble_prompt(a, self.state, kind, situation, actions=actions)
        # The mind gets a copy: whatever it does to the view must not reach world state.
        view = json.loads(json.dumps(view))
        req = MindRequest(kind, a["id"], self.state["tick"], prompt, prompt_hash(prompt), view)
        self.pending[a["id"]] = (req, list(parents), sum(counts.values()), apply)

    def _order(self, ids) -> list[int]:
        """Who goes first when agents contest the same food, target or listener. Decided only here."""
        ids = sorted(ids)
        if self.order == "shuffle":  # keyed per (tick, id): one more agent doesn't reorder the rest (T22)
            tick = self.state["tick"]
            ids.sort(key=lambda i: (self.rng.raw(f"order:{tick}", i), i))
        return ids

    def _commit(self):
        """Ask every pending request at once, then validate and apply replies in `_order`."""
        aids = sorted(self.pending)
        replies = dict(zip(aids, dispatch(self.mind, [self.pending[i][0] for i in aids], self.workers)))
        # A reflection cut off at max_tokens is asked once more, with a keyed seed (the prompt-hash seed
        # repeats the same loop) and frequency penalty 1.0 (T21: 4B fiction loss 22.7% -> 7.8% in the
        # same-requests check). First asks are unchanged. Asked together, applied in _order below.
        retry = {}
        for aid in aids:
            req = self.pending[aid][0]
            if req.kind == "reflect" and replies[aid].finish_reason == "length":
                day = self.state["tick"] // TICKS_PER_DAY
                retry[aid] = replace(req, attempt=1, seed=self.rng.raw(f"reflect-retry:{aid}:{day}", 0) % 2 ** 32,
                                     frequency_penalty_milli=RETRY_PENALTY_MILLI)
        r_ids = sorted(retry)
        retried = dict(zip(r_ids, dispatch(self.mind, [retry[i] for i in r_ids], self.workers)))
        pending, self.pending = self.pending, {}
        players, self.player_pending = self.player_pending, {}
        assert not set(players) & set(pending), "a possessed body never has a mind request and a player act"
        for aid in self._order(aids + list(players)):
            if aid in players:  # never dispatched to the mind, no mind event: applied from the recorded act
                self._apply_act(self.state["agents"][aid], *players[aid])
                continue
            req, parents, prompt_tokens, apply = pending[aid]
            reply = replies[aid]
            data = {"kind": req.kind}
            if req.kind == "deliberate":  # what the prompt didn't offer, so infeasible choices are countable
                data["withheld"] = [x for x in DELIBERATE_ACTIONS if x not in req.view["offer"]]
            if aid in retried:
                data["retried"] = True  # superseded by the next event; readers of reflections skip it
            eid = self._emit_mind(req, reply, data, parents, prompt_tokens)
            if aid in retried:
                req, reply = retry[aid], retried[aid]
                eid = self._emit_mind(req, reply, {"kind": req.kind, "attempt": 1}, [eid], prompt_tokens)
            apply(eid, _parse(reply))

    def _emit_mind(self, req: MindRequest, reply, data: dict, parents, prompt_tokens: int) -> int:
        return self._emit(f"mind_{req.kind}", "mind", req.agent_id, data, parents, mind={
            "model": reply.model, "prompt_hash": req.prompt_hash, "prompt_tokens": prompt_tokens,
            "output_tokens": tokens(reply.raw_output), "sampling": reply.sampling,
            "finish_reason": reply.finish_reason, "raw_output": reply.raw_output})

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
        """The "# Now" section. Everything but the list of people here comes first; that list then gets
        whatever room the section budget leaves, and the view (so the reply schema) is cut to match.
        Tail-clipping a crowded situation (85+ people in one place) had cut the head off and left the
        schema offering steal targets the prompt no longer showed (PR #15 review)."""
        m = (self.state["tick"] % TICKS_PER_DAY) * 10
        agents = self.state["agents"]
        kin = ", ".join(f"{agents[k['id']]['name']} (#{k['id']})" for k in view["kin"])
        rest = f" Your family: {kin or 'none'}. " + self._economy(a)
        if kind == "deliberate":
            rest += self._market_belief(a)
        if "market_food" in view:
            rest += f" The market has {view['market_food']} food at {FOOD_PRICE} coin."
        if kind == "deliberate":
            if a["inv"]["coin"] < FOOD_PRICE:
                rest += NO_COIN_LINE
            rest += self._occasion_text(a, view.get("occasion")) + deliberate_ask(view["offer"])
        if kind == "plan":
            day_start = self.state["tick"] - self.state["tick"] % TICKS_PER_DAY
            bed_h = min(24, (a["sleep_tick"] - day_start) * 10 // 60)
            # Here, not in the shared SCHEMA: plan wording there shifted deliberate choices (PR #8 A/B,
            # results/raw/pr8-4b-2d), so deliberate prompts keep main's SCHEMA byte for byte.
            rest += (f" You just woke and will be ready for bed around {bed_h:02d}:00."
                     f" Plan your whole day from {m // 60:02d}:00 to {bed_h:02d}:00 in 2 to 6 back-to-back blocks,"
                     ' like {"plan":[{"activity":A1,"start_h":WAKE,"end_h":H1},{"activity":A2,"start_h":H1,'
                     '"end_h":H2},{"activity":A3,"start_h":H2,"end_h":BED}]}.'
                     " A shop block takes you to the market, where you buy the food you need.")
        elif kind == "reflect":
            rest += REFLECT_ASK
        where = f"You are at {a['place']}." if a["place"] else f"You are on your way to the {a['route']['to']}."
        head = (f"Day {self.state['tick'] // TICKS_PER_DAY}, {m // 60:02d}:{m % 60:02d}. {where} "
                f"Hunger {a['needs']['hunger']}/1000. You carry {a['inv']['food']} food and {a['inv']['coin']} coin. "
                f"Here: ")
        room = situation_budget(kind) * 4 - len(f"# Now ({kind})\n") - len(head) - len(rest) - 2
        # Kin first, then the strongest feelings, then id: the people one would notice in a crowd.
        rel = a["relationships"]
        ranked = sorted(view["present"], key=lambda p: (not p["kin"], -abs(rel.get(str(p["id"]), {}).get("affinity", 0)),
                                                        p["id"]))
        shown, size = [], 0
        for i, p in enumerate(ranked):
            more = len(ranked) - i - 1
            size += len(f"#{p['id']} ({p['role']})") + (2 if shown else 0)
            if size + (len(f" and {more} others") if more else 0) > room:
                break
            shown.append(p)
        shown.sort(key=lambda p: p["id"])  # listed in id order, as before
        text = ", ".join(f"#{p['id']} ({p['role']})" for p in shown)
        omitted = len(ranked) - len(shown)
        if omitted:
            text += f" and {omitted} others" if shown else f"{omitted} people"
            view["present"] = shown
        # Whom a deliberation may name: family, the people listed here, and the people the prompt's
        # "People I know" section shows. The reply schema offers exactly these.
        view["people"] = sorted({k["id"] for k in view["kin"]} | {p["id"] for p in view["present"]}
                                | set(shown_relationships(a, self.state)))
        return head + (text or "nobody") + "." + rest

    def _who(self, i) -> str:
        return f"{self.state['agents'][i]['name']} (#{i})"

    def _occasion_text(self, a: dict, occ: dict | None) -> str:
        """Why the mind is asked now, in the deliberate situation. Facts only, no trait or value words."""
        occ = occ or {"why": "hungry"}
        who = self._who(occ["who"]) if occ.get("who") is not None else "someone"
        text = OCCASIONS[occ["why"]].format(who=who, detail=occ.get("detail", ""))
        return f" Why you stop to think: {text}."

    def _offer(self, a: dict, view: dict) -> list[str]:
        """The actions a deliberation offers: those that can work now, by what the agent knows (T16f's F1).
        buy_food only at the market with enough coin and food on show; go_market not without enough coin,
        nor when today's belief or the stall in front of you says it is empty; buy_from only with the coin.
        The situation says why each is withheld: NO_COIN_LINE, the belief line, the market's shown stock,
        or "sold only at the market"."""
        coin_ok = a["inv"]["coin"] >= FOOD_PRICE
        shown = view.get("market_food")
        drop = set()
        if not (view["at_market"] and coin_ok and shown):
            drop.add("buy_food")
        if not coin_ok or self._market_belief(a) or shown == 0:
            drop.add("go_market")
        if not coin_ok:
            drop.add("buy_from")
        # A person action with nobody it may name is not offered either, so the prompt, the reply schema
        # (llm.PERSON_ACTIONS) and `withheld` agree (PR #22 review: 7.6% of famine deliberations).
        if not view["present"]:
            drop |= {"ask_food", "buy_from", "steal"}
        if not any(p["kin"] for p in view["present"]):  # family here (the map's place check)
            drop.add("ask_kin")
        if "people" in view and not view["people"]:  # set by _situation; _request checks the offer again after it
            drop |= {"give_food", "give_coin", "seek", "avoid"}
        return [x for x in DELIBERATE_ACTIONS if x not in drop]

    def _market_belief(self, a: dict) -> str:
        """What the agent last saw of the market today, if it was empty (T16c's Y1). The latest look wins,
        so a market seen empty and then stocked again says nothing. Deliberations only, as A/B'd."""
        seen, tick = a["market_seen"], self.state["tick"]
        if not seen or seen["food"] > 0 or seen["tick"] >= tick or seen["tick"] // TICKS_PER_DAY != tick // TICKS_PER_DAY:
            return ""
        m = (seen["tick"] % TICKS_PER_DAY) * 10
        return f" When you were at the market at {m // 60:02d}:{m % 60:02d}, it had no food."

    def _economy(self, a: dict) -> str:
        """What any villager knows about making a living: what this role's work pays, and where food is
        sold. Not in SCHEMA (L11). At engine 5, 4B planned no work on 39% of non-farm plan-days, hungry
        and broke, because nothing in the prompt said work earns coin."""
        rate = max(1, a["body"]["strength"] * a["body"]["stamina"] // 1000)
        hours = max(1, (-(-3000 // rate) * 10 + 30) // 60)  # ticks per unit of work, in whole hours
        where = WORKPLACE.get(a["role"])
        if a["role"] == "farmer" and self._food_yield(a) == 0:  # only a capped famine arm gets here
            s = f"The {where} will give no more food today."
        elif a["role"] == "farmer":
            s = (f"Working at the {where} yields {self._food_yield(a)} food about every {hours} hours; what you"
                 f" don't keep you sell at {FARMER_SELL_PRICE} coin each.")
        elif where is None or a["role"] in ("child", "elder"):
            s = "You don't work for pay; when you have no food you eat from your family's."
        else:
            s = f"Working at the {where} pays {WORK_WAGE} coin about every {hours} hours."
        # "open at all hours": at engine 6, 4B answered "wait" to 0.195 of deliberations, some citing a
        # closed market, which never exists (T1j).
        return s + f" Food costs {FOOD_PRICE} coin and is sold only at the market, which is open at all hours."

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
        self._acts = {}
        for act in player_actions:
            self._player(act)
        for a in agents:
            if a["route"] and tick >= a["route"]["arrive"]:
                self._arrive(a)
        occupancy = defaultdict(list)  # who has arrived where; walkers are nowhere
        for a in agents:
            if a["place"] is not None:
                occupancy[a["place"]].append(a["id"])

        for a in agents:
            self._body(a, occupancy)
            if a["asleep"]:
                if tick >= a["wake_tick"]:
                    self._wake(a, occupancy)
                self._refuse_acts(a, "asleep")
                continue
            if tick >= a["sleep_tick"] or a["needs"]["fatigue"] >= 1000:
                if a["place"] == a["home"]:
                    self._sleep(a, occupancy)
                    self._refuse_acts(a, "falling asleep")
                else:
                    self._bedtime_walk(a)
                    self._refuse_acts(a, "bedtime")
                continue
            self._act(a)
            self._eat(a)
            if a["id"] in self._acts:
                self._player_deliberate(a, occupancy)
            elif not self._possessed(a):
                self._maybe_deliberate(a, occupancy)

        self._commit()
        self._talk()
        for a in agents:  # what everyone awake at the market saw of its stock as the tick ended (walkers: None)
            if a["place"] == "market" and not a["asleep"]:
                a["market_seen"] = {"tick": tick, "food": st["market_food"]}
        for a in agents:
            if not a["asleep"]:
                self.metrics["activity"][(tick // TICKS_PER_DAY, self._hour())][a["activity"]] += 1
        if tick % TICKS_PER_DAY == TICKS_PER_DAY - 1:
            self._spoil()
            self._emit("checkpoint", "rule", None, {"day": tick // TICKS_PER_DAY, "hash": self.state_hash()})
        st["tick"] = tick + 1

    # ------------------------------------------------------------------ map
    def _dest(self, a: dict) -> str:
        """Where the agent is, or where it is walking to."""
        return a["route"]["to"] if a["route"] else a["place"]

    def _pos(self, a: dict) -> list[int]:
        places = self.map["places"]
        if not a["route"]:
            return places[a["place"]]["anchor"]
        return position(a["route"], places[a["route"]["to"]]["anchor"], self.state["tick"])

    def _together(self, a: dict, b: dict) -> bool:
        """Both have arrived at one place. Two walkers are never together. Every person action, talk and
        notice reads presence through here or through the tick's occupancy, which holds arrivals only."""
        return a["place"] is not None and a["place"] == b["place"]

    def _set_activity(self, a: dict, activity: str, place: str, parents):
        """Start `activity` at `place`: there at once if the agent is already there or headed there, else a
        walk from where it is now (which turns a walker around mid-route). The event's `arrive` is the tick
        it gets there."""
        tick = self.state["tick"]
        if place != self._dest(a):
            frm = self._pos(a)
            arrive = tick + travel_ticks(frm, self.map["places"][place]["anchor"])
            a["place"], a["route"] = (place, None) if arrive == tick else \
                (None, {"from": frm, "to": place, "depart": tick, "arrive": arrive, "event": None})
        a["activity"] = activity
        data = {"activity": activity, "place": place, "arrive": a["route"]["arrive"] if a["route"] else tick}
        new_route = a["route"] is not None and a["route"]["depart"] == tick
        if new_route:  # where the walk starts, so a reader of the log can place the walker at any tick
            data["from"] = a["route"]["from"]
        eid = self._emit("activity", "rule", a["id"], data, parents)
        if new_route:
            a["route"]["event"] = eid

    def _bedtime_walk(self, a: dict):
        """Bedtime away from home: the world walks the agent home, and it eats on the way. It reflects and
        falls asleep when it gets there. No plan, step or habit authors these ticks."""
        if self._dest(a) != a["home"]:
            self._set_activity(a, "home", a["home"], [a["rhythm_event"]])
        self._eat(a)

    def _arrive(self, a: dict):
        """At the start of the arrival tick, before anyone acts. A planned shop block buys on arrival."""
        r = a["route"]
        a["place"], a["route"] = r["to"], None
        self._emit("arrive", "rule", a["id"], {"place": a["place"]}, [r["event"]])
        if a["activity"] == "shop" and a["place"] == "market" and self._running(a)[1] is None:
            self._buy(a, a["plan_event"])  # the receipt cites the plan, as a shop block's always has

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
        promoted, rejected, unknown = [], 0, 0
        known = known_people(a, self.state)  # before this reflection's day is folded into episodes
        if out and isinstance(out.get("core"), list):
            for c in out["core"][:2]:
                if isinstance(c, str) and c.strip():
                    if is_bare_value(c):
                        rejected += 1
                    elif any(not g & known for g in named_people(c, self.state["agents"], a["id"])):
                        unknown += 1  # names someone the agent's own record never mentions
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
        self._emit("sleep", "rule", a["id"], {"promoted": len(promoted), "rejected": rejected,
                                                 "unknown_person": unknown}, [eid])
        self._schedule_wake(a, tick)

    def _running(self, a: dict):
        """The step authoring this tick, if any: (author, step, holder). Precedence, after the world's forced
        wake tick: a deliberated one-off (the override), then the current step of an intent; the plan
        block and then the habit layer author only when neither runs. asiimov.authorship mirrors this."""
        if a["override"]:
            return "deliberate", a["override"], a["override"]
        if a["intent"]:
            return "intent", a["intent"]["steps"][a["intent"]["i"]], a["intent"]
        return None, None, None

    def _step_activity(self, a: dict, step: dict) -> tuple[str, str]:
        """What a step has the agent do, and where. Reads where its target is now (L2); no side effects."""
        do, t = step["do"], step["target"]
        if do in INSTANT_STEPS:  # acts where the agent already is, or is headed
            return a["activity"], self._dest(a)
        if do in REACH_STEPS:  # where the target is, or is walking to
            return "visit", self._dest(self.state["agents"][t])
        if do == "avoid":
            return ("explore", ACTIVITY_PLACE["explore"]) if self._dest(self.state["agents"][t]) == a["home"] \
                else ("home", a["home"])
        return do, self._place(a, do)

    def _act(self, a: dict):
        hour = self._hour()
        author, step, holder = self._running(a)
        if step is None:
            if self.state["tick"] % 6 == 0 or a["habit"] is None:
                a["habit"] = self._habit(a)
            activity, parent = a["habit"], a["plan_event"]
            for b in a["plan"]:
                if b["start_h"] <= hour < b["end_h"]:
                    activity = b["activity"]
            place = self._place(a, activity)
        else:
            (activity, place), parent = self._step_activity(a, step), holder["event"]
        if activity != a["activity"] or place != self._dest(a):
            self._set_activity(a, activity, place, [parent] if parent is not None else [])
            if activity == "shop" and step is None and a["place"] == "market":  # a planned shop block buys
                self._buy(a, parent)                                             # here, or on arrival
        if activity == "work" and a["role"] not in ("child", "elder") and a["place"] == WORKPLACE[a["role"]]:
            a["work_progress"] += a["body"]["strength"] * a["body"]["stamina"] // 1000
            if a["work_progress"] >= 3000:
                a["work_progress"] -= 3000
                if a["role"] == "farmer":
                    food = self._harvest(a)
                    if food:
                        a["inv"]["food"] += food
                        self._emit("produce", "rule", a["id"], {"food": food}, [a["plan_event"]])
                        # Sold as it's made: selling only on leaving work missed every farmer who
                        # worked until bed, so 30-40 of about 100 a day reached the market.
                        self._farmer_sells(a)
                else:
                    a["inv"]["coin"] += WORK_WAGE
                    self._emit("produce", "rule", a["id"], {"coin": WORK_WAGE}, [a["plan_event"]])
        if step is not None:
            self._progress(a, author, step, holder)

    def _progress(self, a: dict, author: str, step: dict, holder: dict):
        """Advance the running step after the agent has acted this tick."""
        do, t = step["do"], step["target"]
        if do == "shop" and a["place"] == "market":
            return self._end_step(a, author, *self._buy(a, holder["event"]))
        if do in INSTANT_STEPS:
            rid = self._person_act(a, do, t, holder["event"])
            return self._end_step(a, author, self.events[rid].data["ok"], self.events[rid].data["why"])
        if do in REACH_STEPS:
            b = self.state["agents"][t]
            if self._together(a, b) and not b["asleep"]:
                if do == "seek":
                    self._meet(a, b, holder["event"])
                    return self._end_step(a, author, True, "met")
                rid = self._give(a, b, do, holder["event"])
                return self._end_step(a, author, self.events[rid].data["ok"], self.events[rid].data["why"])
        holder["left"] -= 1
        if holder["left"] <= 0:  # activities and avoid are done when their time is up; the rest ran out of it
            self._end_step(a, author, do not in WAIT_STEPS, "time ran out" if do in WAIT_STEPS else "done")

    def _end_step(self, a: dict, author: str, ok: bool, why: str):
        if author == "deliberate":
            a["override"] = None
            return
        it = a["intent"]
        c = a["traits"]["conscientiousness"]
        soft = not ok and it["steps"][it["i"]]["do"] in ASK_STEPS and why in ASK_NOS and it["i"] + 1 < len(it["steps"])
        if soft:  # a "no" moves on to the next step; a refusal has already fired its own occasion
            self._emit("intent_step", "rule", a["id"], {"i": it["i"], "do": it["steps"][it["i"]]["do"], "why": why},
                       [it["event"]])
            it["i"] += 1
            it["left"] = STEP_TICKS.get(it["steps"][it["i"]]["do"], DEFAULT_STEP_TICKS)
            return
        if not ok:
            self._emit("intent_failed", "rule", a["id"], {"i": it["i"], "do": it["steps"][it["i"]]["do"], "why": why},
                       [it["event"]])
            a["intent"] = None
            reason = f" ({BUY_FAIL_TEXT[why]})" if why in BUY_FAIL_TEXT else ""  # a shop step that couldn't buy
            return self._occasion(a, "blocked", 450 + c // 4, detail=it["goal"] + reason)
        self._emit("intent_step", "rule", a["id"], {"i": it["i"], "do": it["steps"][it["i"]]["do"]}, [it["event"]])
        it["i"] += 1
        if it["i"] == len(it["steps"]):
            self._emit("intent_done", "rule", a["id"], {"steps": it["i"]}, [it["event"]])
            a["intent"] = None
            return self._occasion(a, "intent_done", 350 + c // 4, detail=it["goal"])
        # The next step starts on its own. A finished step is not an occasion: at 4B, waking the mind
        # there replaced 101 of 117 replaced intents with a new goal (t15-accept-4b-2; steering ruling).
        it["left"] = STEP_TICKS.get(it["steps"][it["i"]]["do"], DEFAULT_STEP_TICKS)

    def _place(self, a: dict, activity: str) -> str:
        return WORKPLACE[a["role"]] if activity == "work" else ACTIVITY_PLACE.get(activity, a["home"])

    def _harvest(self, a: dict) -> int:
        """Food actually made by one completed unit of farm work. A hook so an arm can cap the day's
        total; the prompt reads the per-unit _food_yield, which has no side effects."""
        return self._food_yield(a)

    def _food_yield(self, a: dict) -> int:
        """Food from one completed unit of farm work. A hook so experiment arms can change it."""
        return FARM_YIELD

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

    def _market_cap(self) -> int:
        """MARKET_KEEP_DAYS of the village's meals, from each agent's metabolism and sleep."""
        rise = sum((1440 - a["rhythm"]["sleep_need_min"]) // 10 * (a["body"]["metabolism"] // 100)
                   + a["rhythm"]["sleep_need_min"] // 10 * (a["body"]["metabolism"] // 200)
                   for a in self.state["agents"])
        return MARKET_KEEP_DAYS * rise // 450

    def _spoil(self):
        """Food left on the stall past a cap rots overnight, so a fed village still needs its farmers."""
        cap = self._market_cap()
        if self.state["market_food"] > cap:
            lost = self.state["market_food"] - cap
            self.state["market_food"] = cap
            self._emit("spoil", "rule", None, {"market_food_lost": lost, "cap": cap})

    def _meals_left_today(self, a: dict) -> int:
        """Meals _eat will take before this agent's bedtime, counting the one due now (at least 1).
        buy_food buys this many, so one trip to the market feeds the rest of the day."""
        rise = max(0, a["sleep_tick"] - self.state["tick"]) * (a["body"]["metabolism"] // 100)
        h = a["needs"]["hunger"] + rise
        return max(1, (h - 500) // 450 + 1) if h >= 500 else 1

    def _eat(self, a: dict):
        if a["needs"]["hunger"] < 500:
            return
        if a["inv"]["food"] > 0:
            a["inv"]["food"] -= 1
            a["needs"]["hunger"] = _clamp(a["needs"]["hunger"] - 450)
            self._emit("eat", "rule", a["id"], {})
        elif a["role"] in ("child", "elder"):
            # Dependants eat from the household's stock: the family member carrying the most food,
            # ties by id, as long as they keep one for themselves.
            fam = self.state["families"][a["family"]]["members"]
            donor = max((m for m in fam if m != a["id"]), default=None,
                        key=lambda m: (self.state["agents"][m]["inv"]["food"], -m))
            if donor is None or self.state["agents"][donor]["inv"]["food"] <= 1:
                return
            d = self.state["agents"][donor]
            d["inv"]["food"] -= 1
            a["needs"]["hunger"] = _clamp(a["needs"]["hunger"] - 450)
            self._emit("eat", "rule", a["id"], {"household": donor})
            self._note(a["id"], f"ate from {d['name']}'s food")
            self._note(donor, f"{a['name']} ate from my food")

    # ------------------------------------------------------------------ appraisal
    def _occasion(self, a: dict, why: str, salience: int, who: int | None = None, event: int | None = None,
                  detail: str = ""):
        """Something happened to `a` that may be worth thinking about. Code appraises it: it can wake the
        mind only if its salience reaches the agent's deliberation_threshold, and a pending occasion keeps
        its place unless this one is more salient or the pending one has faded.

        A refusal always reaches the mind (T19): it is what makes a hungry agent weigh other means. It
        fades only OCCASION_TICKS after the agent may next deliberate, since an ask made in a deliberation
        is refused during that deliberation's cooldown, and it outranks hunger then (_maybe_deliberate)."""
        tick, cur = self.state["tick"], a["occasion"]
        if why == "refused":
            salience, tick = max(salience, a["cognition"]["deliberation_threshold"]), max(tick, a["cooldown_until"])
        if salience < a["cognition"]["deliberation_threshold"]:
            return
        if cur and self.state["tick"] - cur["tick"] < OCCASION_TICKS and cur["salience"] >= salience:
            return
        a["occasion"] = {"why": why, "salience": salience, "who": who, "event": event, "detail": detail,
                         "tick": tick}

    def _notice(self, a: dict, occupancy: dict) -> dict | None:
        """Someone the agent feels strongly about is here and awake, noticed at most once per person per
        day. Strength is |affinity|; family counts as at least 200 + 2 * the kin_care need weight."""
        day = self.state["tick"] // TICKS_PER_DAY
        if a["noticed"]["day"] != day:
            a["noticed"] = {"day": day, "ids": []}
        t, fam, best = a["traits"], self.state["families"][a["family"]]["members"], None
        for o in occupancy.get(a["place"], []):  # id order
            if o == a["id"] or o in a["noticed"]["ids"] or self.state["agents"][o]["asleep"]:
                continue
            aff = a["relationships"].get(str(o), {}).get("affinity", 0)
            strength = max(abs(aff), 200 + 2 * a["need_weights"]["kin_care"] if o in fam else 0)
            if strength < 200:
                continue
            s = strength // 2 + (t["extraversion"] if aff >= 0 else t["emotionality"]) // 4
            if s >= a["cognition"]["deliberation_threshold"] and (best is None or s > best[0]):
                best = (s, o, aff)
        if best is None:
            return None
        s, o, aff = best
        a["noticed"]["ids"].append(o)
        feel = ("are fond of" if aff >= 600 else "like" if aff >= 200 else "can't stand" if aff <= -600
                else "dislike" if aff <= -200 else "care about")  # the last: family, no strong feeling yet
        return {"why": "person", "salience": s, "who": o, "event": a["relationships"].get(str(o), {}).get("cause"),
                "detail": feel, "tick": self.state["tick"]}

    def _maybe_deliberate(self, a: dict, occupancy: dict):
        """Ask the mind when there is an occasion: hunger with no food (always), or an appraised one."""
        tick = self.state["tick"]
        calls = a["calls"]["n"] if a["calls"]["day"] == tick // TICKS_PER_DAY else 0
        # Room is kept for this day's reflection and, after a late bedtime, a plan the same calendar day.
        if tick < a["cooldown_until"] or calls >= DAILY_MIND_CALLS - 2:
            return
        trigger = 550 + a["cognition"]["deliberation_threshold"] // 3
        pending = a["occasion"] if a["occasion"] and tick - a["occasion"]["tick"] < OCCASION_TICKS else None
        refused = pending is not None and pending["why"] == "refused"
        if a["inv"]["food"] == 0 and a["needs"]["hunger"] >= trigger and not refused:
            occ = {"why": "hungry_coin" if a["inv"]["coin"] >= FOOD_PRICE else "hungry", "who": None,
                   "event": a["last_food_loss_event"]}
        else:
            occ = pending or self._notice(a, occupancy)
            if occ is None:
                a["occasion"] = None
                return
            a["occasion"] = None
        a["cooldown_until"] = tick + 12
        view = self._view(a, occupancy)
        view["occasion"] = {k: occ.get(k) for k in ("why", "who", "detail")}
        self._request(a, "deliberate", view, [occ["event"]] if occ["event"] is not None else [],
                      lambda eid, out: self._resolve(a, out, eid))

    # ------------------------------------------------------------------ actions -> receipts
    def _receipt(self, a, eid, action, ok, why, origin="rule", extra=None):
        data = {"action": action, "ok": ok, "why": why}
        data.update(extra or {})
        rid = self._emit("receipt", origin, a["id"], data, [eid])
        if not ok and why not in ("kin refused", "refused", "caught"):  # those have occasions of their own
            self._occasion(a, "failed", 300 + a["traits"]["conscientiousness"] // 3, event=rid,
                           detail=f"{action.replace('_', ' ')} ({BUY_FAIL_TEXT.get(why, why)})")
        return rid

    def _buy(self, a: dict, eid: int) -> tuple[bool, str]:
        """Buy at the market: the meals left today that the agent doesn't carry, and a working adult
        also what the family's children and elders still lack, which they then eat from the household
        (see _eat). Capped by coin and stock."""
        if a["place"] != "market":
            self._receipt(a, eid, "buy_food", False, "not at market")
            return False, "not at market"
        if self.state["market_food"] <= 0 or a["inv"]["coin"] < FOOD_PRICE:
            why = "no stock" if self.state["market_food"] <= 0 else "no coin"  # both: the stock is what failed
            if why == "no stock":
                a["market_seen"] = {"tick": self.state["tick"], "food": 0}
            self._receipt(a, eid, "buy_food", False, why)
            return False, why
        want = max(0, self._meals_left_today(a) - a["inv"]["food"])
        if a["role"] not in ("child", "elder"):
            for m in self.state["families"][a["family"]]["members"]:
                d = self.state["agents"][m]
                if m != a["id"] and d["role"] in ("child", "elder") and d["sleep_tick"] is not None:
                    want += max(0, self._meals_left_today(d) - d["inv"]["food"])
        n = min(want, a["inv"]["coin"] // FOOD_PRICE, self.state["market_food"])
        self.state["market_food"] -= n
        a["inv"]["coin"] -= n * FOOD_PRICE
        a["inv"]["food"] += n
        if n:
            self._note(a["id"], "bought food at the market")
        self._receipt(a, eid, "buy_food", True, "bought" if n else "had enough", extra={"food": n})
        return True, "bought"

    def _target_error(self, a: dict, target) -> str | None:
        if target is None:
            return "no target named"
        if not _is_int(target) or not 0 <= target < len(self.state["agents"]) or target == a["id"]:
            return "no such person"
        return None

    def _steps(self, a: dict, raw) -> tuple[list[dict], int]:
        """The intent steps the world accepts from a reply's `then`, and how many it drops."""
        steps, dropped = [], 0
        for st in raw if isinstance(raw, list) else []:
            do = st.get("do") if isinstance(st, dict) else None
            target = st.get("target") if isinstance(st, dict) else None
            if do not in STEP_DOS:
                dropped += 1
            elif do in PERSON_STEPS and self._target_error(a, target):
                dropped += 1
            else:
                steps.append({"do": do, "target": target if do in PERSON_STEPS else None})
        return steps, dropped

    def _resolve(self, a: dict, out: dict | None, eid: int):
        """Apply a deliberation: the action happens now; with a goal, it and the `then` steps become an
        intent that persists until done, failed or replaced."""
        out = out or {}
        action, target = out.get("action"), out.get("target")
        first = None
        if action == "buy_food":
            self._buy(a, eid)
        elif action in INSTANT_STEPS:
            err = self._target_error(a, target)
            if err:
                self._receipt(a, eid, action, False, err)
            else:
                self._person_act(a, action, target, eid)
        elif action == "wait":
            self._receipt(a, eid, action, True, "waited")
        elif action == "go_market" or action in STEP_DOS:
            do = "shop" if action == "go_market" else action
            err = self._target_error(a, target) if do in PERSON_STEPS else None
            if err:
                self._receipt(a, eid, action, False, err)
            else:
                first = {"do": do, "target": target if do in PERSON_STEPS else None}
                self._receipt(a, eid, action, True, "heading to market" if do == "shop" else "started")
        else:
            self._receipt(a, eid, str(action), False, "unparseable or unknown action")
        goal = out.get("goal").strip()[:120] if isinstance(out.get("goal"), str) else ""
        then, dropped = self._steps(a, out.get("then"))
        steps = ([first] if first else []) + then
        if goal and steps:
            if a["intent"]:
                self._emit("intent_dropped", "rule", a["id"], {"i": a["intent"]["i"], "why": "replaced"},
                           [a["intent"]["event"]])
            steps = steps[:MAX_STEPS]
            iid = self._emit("intent_set", "rule", a["id"], {"goal": goal, "steps": steps, "dropped": dropped}, [eid])
            a["override"] = None
            a["intent"] = {"goal": goal, "steps": steps, "i": 0, "event": iid, "tick": self.state["tick"],
                           "left": STEP_TICKS.get(steps[0]["do"], DEFAULT_STEP_TICKS)}
        elif first:
            a["override"] = {**first, "event": eid, "left": STEP_TICKS.get(first["do"], DEFAULT_STEP_TICKS)}

    def _person_act(self, a: dict, action: str, target: int, eid: int) -> int:
        """ask_kin, ask_food, buy_from or steal, with a valid target id. Returns the receipt's event id."""
        t = self.state["agents"][target]
        if action in SHARE_BASE:
            return self._ask_share(a, t, action, eid)
        if action == "steal":
            # Live places, not the tick-start occupancy: the target may already have moved this tick.
            if not self._together(a, t):
                return self._receipt(a, eid, action, False, "target not here")
            if t["inv"]["food"] <= 0:
                return self._receipt(a, eid, action, False, "target has no food")
            return self._steal(a, t, eid)
        fam = self.state["families"][a["family"]]["members"]
        if target not in fam:
            return self._receipt(a, eid, action, False, "not kin")
        if not self._together(a, t):  # the map's place check (plan/t6-map-spec.md section 5)
            return self._receipt(a, eid, action, False, "target not here", extra={"to": t["id"]})
        if t["asleep"]:
            return self._receipt(a, eid, action, False, "target asleep", extra={"to": t["id"]})
        if t["inv"]["food"] <= 1:
            return self._receipt(a, eid, action, False, "kin has nothing to spare")
        stream, idx, roll = self._ask_roll(f"kin:{t['id']}:{a['id']}")
        limit = self._kin_limit(t, a)
        if roll < limit:
            t["inv"]["food"] -= 1
            a["inv"]["food"] += 1
            self._note(a["id"], f"{t['name']} gave me food")
            self._note(t["id"], f"gave food to {a['name']}")
            r = self._met(a, t["id"])
            r["affinity"] = min(1000, r["affinity"] + 50)
            return self._receipt(a, eid, action, True, "kin gave food", "chance",
                                 {"stream": stream, "index": idx, "roll": roll, "limit": limit, "to": t["id"]})
        self._note(a["id"], f"{t['name']} refused to share food")
        r = self._met(a, t["id"])
        r["affinity"] = max(-1000, r["affinity"] - 50)
        rid = self._receipt(a, eid, action, False, "kin refused", "chance",
                            {"stream": stream, "index": idx, "roll": roll, "limit": limit, "to": t["id"]})
        self._occasion(a, "refused", 450 + (1000 - a["traits"]["agreeableness"]) // 4, t["id"], rid)
        return rid

    def _ask_roll(self, key: str) -> tuple[str, int, int]:
        """The roll for an ask, keyed by (target, asker, tick) so other askers of the target don't shift
        it (T22), plus how often this asker already asked this target this tick: a refused ask can
        wake the mind, which may ask again in the same tick, and must get a fresh roll (engine 12).
        Only the asker's own asks move that count. Snapshots fall on tick starts, where it is empty."""
        tick = self.state["tick"]
        if self._asks[0] != tick:
            self._asks = (tick, {})
        n = self._asks[1].get(key, 0)
        self._asks[1][key] = n + 1
        stream = key if n == 0 else f"{key}:{n}"
        return stream, tick, self.rng.at(stream, tick, 0, 999)

    def _kin_limit(self, t: dict, a: dict) -> int:
        """How likely family member `t` is to give `a` food when asked, out of 1000 (ask_kin's rule)."""
        return 400 + t["traits"]["agreeableness"] // 2 + self._rel(t, a["id"])["affinity"] // 4

    def _ask_share(self, a: dict, t: dict, action: str, eid: int) -> int:
        """ask_food or buy_from: ask a person here, kin or not, for 1 food (buy_from pays FOOD_PRICE coin
        for it). Code decides with a seeded draw logged on the receipt; a refusal is the "refused" occasion."""
        to = {"to": t["id"]}
        if not self._together(a, t):  # live places (L2)
            return self._receipt(a, eid, action, False, "target not here", extra=to)
        if t["asleep"]:
            return self._receipt(a, eid, action, False, "target asleep", extra=to)
        if action == "buy_from" and a["inv"]["coin"] < FOOD_PRICE:
            return self._receipt(a, eid, action, False, "no coin", extra=to)
        if t["inv"]["food"] <= 1:  # they keep one for themselves
            return self._receipt(a, eid, action, False, "nothing to spare", extra=to)
        kin = t["id"] in self.state["families"][a["family"]]["members"]
        stream, idx, roll = self._ask_roll(f"share:{t['id']}:{a['id']}")
        if action == "ask_food" and kin:  # the same odds as ask_kin for the same ask (steering, T19)
            limit = self._kin_limit(t, a)
        else:
            limit = _clamp(SHARE_BASE[action] + t["traits"]["agreeableness"] // 4
                           + self._rel(t, a["id"])["affinity"] // 4 + (SHARE_KIN if kin else 0), 0, SHARE_MAX)
        extra = {"stream": stream, "index": idx, "roll": roll, "limit": limit, "kin": kin, **to}
        r = self._met(a, t["id"])
        self._met(t, a["id"])
        if roll < limit:
            t["inv"]["food"] -= 1
            a["inv"]["food"] += 1
            if action == "buy_from":
                a["inv"]["coin"] -= FOOD_PRICE
                t["inv"]["coin"] += FOOD_PRICE
                self._note(a["id"], f"bought 1 food from {t['name']} for {FOOD_PRICE} coin")
                self._note(t["id"], f"sold 1 food to {a['name']} for {FOOD_PRICE} coin")
            else:
                self._note(a["id"], f"{t['name']} gave me 1 food when I asked")
                self._note(t["id"], f"gave 1 food to {a['name']}, who asked")
            r["affinity"] = min(1000, r["affinity"] + 50)
            return self._receipt(a, eid, action, True, "sold" if action == "buy_from" else "gave", "chance", extra)
        self._note(a["id"], f"{t['name']} refused to {'sell me' if action == 'buy_from' else 'give me'} food")
        r["affinity"] = max(-1000, r["affinity"] - 50)
        rid = self._receipt(a, eid, action, False, "refused", "chance", extra)
        self._occasion(a, "refused", 450 + (1000 - a["traits"]["agreeableness"]) // 4, t["id"], rid)
        return rid

    def _meet(self, a: dict, b: dict, eid: int):
        """A sought-out meeting: like a talk, but chosen, so it always happens once both are there."""
        self._emit("meet", "rule", a["id"], {"with": b["id"], "place": a["place"]}, [eid])
        for x, y in ((a, b), (b, a)):
            x["needs"]["loneliness"] = _clamp(x["needs"]["loneliness"] - 200)
            r = self._met(x, y["id"])
            r["affinity"] = min(1000, r["affinity"] + 20)
        self._note(a["id"], f"went to see {b['name']} at the {a['place']}")
        self._note(b["id"], f"{a['name']} came to see me at the {a['place']}")

    def _give(self, a: dict, b: dict, do: str, eid: int) -> int:
        """A gift: a transfer with a receipt. The receiver warms to the giver and may stop to think about it."""
        what, n = ("food", 1) if do == "give_food" else ("coin", GIFT_COIN)
        if a["inv"][what] < n:
            return self._receipt(a, eid, do, False, f"no {what} to give", extra={"to": b["id"]})
        a["inv"][what] -= n
        b["inv"][what] += n
        rid = self._receipt(a, eid, do, True, "gave", extra={"to": b["id"], what: n})
        r = self._met(b, a["id"])
        r["affinity"] = min(1000, r["affinity"] + 100)
        r["cause"] = rid
        self._met(a, b["id"])
        self._note(a["id"], f"gave {n} {what} to {b['name']}")
        self._note(b["id"], f"{a['name']} gave me {n} {what}")
        self._occasion(b, "gift", 400 + b["traits"]["agreeableness"] // 4, a["id"], rid, f"{n} {what}")
        return rid

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
            rel = self._met(t, a["id"])
            rel["affinity"] = max(-1000, rel["affinity"] - 400)
            rel["note"] = "stole from me" if success else "tried to rob me"
            rel["cause"] = rid
            t["beliefs"][f"thief:{a['id']}"] = f"{a['name']} is a thief"
            t["belief_causes"][f"thief:{a['id']}"] = rid
            self._note(t["id"], f"{a['name']} {'robbed' if success else 'tried to rob'} me")
            self._occasion(t, "robbed" if success else "rob_tried", 600 + t["traits"]["emotionality"] // 3, a["id"], rid)
        elif success:
            t["beliefs"]["food_missing"] = "someone took my food"
            self._occasion(t, "food_missing", 300 + t["traits"]["emotionality"] // 4, event=rid)
        return rid

    def _talk(self):
        agents = self.state["agents"]
        by_place = defaultdict(list)
        for a in agents:
            if not a["asleep"] and a["place"] not in (a["home"], None) and a["activity"] != "work":
                by_place[a["place"]].append(a["id"])
        for place in sorted(by_place):
            ids = by_place[place]
            if len(ids) < 2:
                continue
            tick = self.state["tick"]
            for aid in self._order(ids):
                a = agents[aid]
                # Keyed draws (T22): whether a speaker talks depends on (speaker, tick) alone, and the
                # partner is whoever present scores lowest on a (speaker, tick, candidate) draw. Someone
                # joining or leaving the place changes a speaker's partner only if they are, or were, it.
                stream = f"talk:{aid}"
                roll = self.rng.at(stream, tick, 0, 999)
                if roll >= a["traits"]["extraversion"] // 4:
                    continue
                pick_stream = f"pair:{aid}:{tick}"
                b = agents[min((o for o in ids if o != aid), key=lambda o: (self.rng.raw(pick_stream, o), o))]
                draws = {"stream": stream, "index": tick, "pick_stream": pick_stream}
                grudge = self._rel(b, aid)
                if grudge["affinity"] <= -300:
                    self._note(b["id"], f"turned away from {a['name']}")
                    self._emit("snub", "chance", b["id"], {**draws, "with": aid},
                               [grudge["cause"]] if grudge.get("cause") is not None else [])
                    continue
                # gossip: each may pass on what they believe about thieves (credulity decides uptake)
                transfers, sources = [], []
                for x, y in ((a, b), (b, a)):
                    for k in sorted(y["beliefs"]):
                        if k.startswith("thief:") and k not in x["beliefs"] and int(k.split(":")[1]) != x["id"]:
                            c_roll = self.rng.at(f"cred:{x['id']}:{y['id']}:{k}", tick, 0, 999)
                            if c_roll < x["cognition"]["credulity"]:
                                transfers.append((x, y, k))
                                if y["belief_causes"].get(k) is not None:
                                    sources.append(y["belief_causes"][k])
                tid = self._emit("talk", "chance", aid, {**draws, "with": b["id"],
                                                         "place": place, "gossip": [[x["id"], k] for x, _, k in transfers]},
                                 sorted(set(sources)))
                for x, y in ((a, b), (b, a)):
                    x["needs"]["loneliness"] = _clamp(x["needs"]["loneliness"] - 200)
                    r = self._met(x, y["id"])
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
    def _possessed(self, a: dict) -> bool:
        return self.state.get("player", {}).get("agent") == a["id"]

    def _refuse_acts(self, a: dict, why: str):
        for eid, action, _ in self._acts.pop(a["id"], []):
            self._receipt(a, eid, action, False, why)

    def _player_deliberate(self, a: dict, occupancy: dict):
        """The possessed body's in-the-moment decision, where a mind's deliberation would be filed. Its
        view, situation and offer are a deliberation request's; it is applied in _commit, in _order."""
        (eid, action, target), *rest = self._acts.pop(a["id"])
        for e2, a2, _ in rest:
            self._receipt(a, e2, a2, False, "one act per tick")
        if action in RATED_ACTS and self.state["tick"] < a["cooldown_until"]:
            self._receipt(a, eid, action, False, "cooldown")
            return
        view = self._view(a, occupancy)
        if a["occasion"]:  # what a deliberation's situation would say; shown to the player as the occasion
            view["occasion"] = {k: a["occasion"].get(k) for k in ("why", "who", "detail")}
        view["offer"] = self._offer(a, view)
        self._situation(a, view, "deliberate")  # fills view["people"]; may cut a crowd, as for a mind
        offer = view["offer"] = self._offer(a, view)
        if action in RATED_ACTS and action in offer:
            a["cooldown_until"] = self.state["tick"] + 12
        self.player_pending[a["id"]] = (eid, action, target, offer, view["people"])

    def _apply_act(self, a: dict, eid: int, action: str, target, offer: list, people: list):
        """A player act, with the checks a mind's reply schema would have made, then the mind's own path."""
        if action not in offer:
            self._receipt(a, eid, action, False, "not offered", extra={"offer": offer})
        elif action in PERSON_STEPS and target not in people:
            self._receipt(a, eid, action, False, "unknown target")
        else:
            self._resolve(a, {"action": action, "target": target if action in PERSON_STEPS else None}, eid)

    def _player(self, act: dict):
        kind = act["kind"]
        if kind == "possess":
            aid = act["agent"]
            if not (_is_int(aid) and 0 <= aid < len(self.state["agents"])):
                raise ValueError(f"possess: no agent {aid!r}")
            if "player" in self.state:  # re-possessing moves the body
                self._release()
            eid = self._emit("possess", "player", aid, {"agent": aid})
            self.state["player"] = {"agent": aid, "event": eid}
            return eid
        if kind == "release":
            if "player" not in self.state:
                raise ValueError("release: no body is possessed")
            return self._release()
        if kind == "act":
            if "player" not in self.state:
                raise ValueError("act: no body is possessed")
            action, target = act["action"], act.get("target")
            if action not in DELIBERATE_ACTIONS or not (target is None or _is_int(target)):
                raise ValueError(f"act: bad action {action!r} or target {target!r}")
            aid = self.state["player"]["agent"]
            data = {"action": action} if target is None else {"action": action, "target": target}
            eid = self._emit("act", "player", aid, data, [self.state["player"]["event"]])
            self._acts.setdefault(aid, []).append((eid, action, target))
            return eid
        if kind == "fire":
            fam = self.state["families"][act["family"]]
            eid = self._emit("fire", "player", None, {"family": act["family"]})
            for m in fam["members"]:
                a = self.state["agents"][m]
                a["inv"]["food"] = 0
                a["last_food_loss_event"] = eid
                self._note(m, "a fire burned our food stores")
                self._occasion(a, "fire", 900, event=eid)
            return eid
        raise ValueError(f"unknown player action {act['kind']}")

    def _release(self) -> int:
        p = self.state.pop("player")
        a = self.state["agents"][p["agent"]]
        a["occasion"] = None  # the player handled whatever it was; the mind isn't asked about it later
        return self._emit("release", "player", a["id"], {"agent": a["id"]}, [p["event"]])
