"""What an inhabitant client is sent (plan/player-body-spec.md, "Protocol 2: the inhabitant scope").

The server filters and the client is never trusted to hide anything. Everything here is built field by field
from a whitelist, never as a copy of engine state (Sim._view carries traits), and it only reads the world:
the live runner calls it between ticks, and test_inhabitant checks the log still equals batch.
"""
from .sim import OCCASION_TICKS, PERSON_STEPS, TICKS_PER_DAY
from .worldmap import position

# The per-tick view's keys, as the spec lists them. A new field needs a spec line, and bar 5 checks this.
VIEW_KEYS = ("tick", "place", "route", "pos", "activity", "asleep", "hunger", "food", "coin", "plan", "recent",
             "present", "kin", "market_food", "occasion", "offer", "cooldown_until", "food_price", "intent")
PUBLIC = frozenset({"fire"})            # events the whole village sees
NAMES = ("with", "to", "target", "household")  # data fields that name the villager an event is about
# Another villager's events it can see, and the fields of each: what it does in plain sight. Nothing that
# is that villager's mind (intent_*, plan_receipt), its body clock (day_rhythm) or its memory (sleep's counts).
SEEN = {"activity": ("activity", "place", "arrive", "from"), "arrive": ("place",), "wake": ("place",),
        "sleep": (), "talk": ("with", "place", "gossip"), "meet": ("with", "place"), "snub": ("with",),
        "eat": ("household",), "sell": ("food",), "fire": ("family",)}
SEEN_NAMING = {**SEEN, "receipt": ("action", "ok", "why", "to")}  # also a receipt for an act aimed at it


def _pos(sim, a) -> list[int]:
    places, r = sim.map["places"], a["route"]
    return list(places[a["place"]]["anchor"] if r is None else position(r, places[r["to"]]["anchor"], sim.state["tick"]))


def present_ids(sim, aid: int) -> list[int]:
    """Who is at the villager's place now, awake or asleep (walkers are nowhere, as in the engine)."""
    agents = sim.state["agents"]
    place = agents[aid]["place"]
    return [] if place is None else [o["id"] for o in agents if o["place"] == place and o["id"] != aid]


def welcome_world(sim, aid: int) -> dict:
    """The map, its own villager and its family. No souls, traits or params, not even its own."""
    a, agents = sim.state["agents"][aid], sim.state["agents"]
    fam = sim.state["families"][a["family"]]
    return {"map": sim.map, "ticks_per_day": TICKS_PER_DAY,
            "me": {"id": aid, "name": a["name"], "role": a["role"], "family": a["family"], "home": a["home"]},
            "family": [{"id": m, "name": agents[m]["name"], "role": agents[m]["role"]} for m in fam["members"] if m != aid]}


def view(sim, aid: int) -> dict:
    """The villager at the current view tick, as the spec's whitelist. `offer` is what a deliberation would
    be offered from here (Sim._view, _situation and _offer, as _player_deliberate computes it); the engine
    decides again when the act applies, after this tick's arrivals and meals, so an act can still be
    refused with "not offered"."""
    st = sim.state
    tick, agents, a = st["tick"], st["agents"], st["agents"][aid]
    fam = st["families"][a["family"]]["members"]
    here = present_ids(sim, aid)
    r = a["route"]
    out = {"tick": tick, "place": a["place"],
           "route": None if r is None else {"from": list(r["from"]), "to": r["to"], "depart": r["depart"], "arrive": r["arrive"]},
           "pos": _pos(sim, a), "activity": a["activity"], "asleep": a["asleep"], "hunger": a["needs"]["hunger"],
           "food": a["inv"]["food"], "coin": a["inv"]["coin"],
           "plan": [{"activity": b["activity"], "start_h": b["start_h"], "end_h": b["end_h"]} for b in a["plan"] or []],
           "recent": list(a["recent"]),
           "present": [{"id": o, "name": agents[o]["name"], "role": agents[o]["role"], "kin": o in fam,
                        "has_food": agents[o]["inv"]["food"] > 0, "pos": _pos(sim, agents[o])} for o in here],
           "kin": [{"id": m, "name": agents[m]["name"], "home": agents[m]["home"]} for m in fam if m != aid],
           "occasion": None, "offer": {}, "cooldown_until": a["cooldown_until"], "food_price": sim._price(),
           "intent": None}
    it = a["intent"]
    if it:  # its own mind's words: what this body meant to do (Steering, 2026-11-10; never another's)
        out["intent"] = {"goal": it["goal"], "i": it["i"],
                         "steps": [{"do": st["do"], "target": st.get("target")} for st in it["steps"]]}
    if a["place"] == "market":
        out["market_food"] = st["market_food"]
    occ = a["occasion"]
    if occ and tick - occ["tick"] < OCCASION_TICKS:
        out["occasion"] = {"why": occ["why"], "who": occ["who"],
                           "text": sim._occasion_text(a, occ).removeprefix(" Why you stop to think: ").rstrip(".")}
    if not a["asleep"]:
        occupancy = {a["place"]: [aid] + here} if a["place"] is not None else {}
        v = sim._view(a, occupancy)
        if out["occasion"]:
            v["occasion"] = {k: occ.get(k) for k in ("why", "who", "detail")}
        v["offer"] = sim._offer(a, v)
        sim._situation(a, v, "deliberate")  # fills v["people"]; may cut a crowd, as for a mind
        out["offer"] = {x: (v["people"] if x in PERSON_STEPS else []) for x in sim._offer(a, v)}
    assert set(out) <= set(VIEW_KEYS), "a view field outside the spec's list"
    return out


def witnessed(e: dict, aid: int, here: set, tick: int) -> bool:
    """Whether the villager would witness this event: its own, one that names it, a public one, or (on the
    tick it happens, live) one by someone at its place. Never a mind event: its own plan is in the view."""
    if e["origin"] == "mind" or e["kind"] in ("genesis", "checkpoint", "end"):
        return False
    if e["actor"] == aid or e["kind"] in PUBLIC:
        return True
    if e["kind"] in SEEN_NAMING and any(e["data"].get(k) == aid for k in NAMES):
        return True
    return e["kind"] in SEEN and e["tick"] == tick and e["actor"] in here


def scrub(e: dict, aid: int) -> dict:
    """An event as an inhabitant gets it: no mind record (prompt hash, raw output), no chance-stream draws,
    and of another villager's event only the fields SEEN lists."""
    if e["actor"] == aid:  # its own, with its own intent's goal and steps (what it meant to do)
        data = {k: v for k, v in e["data"].items() if k not in ("stream", "index", "pick_stream", "jitter_min")}
    else:
        data = {k: e["data"][k] for k in SEEN_NAMING[e["kind"]] if k in e["data"]}
    return {"id": e["id"], "tick": e["tick"], "kind": e["kind"], "origin": e["origin"], "actor": e["actor"],
            "data": data, "parents": e["parents"]}
