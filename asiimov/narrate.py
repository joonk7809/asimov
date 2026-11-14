"""Words for what a player can see (plan/readable-world-spec.md E2, signed off by Steering 2026-11-14).

After each live tick, the runner picks the events its inhabitant witnessed that people would speak in:
- a `talk` or `snub` at the player's place while the player is awake there;
- a refusal of the player's own ask (the refuser's one-line reply);
- a villager coming to see the player (`meet` with the player).

Each gets a line or two of speech, written after the tick so it never slows one. The words go to the
runner's sidecar (narration.jsonl) and to the inhabitant as a `heard` message. **Nothing feeds back**: no
villager remembers what it said, and the world log and replay are unchanged. These are words about the
minds' acts, not acts.

The narrator is one model for both sessions of a blind pair, over whichever mind ran the acts (Steering:
otherwise the pair compares narrators). TemplateNarrator is only the fallback when no model is served.
The prompt holds only what the event and the speakers' public state hold: names, roles, the place, the
gossip passed on, each side's hunger as a word and their last two notes.
"""
import json
import time

from .inhabitant import present_ids

ASK_ACTS = ("ask_kin", "ask_food", "buy_from")
REFUSALS = ("refused", "kin refused", "nothing to spare", "kin has nothing to spare")
MAX_CHARS = 120
MAX_LINES = 2


def hunger_word(h: int) -> str:
    return "full" if h < 300 else "a little hungry" if h < 550 else "hungry" if h < 800 else "starving"


def items(sim, events, aid: int) -> list[dict]:
    """What in this tick's events gets words for the inhabitant `aid`. Read between ticks, from the state the
    tick left (who is where, hunger, notes), so the words match what the player saw."""
    agents = sim.state["agents"]
    me = agents[aid]
    here = set(present_ids(sim, aid))
    awake_here = not me["asleep"] and me["place"] is not None
    out = []
    for e in events:
        d = e.data
        if e.kind in ("talk", "snub") and awake_here and e.actor is not None:
            # talk: the actor starts it with `with`. snub: the actor turns away from `with`, who tried to talk.
            first, second = (e.actor, d.get("with")) if e.kind == "talk" else (d.get("with"), e.actor)
            if all(p == aid or p in here for p in (first, second)):
                # Never words for the player's own villager: only the other one speaks, to the player.
                speakers = [p for p in (first, second) if p != aid]
                out.append({"event": e.id, "kind": e.kind, "speakers": speakers, "to_player": aid in (first, second),
                            "opener": first, "gossip": [
                                f"{agents[x]['name']} hears that {agents[int(k.split(':')[1])]['name']} is a thief"
                                for x, k in d.get("gossip", []) if k.startswith("thief:")]})
        elif e.kind == "receipt" and e.actor == aid and d.get("action") in ASK_ACTS and d.get("why") in REFUSALS \
                and isinstance(d.get("to"), int):
            out.append({"event": e.id, "kind": "refusal", "speakers": [d["to"]], "to_player": True,
                        "action": d["action"], "why": d["why"]})
        elif e.kind == "meet" and d.get("with") == aid and e.actor != aid:
            out.append({"event": e.id, "kind": "visit", "speakers": [e.actor], "to_player": True})
    for it in out:
        it["tick"] = sim.state["tick"] - 1
        it["place"] = me["place"]
        it["people"] = {str(p): {"name": agents[p]["name"], "role": agents[p]["role"],
                                 "hunger": hunger_word(agents[p]["needs"]["hunger"]),
                                 "notes": [n.split(" ", 1)[-1] for n in agents[p]["recent"][-2:]]}
                        for p in set(it["speakers"]) | {aid}}
        it["player"] = aid
    return out


def _who(it, p) -> str:
    return it["people"][str(p)]["name"]


def prompt(it: dict) -> str:
    """One villager's (or two villagers') next words, from the item's facts alone."""
    pl = it["player"]
    lines = ["You write the spoken words of villagers in a small farming village. Plain, short, everyday speech.",
             f"Place: the {str(it['place']).replace('_', ' ')}."]
    for p, x in sorted(it["people"].items()):
        tag = " (the person listening)" if int(p) == pl and pl not in it["speakers"] else ""
        notes = "; ".join(x["notes"]) or "nothing much"
        lines.append(f"- {x['name']}, {x['role']}, {x['hunger']}{tag}. Lately: {notes}.")
    a = _who(it, it["speakers"][0])
    you = _who(it, pl)
    gossip = (" " + "; ".join(it["gossip"]) + ".") if it.get("gossip") else ""
    if it["kind"] == "talk" and not it["to_player"]:
        b = _who(it, it["speakers"][1])
        lines.append(f"{a} starts a short chat with {b}.{gossip}")
        lines.append(f"Write {a}'s line, then {b}'s reply.")
    elif it["kind"] == "talk":
        lines.append((f"{you} starts a short chat with {a}." if it["opener"] == pl else f"{a} starts a short chat with {you}.") + gossip)
        lines.append(f"Write {a}'s one line to {you}.")
    elif it["kind"] == "snub" and not it["to_player"]:
        b = _who(it, it["speakers"][1])
        lines.append(f"{a} tries to talk to {b}, but {b} holds a grudge and turns away.")
        lines.append(f"Write {a}'s line, then {b}'s cold reply.")
    elif it["kind"] == "snub":
        lines.append(f"{you} tries to talk to {a}, but {a} holds a grudge and turns away." if it["opener"] == pl
                     else f"{a} tries to talk to {you}, but {you} turns away.")
        lines.append(f"Write {a}'s one line to {you}.")
    elif it["kind"] == "refusal":
        what = "buy food from them" if it["action"] == "buy_from" else "spare some food"
        lines.append(f"{_who(it, pl)} just asked {a} to {what}. {a} says no"
                     + (" because they have nothing to spare." if "spare" in it["why"] else "."))
        lines.append(f"Write {a}'s one-line reply to {_who(it, pl)}.")
    else:
        lines.append(f"{a} has come to see {_who(it, pl)}.")
        lines.append(f"Write what {a} says on arriving.")
    lines.append('Answer as JSON: {"lines": [{"who": NAME, "text": WORDS}]}, at most two lines, each under 100 characters.')
    return "\n".join(lines)


def clean(it: dict, out) -> list[dict] | None:
    """The model's lines, if they are well formed and spoken by the item's speakers; else None."""
    names = {_who(it, p): p for p in it["speakers"]}
    if not isinstance(out, dict) or not isinstance(out.get("lines"), list) or not out["lines"]:
        return None
    lines = []
    for x in out["lines"][:MAX_LINES]:
        if not (isinstance(x, dict) and x.get("who") in names and isinstance(x.get("text"), str) and x["text"].strip()):
            return None
        lines.append({"who": names[x["who"]], "text": x["text"].strip().strip('"')[:MAX_CHARS]})
    return lines


class TemplateNarrator:
    """Fixed lines per kind: the fallback when no model is served (and the tests' narrator)."""
    model = "template-v0"

    def lines(self, it: dict) -> list[dict]:
        s = it["speakers"]
        if it["kind"] == "talk":
            return [{"who": s[0], "text": "Hello. How's the day treating you?"}] + \
                ([{"who": s[1], "text": "Not bad. Same as always."}] if len(s) > 1 else [])
        if it["kind"] == "snub":
            if len(s) > 1:
                return [{"who": s[0], "text": "Can we talk?"}, {"who": s[1], "text": "I've nothing to say to you."}]
            return [{"who": s[0], "text": "Can we talk?" if it["opener"] == s[0] else "I've nothing to say to you."}]
        if it["kind"] == "refusal":
            return [{"who": s[0], "text": "Sorry, I can't spare any." if "spare" in it["why"] else "No, not this time."}]
        return [{"who": s[0], "text": "I wanted to see you."}]


class LLMNarrator:
    """The session's model, one call per item, JSON lines; falls back to the template on anything malformed."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, temperature: float = 0.7,
                 timeout: float = 20.0):
        from .llm import LLMMind
        self.client = LLMMind(base_url, model, temperature=temperature, max_tokens=160, timeout=timeout, retries=1,
                              structured=False, api_key=api_key)
        self.model = model
        self.fallback = TemplateNarrator()

    def lines(self, it: dict) -> list[dict]:
        body = {"model": self.client.model, "messages": [{"role": "user", "content": prompt(it)}],
                "temperature": self.client.temperature, "top_p": 0.9, "max_tokens": 160,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_object"}}
        t0 = time.monotonic()
        try:
            text = self.client.post("/v1/chat/completions", body)["choices"][0]["message"].get("content") or ""
            got = clean(it, json.loads(text))
        except (RuntimeError, ValueError, KeyError, IndexError, TypeError):
            got = None
        it["latency_s"] = round(time.monotonic() - t0, 3)
        it["fallback"] = got is None
        return got if got is not None else self.fallback.lines(it)
