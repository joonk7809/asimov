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
The prompt holds only what the player could see: names, roles, the place, the gossip passed on in the
talk, and whether someone looks starving. Never a villager's memory or needs (review of #61). E2b (Steering,
10:09Z, after the 8B dry run's lines invented rain, crops and prices): it also gives the hour and part of day,
the season from year-1, what each speaker is visibly doing, and says to speak only of these facts.
E2c (Steering, 13:07Z, after dry run 2's lines still named beans, stalls, a creek and a cow): the negative list
didn't hold at 8B, so the words are grounded in positive facts. The prompt says what the place holds, gives one
or two topics from the speakers' own real events of the last day (Recent), and names the greeting word for the
hour. A short deny-list check on the output asks once more on a hit, then falls back to the template.
E2d (Steering, 13:44Z, after the E2c check read true but stiff: both speakers greeted and recited a trade
receipt, and the reply never answered): only the opener greets and the reply answers what was said; one topic
per speaker, chosen for touching the other speaker or the listener or standing out (a refusal, a gift,
sickness) over routine trade, with at most one routine trade topic per exchange; a speaker who looks starving
or was just turned down may sound it. A second greeting counts as a hit.
"""
import json
import re
import time
from collections import deque

from .inhabitant import present_ids
from .sim import TICKS_PER_DAY, TRADE_INPUT

ASK_ACTS = ("ask_kin", "ask_food", "buy_from")
REFUSALS = ("refused", "kin refused", "nothing to spare", "kin has nothing to spare")
MAX_CHARS = 120
MAX_LINES = 2
# What a speaker is visibly doing (E2b, Steering 10:09Z): the player sees this anyway.
DOING = {"work": "at work", "socialize": "passing the time", "home": "at home", "rest": "resting",
         "explore": "wandering", "shop": "shopping for food", "sleep": "getting ready for bed"}

# What each place holds, said outright (E2c): the world has only "food", one stock at the market, no stalls.
VILLAGE = ("The village has a market, a field, a workshop, a plaza, the woods and the families' homes. "
           "Its only food is called just \"food\": no kinds of food are named.")
PLACE_HOLDS = {
    "market": "The market sells food, just \"food\", for coin. There are no stalls, goods or animals.",
    "field": "Farmers work the field, and it yields food, just \"food\". There are no animals and no named crops.",
    "workshop": "Crafters make tools in the workshop. Nothing else is made or sold there.",
    "plaza": "The plaza is the open square where people meet; healers and guards work there. Nothing is sold there.",
    "woods": "The woods are where people go to wander. Nothing in them is named: no animals, water or paths.",
}
HOME_HOLDS = "This is a family's home, where they eat and sleep."
# What kinds of things the 8B lines invented in the dry runs (Steering's E2c list: kinds of food, stall,
# animals, weather). A hit is asked once more, then falls back to the template.
DENY = re.compile(r"\b(" + "|".join([
    # kinds of food
    r"beans?", r"rice", r"flour", r"corn", r"wheat", r"barley", r"oats?", r"grain", r"bread", r"loaf", r"loaves",
    r"potato(?:es)?", r"turnips?", r"carrots?", r"cabbages?", r"onions?", r"apples?", r"berr(?:y|ies)", r"fruit",
    r"vegetables?", r"veg", r"eggs?", r"milk", r"cheese", r"meat", r"fish", r"stew", r"soup", r"pie", r"honey",
    # stalls and goods
    r"stalls?", r"booths?", r"carts?", r"wares",
    # animals
    r"cows?", r"cattle", r"oxen", r"ox", r"pigs?", r"hogs?", r"sheep", r"lambs?", r"goats?", r"horses?",
    r"chickens?", r"hens?", r"geese", r"goose", r"ducks?", r"fox(?:es)?", r"wolf", r"wolves", r"deer", r"rabbits?",
    r"dogs?", r"cats?", r"birds?", r"crows?",
    # weather
    r"weather", r"rain(?:s|y|ing|ed)?", r"sun(?:ny|shine)?", r"snow(?:s|y|ing)?", r"storms?", r"wind(?:y)?",
    r"clouds?", r"cloudy", r"frost", r"fog", r"drought", r"dust(?:y)?",
]) + r")\b", re.I)
GREETINGS = {"morning": ("mornin", "morning"), "afternoon": ("afternoon",), "evening": ("evenin", "evening")}
# Any greeting form, for "at most one greeting per exchange" (E2d): a reply opening with one of these greets.
# A bare part-of-day word greets only when a "," "!" "." "to you" or the end follows it (review of #71):
# "Evening, Ari." greets, "Evening comes soon." doesn't.
OPENERS = re.compile(r"^\W*(?:(?:good (?:mornin|morning|afternoon|evenin|evening|day)|hello|hullo|hi|hey|greetings|"
                     r"well met)(?:'|\b)(?![a-z'])|(?:mornin|morning|afternoon|evenin|evening)'?\s*(?:[,!.]|$|to you\b))", re.I)
JUST_TURNED_DOWN = 6  # ticks (an hour): a refusal this recent may color a speaker's tone (E2d)
# Topic weights (E2d): what touches the other speaker or the listener first, then what stands out, then a
# purchase, then routine trade (trade-1's inputs), which an exchange gets at most one of.
TOUCH, STANDOUT, BUY, ROUTINE = 3, 2, 1, 0


def greeting(hour: int) -> str:
    """The greeting word for the hour, given outright (E2c): 5-11 morning, 12-16 afternoon, else evening."""
    return "Good morning" if 5 <= hour < 12 else "Good afternoon" if 12 <= hour < 17 else "Good evening"


def invented(text: str, hour: int) -> list[str]:
    """What in a line the world doesn't have (the deny list), or a greeting for another part of the day."""
    hits = [m.group(0).lower() for m in DENY.finditer(text)]
    right = greeting(hour).split()[1]
    low = text.lower().strip()
    for part, words in GREETINGS.items():
        # Only greeting forms (review of #70): "good X", or X opening a sentence or followed by "," or "!".
        # A plain time word ("this morning") is a true topic, not a greeting.
        w = "(?:" + "|".join(words) + ")(?:'|\\b)(?![a-z'])"  # "mornin'" yes, "afternoon's" no
        # A bare word opening a sentence greets only before ",", "!", ".", "to you" or the end (review of #71).
        if part != right and re.search(r"\bgood " + w + r"|(?:^|[.!?]\s+)" + w + r"\s*(?:[,!.]|$|to you\b)|\b" + w
                                       + r"\s*[,!]", low):
            hits.append(f"{part} greeting")
    return hits


def check(it: dict, lines: list[dict]) -> list[str]:
    """Every hit in an answer: invented facts or a wrong greeting in any line, and a greeting after the first
    line (E2d: only the opener greets)."""
    hits = invented(" ".join(x["text"] for x in lines), it.get("hour", 12))
    if any(OPENERS.match(x["text"]) for x in lines[1:]):
        hits.append("second greeting")
    return hits


class Recent:
    """Each villager's own real events of the last day, as they'd say them aloud (E2c topics): what they
    bought, asked for, gave or were refused, the trade input they need from another family (trade-1), and,
    read from the state, sickness in their family. Fed every tick by the runner. Words only: nothing in the
    world reads it, and a resumed runner starts it empty."""

    def __init__(self, window: int = TICKS_PER_DAY):
        self.window = window
        self.by: dict[int, deque] = {}  # aid -> (tick, text, weight, partner or None)
        self.due: dict[int, tuple[int, int, str]] = {}  # trade-1: aid -> (tick, supplier, input), until supplied
        self.refused: dict[int, int] = {}  # aid -> tick it was last turned down for food

    def feed(self, sim, events):
        agents = sim.state["agents"]
        name = lambda i: agents[i]["name"]
        for e in events:
            d, a = e.data, e.actor
            if a is None:
                continue
            if e.kind == "trade_due" and isinstance(d.get("to"), int):
                self.due[a] = (e.tick, d["to"], TRADE_INPUT.get(agents[d["to"]]["role"], "goods"))
                continue
            if e.kind != "receipt":
                continue
            act, why, to = d.get("action"), d.get("why"), d.get("to")
            who = name(to) if isinstance(to, int) else None
            t, w = None, STANDOUT
            if (act in ("ask_kin", "ask_food", "buy_from") and why in
                    ("kin refused", "kin has nothing to spare", "refused", "nothing to spare")):
                self.refused[a] = e.tick
            if act == "buy_food" and why == "bought":
                w = BUY
                t = f"{name(a)} bought food at the market"
            elif act == "buy_food" and why == "no coin":
                t = f"{name(a)} hadn't the coin to buy food"
            elif act == "ask_kin" and why == "kin gave food":
                t = f"{who} in {name(a)}'s family gave them food" if who else f"{name(a)}'s family gave them food"
            elif act == "ask_kin" and why in ("kin refused", "kin has nothing to spare"):
                t = f"{name(a)} asked {who or 'their family'} for food and got none"
            elif act in ("ask_food", "buy_from") and who and why in ("gave", "sold"):
                t = f"{who} gave {name(a)} food when asked" if act == "ask_food" else f"{name(a)} bought food from {who}"
                w = STANDOUT if act == "ask_food" else BUY
            elif act in ("ask_food", "buy_from") and who and why in ("refused", "nothing to spare"):
                t = f"{name(a)} asked {who} for food and was turned down"
            elif act == "give_food" and who and why == "gave":
                t = f"{name(a)} gave food to {who}"
            elif act == "give_coin" and who and why == "gave":
                t = f"{name(a)} gave coin to {who}"
            elif act == "trade" and who and why == "supplied":
                self.due.pop(a, None)
                t = f"{name(a)} got {TRADE_INPUT.get(agents[to]['role'], 'goods')} from {who}, the {agents[to]['role']}"
                w = ROUTINE
            if t:
                q = self.by.setdefault(a, deque(maxlen=8))
                if not q or q[-1][1] != t:
                    q.append((e.tick, t, w, to if isinstance(to, int) else None))

    def candidates(self, sim, aid: int, near=()) -> list[tuple[int, str]]:
        """`aid`'s own topics of the last day as (weight, text), best first (E2d): one that touches someone in
        `near` (the other speaker, the listener) first, then what stands out (sickness in the family, a refusal,
        a gift), then a purchase, then routine trade; newest first within a weight."""
        st, agents = sim.state, sim.state["agents"]
        now, near = st["tick"], set(near)
        out = []  # (weight, age, text)
        ill = st.get("sick", {}).get("ill", {})
        for m in st["families"][agents[aid]["family"]]["members"]:
            if str(m) in ill:
                out.append((STANDOUT, 0, f"{agents[aid]['name']} is sick" if m == aid
                            else f"{agents[m]['name']} in {agents[aid]['name']}'s family is sick"))
        due = self.due.get(aid)
        if due and now - due[0] < self.window:
            out.append((TOUCH if due[1] in near else ROUTINE, now - due[0],
                        f"{agents[aid]['name']} needs {due[2]} from {agents[due[1]]['name']}, "
                        f"the {agents[due[1]]['role']}, today"))
        for tick, t, w, partner in self.by.get(aid, ()):
            if now - tick < self.window:
                out.append((TOUCH if partner in near else w, now - tick, t))
        best, seen = [], set()
        for w, _, t in sorted(out, key=lambda x: (-x[0], x[1])):
            if t not in seen:
                seen.add(t)
                best.append((w, t))
        return best

    def topics(self, sim, aid: int, k: int = 2, near=()) -> list[str]:
        return [t for _, t in self.candidates(sim, aid, near)[:k]]

    def just_refused(self, sim, aid: int) -> bool:
        t = self.refused.get(aid)
        return t is not None and sim.state["tick"] - t <= JUST_TURNED_DOWN


def items(sim, events, aid: int, recent: "Recent | None" = None) -> list[dict]:
    """What in this tick's events gets words for the inhabitant `aid`. Read between ticks, from the state the
    tick left (who is where, hunger, notes), so the words match what the player saw. `recent`, if given,
    supplies each speaker's own topics (E2c)."""
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
    tick = sim.state["tick"] - 1
    for it in out:
        it["tick"] = tick
        it["place"] = me["place"]
        it["hour"] = (tick % TICKS_PER_DAY) * 10 // 60
        it["season"] = season(sim, tick // TICKS_PER_DAY)
        # Only what the player could see of them (review of #61): no memory, no needs but a visible starving.
        it["people"] = {str(p): {"name": agents[p]["name"], "role": agents[p]["role"],
                                 "looks": "starving" if agents[p]["needs"]["hunger"] >= 800 else "",
                                 "doing": DOING.get(agents[p]["activity"], ""),
                                 # E2d: a speaker's own refusal this past hour may color the tone
                                 "mood": "just turned down for food" if recent is not None and p != aid
                                 and recent.just_refused(sim, p) else ""}
                        for p in set(it["speakers"]) | {aid}}
        it["player"] = aid
        # At most one topic per speaker from their own events (E2c, E2d), and only one that touches the other
        # speaker or the listener (pt8), or sickness in the speaker's family; a bystander's trade is not offered.
        it["topics"] = []
        if recent is not None:
            for p in it["speakers"]:
                near = (set(it["speakers"]) | {aid}) - {p}
                # Steering 11:05Z: sickness in the speaker's own family bears on any neighbour, so it is offered too.
                pick = next((t for w, t in recent.candidates(sim, p, near)
                             if w == TOUCH or t.endswith(" is sick")), None)
                if pick is not None:
                    it["topics"].append({"who": p, "text": pick})
    return out


def season(sim, day: int) -> str:
    """The part of the year a villager would name (year-1): winter, the harvest day before it, or the growing
    season. A world with no year has none."""
    y = sim.state.get("stakes", {}).get("year")
    if not y:
        return ""
    n = len(y["yields"])
    if y["yields"][day % n] == 0:
        return "winter"
    return "harvest day, the last before winter" if y["yields"][(day + 1) % n] == 0 else "the growing season"


def part_of_day(hour: int) -> str:
    return ("night" if hour < 5 else "early morning" if hour < 8 else "morning" if hour < 12 else
            "afternoon" if hour < 17 else "evening" if hour < 21 else "night")


def _who(it, p) -> str:
    return it["people"][str(p)]["name"]


def prompt(it: dict) -> str:
    """One villager's (or two villagers') next words, from the item's facts alone."""
    pl = it["player"]
    h = it.get("hour", 12)
    place = str(it["place"])
    lines = ["You write the spoken words of villagers in a small farming village. Plain, short, everyday speech.",
             VILLAGE,
             f"Place: the {'home' if place.startswith('home') else place.replace('_', ' ')}. "
             + PLACE_HOLDS.get(place, HOME_HOLDS),
             f"Time: {h:02d}:00, {part_of_day(h)}" + (f", {it['season']}." if it.get("season") else ".")
             + f' A greeting now is "{greeting(h)}".']
    for p, x in sorted(it["people"].items()):
        tag = " (the person listening)" if int(p) == pl and pl not in it["speakers"] else ""
        doing = f", {x['doing']}" if x.get("doing") else ""
        mood = f", {x['mood']}" if x.get("mood") else ""
        lines.append(f"- {x['name']}, {x['role']}{doing}{', looks starving' if x['looks'] else ''}{mood}{tag}.")
    a = _who(it, it["speakers"][0])
    you = _who(it, pl)
    gossip = (" " + "; ".join(it["gossip"]) + ".") if it.get("gossip") else ""
    if it["kind"] == "talk" and not it["to_player"]:
        b = _who(it, it["speakers"][1])
        lines.append(f"{a} starts a short chat with {b}.{gossip}")
        lines.append(f"Write {a}'s line, then {b}'s reply. Only {a} may greet. {b} doesn't greet: {b} answers "
                     f"what {a} just said, with an answer, a comment, sympathy or a refusal.")
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
    for t in it.get("topics", []):
        lines.append(f"Something true {_who(it, t['who'])} might bring up: {t['text']}.")
    if len(it.get("topics", [])) > 1:
        lines.append(f"Each line uses at most one of these, and {_who(it, it['speakers'][1])} may skip theirs.")
    if any(x.get("looks") or x.get("mood") for k, x in it["people"].items() if int(k) in it["speakers"]):
        lines.append("Let how they look or feel color how they speak.")
    lines.append(f"Speak only of these facts and the people here. Only the first line may greet, with \"{greeting(h)}\".")
    lines.append("Every line is said to someone here, by name. A line nobody is meant to hear is thinking aloud:"
                 ' its "to" is "self".')
    lines.append('Answer as JSON: {"lines": [{"who": NAME, "to": NAME or "self", "text": WORDS}]}, at most two lines,'
                 " each under 100 characters.")
    return "\n".join(lines)


def addressee(it: dict, i: int, who: int) -> int:
    """Whom line i of an item is said to, by the item's own shape: in a chat between two villagers, each to the
    other; any other line to the player (pt8, joon: "words should be directed to a person")."""
    s = it["speakers"]
    if len(s) > 1:
        return s[1] if who == s[0] else s[0]
    return it["player"]


def addressed(it: dict, lines: list[dict]) -> list[dict]:
    """Every line with its `to`: an id of someone in the item, or "self" (thinking aloud). A line that names
    nobody valid gets the item's own addressee."""
    out = []
    for i, x in enumerate(lines):
        to = x.get("to")
        if not (to == "self" or (isinstance(to, int) and str(to) in it["people"] and to != x["who"])):
            to = addressee(it, i, x["who"])
        out.append({**x, "to": to})
    return out


def clean(it: dict, out) -> list[dict] | None:
    """The model's lines, if they are well formed and spoken by the item's speakers; else None."""
    names = {_who(it, p): p for p in it["speakers"]}
    if not isinstance(out, dict) or not isinstance(out.get("lines"), list) or not out["lines"]:
        return None
    lines = []
    for x in out["lines"][:MAX_LINES]:
        if not (isinstance(x, dict) and x.get("who") in names and isinstance(x.get("text"), str) and x["text"].strip()):
            return None
        to = x.get("to")
        everyone = {_who(it, p): int(p) for p in it["people"]}
        to = "self" if isinstance(to, str) and to.strip().lower() == "self" else everyone.get(to)
        line = {"who": names[x["who"]], "text": x["text"].strip().strip('"')[:MAX_CHARS]}
        if to is not None and to != line["who"]:  # pt8: whom the model says it is to; else `addressed` decides
            line["to"] = to
        lines.append(line)
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

    def _ask(self, content: str, it: dict):
        body = {"model": self.client.model, "messages": [{"role": "user", "content": content}],
                "temperature": self.client.temperature, "top_p": 0.9, "max_tokens": 160,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_object"}}
        try:
            text = self.client.post("/v1/chat/completions", body)["choices"][0]["message"].get("content") or ""
            return clean(it, json.loads(text))
        except (RuntimeError, ValueError, KeyError, IndexError, TypeError):
            return None

    def lines(self, it: dict) -> list[dict]:
        """One call; on a deny-list hit (E2c) one more, naming what isn't in the village; else the template.
        it["denied"] keeps each hit asked about, it["fallback"] whether the template was used."""
        t0 = time.monotonic()
        p = prompt(it)
        got = self._ask(p, it)
        it["denied"] = []
        hits = check(it, got) if got else []
        if hits:
            it["denied"].append(hits)
            got = self._ask(p + "\nNot " + ", ".join(sorted(set(hits))) + ": the village has none of that, and only "
                            "the first line greets. Say only what is listed above.", it)
            again = check(it, got) if got else []
            if again:
                it["denied"].append(again)
                got = None
        it["latency_s"] = round(time.monotonic() - t0, 3)
        it["fallback"] = got is None
        return got if got is not None else self.fallback.lines(it)
