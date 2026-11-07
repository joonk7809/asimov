"""SANDBOX (T24 part 1.3, plan/t24-soul-channels-spec.md). This branch never merges.

The stub wired to the genie's six one-rule channels (plan/genie-soul-legibility.md, section D), so that a
leave-one-agent-out oracle can say what test 11's format could read if the world gave every sheet line a choice.
Each channel is one rule on one sheet line, decided by the stub's own params at a moment the world creates, and
logged as a `choice` event (origin "mind") that a day-log can show as one line:

  share     D1  agreeableness      asked for food: high gives to anyone; mid to kin and friends; low refuses
  work_end  D2  conscientiousness  at the end of a planned work block: high works one more hour; mid only when
                                   short of coin; low stops
  surplus   D3  values             at bedtime, carrying more than a day's food or 20 coin: benevolence in the
                                   top three gives to the hungriest person they know; security keeps; otherwise
                                   sell the food and keep the coin
  gossip    D4  honesty            told a thief belief could be passed on: low repeats anything; mid repeats
                                   hearsay only to kin; high repeats only what it saw itself
  path      D5  emotionality       the first walk to work each day: low takes the short cut through the woods
                                   (a keyed chance of a find or a loss); high takes the road; mid flips a coin
  company   D6  extraversion,      a socialize block starts: high extraversion goes looking for a new face, low
                openness           for the best friend, mid alternates by day; high openness rotates the place
                                   (plaza, market, woods), mid goes to the market every third day, low stays
                                   at the plaza

With no channel on, WiredSim is Sim and WiredStub is StubMind: the same log for the same seed.

    python -m asiimov.wired --out DIR --channels all|none|share,work_end,... [--days 7 --agents 100 --seed 1 --fire 3@1]
"""
import argparse
import gzip
import json
import os
import time

from .minds import StubMind
from .scorecard import ArmSim, labeled_population
from .sim import FARMER_SELL_PRICE, FOOD_PRICE, TICKS_PER_DAY, Sim, _clamp

CHANNELS = ("share", "work_end", "surplus", "gossip", "path", "company")
FOOD_KEEP, COIN_KEEP = 3, 20          # a day's meals; what the surplus rule leaves alone
WOODS_FIND, WOODS_LOSS = 250, 900     # out of 1000: a find below, a loss at or above
PLACES = ("plaza", "market", "woods")


def band(v: int) -> str:
    return "low" if v < 350 else "high" if v > 650 else "mid"


class WiredStub(StubMind):
    model = "stub-wired-v0"

    def choose(self, what: str, v: dict) -> dict:
        """One rule per channel, on the sheet line the channel is for. `v` holds only what a mind could see."""
        t = v["traits"]
        if what == "share":
            b = band(t["agreeableness"])
            return {"give": b == "high" or (b == "mid" and (v["kin"] or v["affinity"] >= 100))}
        if what == "work_end":
            b = band(t["conscientiousness"])
            return {"go_on": b == "high" or (b == "mid" and v["coin"] < 10)}
        if what == "surplus":
            for val in v["values"]["top"]:  # the higher-ranked of benevolence and security wins
                if val == "benevolence":
                    return {"do": "give" if v["to"] is not None else "keep"}
                if val == "security":
                    return {"do": "keep"}
            return {"do": "sell" if v["food_surplus"] > 0 else "keep"}
        if what == "gossip":
            if not v["hearsay"]:
                return {"pass": True}
            b = band(t["honesty"])
            return {"pass": b == "low" or (b == "mid" and v["kin"])}
        if what == "path":
            b = band(t["emotionality"])
            return {"risky": b == "low" or (b == "mid" and v["coin_flip"])}
        if what == "company":
            b = band(t["extraversion"])
            want_new = b == "high" or (b == "mid" and v["day"] % 2 == 1)
            who = (v["new"] if want_new else v["friend"])
            if who is None:
                who = v["friend"] if want_new else v["new"]
            o = band(t["openness"])
            where = PLACES[v["day"] % 3] if o == "high" else ("market" if o == "mid" and v["day"] % 3 == 2 else "plaza")
            return {"who": who, "new_face": who is not None and who == v["new"], "where": where}
        raise ValueError(what)


class WiredSim(Sim):
    def __init__(self, seed, n_agents, mind, genesis=None, channels=CHANNELS, **kw):
        self.channels = frozenset(channels)
        super().__init__(seed, n_agents, mind, genesis=genesis, **kw)

    def on(self, ch: str) -> bool:
        return ch in self.channels

    def _choice(self, a: dict, what: str, facts: dict, decision: dict, parents=()) -> int:
        """Log a wired decision as the agent's own: what was asked, the facts in view, and the answer."""
        return self._emit("choice", "mind", a["id"], {"what": what, **facts, **decision}, parents, mind={
            "model": self.mind.model, "prompt_hash": None, "prompt_tokens": 0, "output_tokens": 0,
            "sampling": {"temperature": 0}, "raw_output": json.dumps(decision, sort_keys=True)})

    def _kin(self, a: dict, other: int) -> bool:
        return other in self.state["families"][a["family"]]["members"]

    # ---- D1: the share decision moves from a roll on the target's agreeableness to the target's mind
    def _kin_limit(self, t, a):  # reached only by ask_kin once _ask_share is wired
        if not self.on("share"):
            return super()._kin_limit(t, a)
        dec = self.mind.choose("share", {"traits": t["traits"], "kin": True, "affinity": self._rel(t, a["id"])["affinity"],
                                         "food": t["inv"]["food"], "action": "ask_kin"})
        self._choice(t, "share", {"asker": a["id"], "kin": True, "action": "ask_kin"}, dec)
        return 1000 if dec["give"] else 0

    def _ask_share(self, a, t, action, eid):
        if not self.on("share"):
            return super()._ask_share(a, t, action, eid)
        to = {"to": t["id"]}
        if t["place"] != a["place"]:
            return self._receipt(a, eid, action, False, "target not here", extra=to)
        if t["asleep"]:
            return self._receipt(a, eid, action, False, "target asleep", extra=to)
        if action == "buy_from" and a["inv"]["coin"] < FOOD_PRICE:
            return self._receipt(a, eid, action, False, "no coin", extra=to)
        if t["inv"]["food"] <= 1:
            return self._receipt(a, eid, action, False, "nothing to spare", extra=to)
        kin = self._kin(a, t["id"])
        stream = f"share:{t['id']}:{a['id']}"
        idx = self.state["tick"]
        roll = self.rng.at(stream, idx, 0, 999)  # drawn as in Sim, so the other streams stay aligned; unused
        dec = self.mind.choose("share", {"traits": t["traits"], "kin": kin, "affinity": self._rel(t, a["id"])["affinity"],
                                         "food": t["inv"]["food"], "action": action})
        self._choice(t, "share", {"asker": a["id"], "kin": kin, "action": action}, dec, [eid])
        limit = 1000 if dec["give"] else 0
        extra = {"stream": stream, "index": idx, "roll": roll, "limit": limit, "kin": kin, "decided": "mind", **to}
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
        self._note(t["id"], f"refused to give food to {a['name']}")
        r["affinity"] = max(-1000, r["affinity"] - 50)
        rid = self._receipt(a, eid, action, False, "refused", "chance", extra)
        self._occasion(a, "refused", 450 + (1000 - a["traits"]["agreeableness"]) // 4, t["id"], rid)
        return rid

    # ---- D2, D5, D6 hang on the plan's blocks
    def _act(self, a):
        day, hour = self.state["tick"] // TICKS_PER_DAY, self._hour()
        author, step, holder = self._running(a)
        if step is None and self.on("work_end") and a["role"] not in ("child", "elder"):
            self._work_end(a, day, hour)
        before = a["activity"]
        super()._act(a)
        if self.on("company") and a["activity"] != "socialize":
            a["social_place"] = None
        if (self.on("path") and before != "work" and a["activity"] == "work" and a["role"] not in ("child", "elder")
                and a.get("path_day") != day):
            self._path(a, day)

    def _work_end(self, a, day, hour):
        for i, b in enumerate(a["plan"]):
            if b["activity"] != "work" or b["end_h"] != hour or "go_on" in b or b["start_h"] >= hour:
                continue
            day_start = self.state["tick"] - self.state["tick"] % TICKS_PER_DAY
            bed_h = min(24, (a["sleep_tick"] - day_start) * 10 // 60)
            if hour + 1 > bed_h:
                continue
            dec = self.mind.choose("work_end", {"traits": a["traits"], "coin": a["inv"]["coin"], "food": a["inv"]["food"],
                                                "hour": hour, "bed_h": bed_h})
            b["go_on"] = dec["go_on"]
            self._choice(a, "work_end", {"hour": hour, "coin": a["inv"]["coin"]}, dec, [a["plan_event"]])
            if dec["go_on"]:
                b["end_h"] += 1
                for later in a["plan"][i + 1:]:  # the evening shifts by the hour
                    later["start_h"] = min(23, later["start_h"] + 1)
                    later["end_h"] = min(24, max(later["end_h"] + 1, later["start_h"] + 1))
                self._note(a["id"], "kept working an hour past the end of the planned block")
            else:
                self._note(a["id"], "stopped work when the planned block ended")
            return

    def _path(self, a, day):
        a["path_day"] = day
        flip = self.rng.at(f"path:{a['id']}", day, 0, 999) < 500
        dec = self.mind.choose("path", {"traits": a["traits"], "coin_flip": flip})
        outcome = "road"
        if dec["risky"]:
            r = self.rng.at(f"woods:{a['id']}", day, 0, 999)
            if r < WOODS_FIND:
                a["inv"]["food"] += 1
                outcome = "found 1 food"
            elif r >= WOODS_LOSS and a["inv"]["coin"] > 0:
                a["inv"]["coin"] -= 1
                outcome = "lost 1 coin"
            else:
                outcome = "nothing happened"
            self._note(a["id"], f"took the short cut through the woods to the {a['place']} ({outcome})")
        else:
            self._note(a["id"], f"took the road to the {a['place']}")
        self._choice(a, "path", {"outcome": outcome, "coin_flip": flip}, dec)

    def _place(self, a, activity):
        if activity == "socialize" and self.on("company"):
            if a["activity"] != "socialize" or not a.get("social_place"):  # the block starts now: choose company
                self._company(a)
            return a["social_place"]
        return super()._place(a, activity)

    def _company(self, a):
        day = self.state["tick"] // TICKS_PER_DAY
        agents = self.state["agents"]
        rel = a["relationships"]
        friend = max((int(k) for k in rel), default=None, key=lambda i: (rel[str(i)]["affinity"], -i))
        strangers = [o["id"] for o in agents if o["id"] != a["id"] and str(o["id"]) not in rel]
        new = strangers[self.rng.at(f"company:{a['id']}", self.state["tick"], 0, len(strangers) - 1)] if strangers else None
        dec = self.mind.choose("company", {"traits": a["traits"], "day": day, "friend": friend, "new": new})
        a["social_place"] = dec["where"]
        who = dec["who"]
        showed = who is not None and agents[who]["place"] == dec["where"] and not agents[who]["asleep"]
        self._choice(a, "company", {"friend": friend, "new": new, "showed": showed}, dec, [a["plan_event"]] if a["plan_event"] is not None else [])
        if who is not None:
            self._note(a["id"], f"went to the {dec['where']} hoping to see {agents[who]['name']}"
                                + (" (a new face)" if dec["new_face"] else ""))
        else:
            self._note(a["id"], f"went to the {dec['where']} for company")

    # ---- D3: surplus, decided at bedtime
    def _sleep(self, a, occupancy):
        if self.on("surplus"):
            self._surplus(a)
        super()._sleep(a, occupancy)

    def _surplus(self, a):
        food_surplus, coin_surplus = a["inv"]["food"] - FOOD_KEEP, a["inv"]["coin"] - COIN_KEEP
        if food_surplus <= 0 and coin_surplus <= 0:
            return
        agents = self.state["agents"]
        fam = [m for m in self.state["families"][a["family"]]["members"] if m != a["id"]]
        known = [int(k) for k in a["relationships"] if int(k) not in fam]
        needy = [i for i in fam + known if agents[i]["inv"]["food"] <= 1]
        to = min(needy, key=lambda i: (agents[i]["inv"]["food"], i)) if needy else None
        dec = self.mind.choose("surplus", {"traits": a["traits"], "values": a["values"], "food_surplus": food_surplus,
                                           "coin_surplus": coin_surplus, "to": to})
        facts = {"food": a["inv"]["food"], "coin": a["inv"]["coin"], "to": to if dec["do"] == "give" else None}
        cid = self._choice(a, "surplus", facts, dec)
        if dec["do"] == "give":
            self._give(a, agents[to], "give_food" if food_surplus > 0 else "give_coin", cid)
        elif dec["do"] == "sell":
            a["inv"]["food"] -= food_surplus
            a["inv"]["coin"] += food_surplus * FARMER_SELL_PRICE
            self.state["market_food"] += food_surplus
            self._emit("sell", "rule", a["id"], {"food": food_surplus}, [cid])
            self._note(a["id"], f"sold {food_surplus} spare food at the market")
        else:
            self._note(a["id"], "kept what was spare")

    # ---- D4: pass a thief belief on, or keep it
    def _talk(self):
        if not self.on("gossip"):
            return super()._talk()
        from collections import defaultdict
        agents = self.state["agents"]
        by_place = defaultdict(list)
        for a in agents:
            if not a["asleep"] and a["place"] not in (a["home"],) and a["activity"] != "work":
                by_place[a["place"]].append(a["id"])
        for place in sorted(by_place):
            ids = by_place[place]
            if len(ids) < 2:
                continue
            tick = self.state["tick"]
            for aid in self._order(ids):
                a = agents[aid]
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
                transfers, sources = [], []
                for x, y in ((a, b), (b, a)):  # y holds the belief, x may take it up
                    for k in sorted(y["beliefs"]):
                        if k.startswith("thief:") and k not in x["beliefs"] and int(k.split(":")[1]) != x["id"]:
                            hearsay = "(heard" in y["beliefs"][k]
                            dec = self.mind.choose("gossip", {"traits": y["traits"], "hearsay": hearsay,
                                                              "kin": self._kin(y, x["id"])})
                            self._choice(y, "gossip", {"to": x["id"], "about": int(k.split(":")[1]), "hearsay": hearsay}, dec)
                            if not dec["pass"]:
                                self._note(y["id"], f"kept what I'd heard about {agents[int(k.split(':')[1])]['name']} to myself")
                                continue
                            self._note(y["id"], f"told {x['name']} about {agents[int(k.split(':')[1])]['name']}")
                            c_roll = self.rng.at(f"cred:{x['id']}:{y['id']}:{k}", tick, 0, 999)
                            if c_roll < x["cognition"]["credulity"]:
                                transfers.append((x, y, k))
                                if y["belief_causes"].get(k) is not None:
                                    sources.append(y["belief_causes"][k])
                tid = self._emit("talk", "chance", aid, {**draws, "with": b["id"], "place": place,
                                                         "gossip": [[x["id"], k] for x, _, k in transfers]},
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


class WiredArmSim(WiredSim, ArmSim):
    """The wired sim under a scorecard arm (famine: the T1m famine rules, so the world creates asks and thefts)."""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", default="all", help="all, none, or a comma list of " + ",".join(CHANNELS))
    ap.add_argument("--famine", action="store_true", help="run under the scorecard's famine arm")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fire", default="3@1", help="player fire as FAMILY@DAY, or none")
    args = ap.parse_args()
    channels = CHANNELS if args.channels == "all" else () if args.channels == "none" else tuple(args.channels.split(","))
    for c in channels:
        if c not in CHANNELS:
            raise SystemExit(f"unknown channel {c}")
    pop = labeled_population(args.seed, args.agents)
    mind = WiredStub() if channels else StubMind()
    if args.famine:
        sim = WiredArmSim(args.seed, args.agents, mind, genesis=pop, channels=channels, arm="famine", souls={})
    else:
        sim = WiredSim(args.seed, args.agents, mind, genesis=pop, channels=channels)
    player = {}
    if args.fire and args.fire != "none":
        fam, day = (int(x) for x in args.fire.split("@"))
        player[day * TICKS_PER_DAY + 60] = [{"kind": "fire", "family": fam}]
    t0 = time.monotonic()
    sim.run(args.days, player)
    sim.finish()
    os.makedirs(args.out, exist_ok=True)
    with gzip.open(os.path.join(args.out, "events.jsonl.gz"), "wt") as f:
        for e in sim.events:
            f.write(e.to_json() + "\n")
    from collections import Counter
    choices = Counter(e.data["what"] for e in sim.events if e.kind == "choice")
    info = {"sandbox": "T24 part 1.3, never merges", "channels": list(channels), "famine": args.famine, "days": args.days,
            "agents": args.agents, "seed": args.seed, "fire": args.fire, "events": len(sim.events), "final_hash": sim.state_hash(),
            "choices": dict(choices), "wall_s": round(time.monotonic() - t0, 1), "mind": sim.mind.model}
    with open(os.path.join(args.out, "run.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
