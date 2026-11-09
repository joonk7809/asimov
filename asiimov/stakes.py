"""SANDBOX (B5, stakes design). This module never merges; it exists to simulate stakes options on the stub.

Six levers, each a world rule, three of them with a one-off ask a mind would answer (a real trade: coin or
food on both sides, no plan-keeping option, no held-out cue word in the text; lessons L24, L25):

  price     the market's food price floats with its stock: 3 at a day of village need or more, 4 under a day,
            5 under half a day, 6 under a quarter; farmers sell at the price less 1
  floor     the pay spread: work progress per tick is pulled 60% of the way toward the working adults' mean
            (mean + 0.4 x (strength x stamina - mean)), so the village makes as much as before but the slowest
            body works at about 2.6x less than the fastest instead of 25x
  dues      a coin sink with a loop: each household owes the chest the larger of DUES_MIN coin and a tenth of
            the payer's coin, daily, and the chest, not thin air, pays the guards' wages (a unit the chest
            can't pay is not watched). ASK "dues": pay or skip (an unpaid home goes unwatched)
  tithe     a bounded sink, no ask: at midnight every adult gives the chest 15% of the coin it carries above 10;
            the chest pays guards (under dues) and spends what it holds above RESERVE on the village each night
  raids     a threat: on some nights raiders strike RAID_HOMES homes, likelier the richer (a keyed draw weighted
            by the household's coin and food); a home is watched if its household paid
            dues that day and the guards' work that day covers it; an unwatched home loses half its sleepers'
            food and a third of their coin
  seasons   a YEAR-day year: field yields by day of year (YIELDS), no harvest in winter, the stall doesn't
            spoil from the harvest night through winter. ASK "stock" on harvest day: lay in winter food or keep
            the coin
  lean      calibration, not a design option: every positive field yield less 1, so the stub's food margin
            (about 2x the village's meals) sits near what real 8B minds make (about 1.2x, T1n 8B full)
  sickness  a contagion: one adult a day may fall sick (keyed chance), and a player event can make anyone sick;
            it passes through talk, shared workplaces and homes; a sick body works at half pace for SICK_DAYS.
            ASK "sick" at waking: go to work at half pace (others may catch it) or stay home, earning nothing

With no lever on, StakesSim is Sim and StakesStub is StubMind: the same log for the same seed (tests in
tests/test_stakes.py). Every new flow is an event with its amounts, so StakesChecker keeps both ledgers.

    python -m asiimov.stakes --levers all --days 7 --out runs/stakes
"""
import argparse
import json
import os
from collections import defaultdict

from . import sim as S
from .invariants import Checker, log_violations, presence_violations, state_violations, totals
from .minds import StubMind
from .propagation import ACT_FIELDS
from .sim import TICKS_PER_DAY, WORKPLACE, Sim, deliberate_ask

LEVERS = ("price", "floor", "dues", "tithe", "raids", "seasons", "sickness", "lean")
BASE_PRICE, BASE_SELL = S.FOOD_PRICE, S.FARMER_SELL_PRICE
DUES_MIN, DUES_RATE = 3, 10   # a household pays the larger of 3 coin and a tenth of the payer's coin, daily
HOMES_PER_UNIT = 2            # homes one completed unit of guard work watches that night
TITHE_RATE, TITHE_FREE = 15, 10  # per cent of carried coin above 10, each midnight, to the chest
RESERVE = 400                 # the chest spends what it holds above this on the village each night (a sink)
RAID_P, RAID_P_WINTER = 350, 650   # per mille, per night
RAID_HOMES = 3
YEAR = 7
YIELDS = (4, 4, 4, 4, 5, 0, 0)     # food per unit of farm work by day of year; 0 is winter. The year makes what a
                                   # flat year at 3 makes (21 units), so winter is a timing pressure, not a cut
SICK_P = 300                  # per mille a day: one adult falls sick by chance
SICK_DAYS = 3
CATCH_TALK, CATCH_WORK, CATCH_HOME = 100, 3, 100  # per mille: a chat, an hour at a shared place, a night
ASKS = {"dues": ("pay", "skip"), "stock": ("stock_up", "keep"), "sick": ("work", "home")}

S.OCCASIONS.update({"dues": "{detail}", "stock": "{detail}", "sick": "{detail}",
                    "raided": "raiders broke into your home in the night and took {detail}"})
CLAUSES = {
    "pay": "pay gives the dues to the village chest now",
    "skip": "skip keeps the coin",
    "stock_up": "stock_up goes to the market and buys food to carry through the days without harvest",
    "keep": "keep keeps the coin",
    "work": "work goes to your work today at half pace",
    "home": "home keeps you at home today, earning nothing",
}
ACT_FIELDS.update({"dues": ("coin",), "stake": ("what", "pick", "food"), "tithe": ("coin",)})


def band(v: int) -> str:
    return "low" if v < 350 else "high" if v > 650 else "mid"


class StakesSim(Sim):
    def __init__(self, seed, n_agents, mind, levers=(), genesis=None, expected=None, order="id", workers=1):
        bad = set(levers) - set(LEVERS)
        if bad:
            raise ValueError(f"unknown levers {sorted(bad)}")
        self.levers = frozenset(levers)
        super().__init__(seed, n_agents, mind, genesis=genesis, expected=expected, order=order, workers=workers)
        if self.levers:
            self.state["stakes"] = {"chest": 0, "price": BASE_PRICE, "dues": {}, "watch": {}, "sick": {},
                                    "immune": {}}
            for a in self.state["agents"]:
                a["stake"] = {"pending": "", "asked": {}, "order": 0}

    def on(self, lever: str) -> bool:
        return lever in self.levers

    def _emit(self, kind, origin, actor, data, parents=(), mind=None):
        if kind == "genesis" and self.levers:
            data = {**data, "stakes": sorted(self.levers)}
        return super()._emit(kind, origin, actor, data, parents, mind)

    # ------------------------------------------------------------------ prices and seasons
    def day(self) -> int:
        return self.state["tick"] // TICKS_PER_DAY

    def season_day(self) -> int:
        return self.day() % YEAR

    def winter(self, day: int | None = None) -> bool:
        return self.on("seasons") and YIELDS[(self.day() if day is None else day) % YEAR] == 0

    def _need1(self) -> int:
        if not hasattr(self, "_need_cache"):
            self._need_cache = self._market_cap() // S.MARKET_KEEP_DAYS
        return self._need_cache

    def _price_now(self) -> int:
        if not self.on("price"):
            return BASE_PRICE
        s, n = self.state["market_food"], self._need1()
        return 3 if s >= n else 4 if 2 * s >= n else 5 if 4 * s >= n else 6

    def step(self, player_actions=()):
        if not self.levers:
            return super().step(player_actions)
        st, tick = self.state["stakes"], self.state["tick"]
        st["price"] = self._price_now()
        S.FOOD_PRICE, S.FARMER_SELL_PRICE = st["price"], st["price"] - 1
        try:
            if tick % TICKS_PER_DAY == 0 and tick:
                if self.on("tithe"):
                    self._tithe()
                if self.on("raids"):
                    self._raid()
            if self.on("sickness"):
                self._sickness()
            super().step(player_actions)
        finally:
            S.FOOD_PRICE, S.FARMER_SELL_PRICE = BASE_PRICE, BASE_SELL

    def _food_yield(self, a):
        y = YIELDS[self.season_day()] if self.on("seasons") else super()._food_yield(a)
        return max(0, y - 1) if self.on("lean") and y > 0 else y

    def _tithe(self):
        st = self.state["stakes"]
        for a in self.state["agents"]:
            k = max(0, a["inv"]["coin"] - TITHE_FREE) * TITHE_RATE // 100
            if k and a["role"] not in ("child", "elder"):
                a["inv"]["coin"] -= k
                st["chest"] += k
                self._emit("tithe", "rule", a["id"], {"coin": k})

    def _spend_chest(self):
        st = self.state["stakes"]
        if st["chest"] > RESERVE:
            self._emit("chest_spent", "rule", None, {"coin": st["chest"] - RESERVE})
            st["chest"] = RESERVE

    def _spoil(self):
        if self.on("dues") or self.on("tithe"):
            self._spend_chest()
        # from the harvest night through winter the stall keeps what it holds
        if self.on("seasons") and (self.winter() or self.winter(self.day() + 1)):
            return
        super()._spoil()

    # ------------------------------------------------------------------ work: the floor, sickness, the chest
    def _rate(self, a) -> int:
        r = a["body"]["strength"] * a["body"]["stamina"] // 1000
        if self.on("floor"):  # compress toward the working adults' mean, so the village makes as much as before
            if not hasattr(self, "_mean_rate"):
                xs = [b["body"]["strength"] * b["body"]["stamina"] // 1000 for b in self.state["agents"]
                      if b["role"] not in ("child", "elder")]
                self._mean_rate = sum(xs) // len(xs)
            r = self._mean_rate + (r - self._mean_rate) * 4 // 10
        if self.on("sickness") and str(a["id"]) in self.state["stakes"]["sick"]:
            r //= 2
        return r

    def _act(self, a: dict):
        """Sim._act with the work rate from _rate and, under dues, guards paid from the chest."""
        if not self.levers:
            return super()._act(a)
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
            if activity == "shop" and step is None and a["place"] == "market":
                self._buy(a, parent)
        if activity == "work" and a["role"] not in ("child", "elder") and a["place"] == WORKPLACE[a["role"]]:
            a["work_progress"] += self._rate(a)
            if a["work_progress"] >= 3000:
                a["work_progress"] -= 3000
                if a["role"] == "farmer":
                    food = self._harvest(a)
                    if food:
                        a["inv"]["food"] += food
                        self._emit("produce", "rule", a["id"], {"food": food}, [a["plan_event"]])
                        self._farmer_sells(a)
                elif a["role"] == "guard" and self.on("dues"):
                    st = self.state["stakes"]
                    pay = min(S.WORK_WAGE, st["chest"])
                    st["chest"] -= pay
                    a["inv"]["coin"] += pay
                    if pay:
                        w = st["watch"]
                        w[str(self.day())] = w.get(str(self.day()), 0) + 1
                    self._emit("produce", "rule", a["id"], {"coin": pay, "chest": pay}, [a["plan_event"]])
                else:
                    a["inv"]["coin"] += S.WORK_WAGE
                    self._emit("produce", "rule", a["id"], {"coin": S.WORK_WAGE}, [a["plan_event"]])
                    if a["role"] == "guard":
                        w = self.state["stakes"]["watch"]
                        w[str(self.day())] = w.get(str(self.day()), 0) + 1
        if step is not None:
            self._progress(a, author, step, holder)

    def _economy(self, a: dict) -> str:
        s = super()._economy(a)
        if not self.levers:
            return s
        rate = max(1, self._rate(a))
        hours = max(1, (-(-3000 // rate) * 10 + 30) // 60)
        old = max(1, (-(-3000 // max(1, a["body"]["strength"] * a["body"]["stamina"] // 1000)) * 10 + 30) // 60)
        s = s.replace(f"about every {old} hours", f"about every {hours} hours")
        if self.on("seasons"):
            left = next((k for k in range(YEAR) if YIELDS[(self.day() + k) % YEAR] == 0), None)
            if self.winter():
                back = next(k for k in range(1, YEAR + 1) if YIELDS[(self.day() + k) % YEAR] > 0)
                s += f" It is winter: the fields give no food for {back} more days."
            elif left is not None:
                s += f" Winter starts in {left} days; the fields give no food in winter."
        return s

    # ------------------------------------------------------------------ threats
    def _raid(self):
        d = self.day()
        st = self.state["stakes"]
        p = RAID_P_WINTER if self.winter() else RAID_P
        if self.rng.at("raid", d, 0, 999) >= p:
            return
        fams = self.state["families"]
        ag = self.state["agents"]
        worth = {f: sum(ag[m]["inv"]["coin"] + 3 * ag[m]["inv"]["food"] for m in fams[f]["members"])
                 for f in range(len(fams))}
        targets = sorted(range(len(fams)), key=lambda f: (self.rng.raw(f"raid-pick:{d}", f) // (worth[f] + 10), f))
        targets = targets[:RAID_HOMES]
        y = str(d - 1)
        paid = [f for f in range(len(fams)) if st["dues"].get(str(f)) == d - 1] if self.on("dues") else []
        cap = st["watch"].get(y, 0) * HOMES_PER_UNIT
        watched = set(sorted(paid, key=lambda f: (self.rng.raw(f"watch:{y}", f), f))[:cap])
        for f in sorted(targets):
            if f in watched:
                eid = self._emit("raid", "chance", None, {"family": f, "watched": 1, "food": 0, "coin": 0})
                for m in fams[f]["members"]:
                    self._note(m, "the night watch drove raiders off from our home")
                continue
            food = coin = 0
            took = []
            for m in fams[f]["members"]:
                a = self.state["agents"][m]
                if not a["asleep"] or a["place"] != a["home"]:
                    continue
                lf, lc = (a["inv"]["food"] + 1) // 2, a["inv"]["coin"] // 3
                a["inv"]["food"] -= lf
                a["inv"]["coin"] -= lc
                food, coin = food + lf, coin + lc
                took.append((a, lf, lc))
            eid = self._emit("raid", "chance", None, {"family": f, "watched": 0, "food": food, "coin": coin})
            for a, lf, lc in took:
                if lf:
                    a["last_food_loss_event"] = eid
                self._note(a["id"], f"raiders took {lf} food and {lc} coin from our home")
                self._occasion(a, "raided", 900, event=eid, detail=f"{lf} food and {lc} coin")

    def _make_sick(self, a: dict, parents) -> int | None:
        st, aid = self.state["stakes"], str(a["id"])
        if aid in st["sick"] or aid in st["immune"]:
            return None
        st["sick"][aid] = self.state["tick"] + SICK_DAYS * TICKS_PER_DAY
        self._note(a["id"], "fell sick")
        eid = self._emit("sick", "chance", None, {"agent": a["id"]}, parents)
        st.setdefault("sick_ev", {})[aid] = eid  # the source of whoever catches it from this agent
        return eid

    def _sickness(self):
        st, tick, ag = self.state["stakes"], self.state["tick"], self.state["agents"]
        for aid in sorted(st["sick"], key=int):
            if tick >= st["sick"][aid]:
                del st["sick"][aid]
                st["immune"][aid] = tick
                self._note(int(aid), "got over the sickness")
                self._emit("recover", "rule", None, {"agent": int(aid)})
        d = self.day()
        if tick % TICKS_PER_DAY == 36 and self.rng.at("sick-chance", d, 0, 999) < SICK_P:  # 06:00
            adults = [a for a in ag if a["role"] not in ("child", "elder")]
            self._make_sick(adults[self.rng.at("sick-pick", d, 0, len(adults) - 1)], [])
        if tick % 6 or not st["sick"]:
            return
        by_place = defaultdict(list)
        for a in ag:
            if a["place"] is not None and (not a["asleep"] or tick % TICKS_PER_DAY == 0):
                by_place[a["place"]].append(a)
        home_hour = tick % TICKS_PER_DAY == 0  # midnight: households share the night
        for place in sorted(by_place):
            here = by_place[place]
            sick = [a for a in here if str(a["id"]) in st["sick"]]
            if not sick or (place.startswith("home_") and not home_hour):
                continue
            p = CATCH_HOME if place.startswith("home_") else CATCH_WORK
            for b in here:
                if str(b["id"]) in st["sick"]:
                    continue
                if self.rng.at(f"catch:{b['id']}", tick, 0, 999) < p * len(sick):
                    src = min(a["id"] for a in sick)
                    self._make_sick(b, [st["sick_ev"][str(src)]])
                    self._note(b["id"], f"caught a sickness at {place.replace('_', ' ')} near {ag[src]['name']}")

    def _talk(self):
        n0 = len(self.events)
        super()._talk()
        if not self.on("sickness"):
            return
        st = self.state["stakes"]
        for e in list(self.events[n0:]):
            if e.kind != "talk":
                continue
            for x, y in ((e.actor, e.data["with"]), (e.data["with"], e.actor)):
                if str(x) in st["sick"] and str(y) not in st["sick"] \
                        and self.rng.at(f"catch-talk:{y}", e.tick, 0, 999) < CATCH_TALK:
                    self._make_sick(self.state["agents"][y], [e.id, st["sick_ev"][str(x)]])

    def _player(self, act: dict):
        if act["kind"] == "sick":
            return self._make_sick(self.state["agents"][act["agent"]], [])
        if act["kind"] == "raid":
            return self._raid_now(act)
        return super()._player(act)

    def _raid_now(self, act):
        """A player-forced raid on one family tonight's terms (for twins): unwatched, as if no watch."""
        f = act["family"]
        food = coin = 0
        took = []
        for m in self.state["families"][f]["members"]:
            a = self.state["agents"][m]
            lf, lc = (a["inv"]["food"] + 1) // 2, a["inv"]["coin"] // 3
            a["inv"]["food"] -= lf
            a["inv"]["coin"] -= lc
            food, coin = food + lf, coin + lc
            took.append((a, lf, lc))
        eid = self._emit("raid", "player", None, {"family": f, "watched": 0, "food": food, "coin": coin})
        for a, lf, lc in took:
            a["last_food_loss_event"] = eid
            self._occasion(a, "raided", 900, event=eid, detail=f"{lf} food and {lc} coin")
        return eid

    # ------------------------------------------------------------------ the asks
    def _due_ask(self, a: dict) -> tuple[str, str] | None:
        """The one-off ask due for this agent now, if any: (why, detail). Facts only (L11, L25)."""
        if a["role"] in ("child", "elder") or a["asleep"]:
            return None
        st, d, tick = self.state["stakes"], self.day(), self.state["tick"]
        asked = a["stake"]["asked"]
        woke = tick - (a["sleep_tick"] - (1440 - a["rhythm"]["sleep_need_min"]) // 10)  # ticks since waking
        price = st["price"]
        if self.on("sickness") and str(a["id"]) in st["sick"] and asked.get("sick") != d and woke >= 1:
            pay = "about half the food" if a["role"] == "farmer" else "about half the coin"  # no cue words (L25)
            return "sick", (f"you are sick; it passes in about {-(-(st['sick'][str(a['id'])] - tick) // TICKS_PER_DAY)}"
                            f" days. At work today you would make {pay} you make when well, and people near you"
                            f" can catch it. Answering work sends you to the {WORKPLACE[a['role']]} at half pace;"
                            f" answering home keeps you at home all day, earning nothing, and nobody catches it"
                            f" from you")
        dues = self._dues(a)
        if (self.on("dues") and st["dues"].get(str(a["family"])) != d and asked.get("dues") != d and woke >= 6
                and a["inv"]["coin"] >= dues):
            raids = sum(1 for e in self.events[-4000:] if e.kind == "raid" and e.tick >= tick - 3 * TICKS_PER_DAY
                        and not e.data["watched"])
            return "dues", (f"your household owes the village chest {dues} coin today, a tenth of what you carry"
                            f" and at least {DUES_MIN}. The chest pays the guards,"
                            f" and tonight they watch only homes whose household has paid. Raiders pick homes holding more food"
                            f" and coin more often. Raiders took food and coin"
                            f" from {raids} unwatched homes in the last 3 nights. You carry {a['inv']['coin']} coin;"
                            f" food costs {price} coin. Answering pay gives {dues} coin to the chest now; answering"
                            f" skip keeps the coin, and your home goes unwatched tonight unless someone in your"
                            f" household pays")
        if (self.on("seasons") and YIELDS[self.season_day()] > 0 and YIELDS[(d + 1) % YEAR] == 0
                and asked.get("stock") != d and woke >= 12 and a["inv"]["coin"] >= price):
            wd = next(k for k in range(1, YEAR + 1) if YIELDS[(d + k) % YEAR] > 0) - 1
            n = self._stock_need(a, wd)
            return "stock", (f"the harvest ends today; for the next {wd} days the fields give no food and the"
                             f" market sells only what it holds, now {self.state['market_food']} food, about"
                             f" {self.state['market_food'] * 10 // max(1, self._need1()) / 10} days of the"
                             f" village's meals, at {price} coin each; its price goes up as it runs low."
                             f" Your household eats about {n} meals in {wd} days. You carry {a['inv']['coin']} coin"
                             f" and {a['inv']['food']} food. Answering stock_up takes you to the market to buy up"
                             f" to {n} food now at {price} coin; answering keep keeps the coin for later")
        return None

    def _dues(self, a: dict) -> int:
        return max(DUES_MIN, a["inv"]["coin"] // DUES_RATE)

    def _stock_need(self, a: dict, days: int) -> int:
        fam = self.state["families"][a["family"]]["members"]
        mine = [m for m in fam if m == a["id"] or self.state["agents"][m]["role"] in ("child", "elder")]
        return 3 * days * len(mine)

    def _maybe_deliberate(self, a: dict, occupancy: dict):
        if self.levers and not (a["occasion"] and a["occasion"]["why"] in ASKS):
            due = self._due_ask(a)
            if due:
                why, detail = due
                a["stake"]["asked"][why] = self.day()
                self._occasion(a, why, a["cognition"]["deliberation_threshold"], detail=detail,
                               event=a["plan_event"])
        super()._maybe_deliberate(a, occupancy)

    def _offer(self, a: dict, view: dict) -> list[str]:
        why = (view.get("occasion") or {}).get("why")
        if self.levers:
            a["stake"]["pending"] = why if why in ASKS else ""
        if why in ASKS:
            return list(ASKS[why])
        return super()._offer(a, view)

    def _situation(self, a: dict, view: dict, kind: str) -> str:
        why = (view.get("occasion") or {}).get("why")
        if kind != "deliberate" or why not in ASKS:
            return super()._situation(a, view, kind)
        saved = dict(S.ACTION_CLAUSES)
        S.ACTION_CLAUSES.update({k: CLAUSES[k] for k in ASKS[why]})
        try:
            text = super()._situation(a, view, kind)
            tail = deliberate_ask(view["offer"])
        finally:
            S.ACTION_CLAUSES.clear()
            S.ACTION_CLAUSES.update(saved)
        assert text.endswith(tail)
        clauses = ". ".join(CLAUSES[x] for x in view["offer"]) + "."
        return text[:-len(tail)] + " Your action happens now; target is null. " + clauses + " Leave goal empty and then []."

    def _resolve(self, a: dict, out, eid: int):
        why = a["stake"]["pending"] if self.levers else ""
        if not why:
            return super()._resolve(a, out, eid)
        a["stake"]["pending"] = ""
        pick = (out or {}).get("action")
        pick = pick if pick in ASKS[why] else None
        data = {"what": why, "pick": pick, "food": 0}
        st = self.state["stakes"]
        if why == "dues" and pick == "pay" and st["dues"].get(str(a["family"])) != self.day():
            k = min(self._dues(a), a["inv"]["coin"])
            a["inv"]["coin"] -= k
            st["chest"] += k
            st["dues"][str(a["family"])] = self.day()
            self._emit("dues", "rule", a["id"], {"coin": k}, [eid])
            self._note(a["id"], f"paid {k} coin to the village chest")
        elif why == "stock" and pick == "stock_up":
            wd = next(k for k in range(1, YEAR + 1) if YIELDS[(self.day() + k) % YEAR] > 0) - 1
            a["stake"]["order"] = self._stock_need(a, wd)
            a["override"] = {"do": "shop", "target": None, "left": S.DEFAULT_STEP_TICKS, "event": eid}
        elif why == "sick" and pick == "home":
            left = max(1, a["sleep_tick"] - self.state["tick"])
            a["override"] = {"do": "home", "target": None, "left": left, "event": eid}
        self._emit("stake", "rule", a["id"], data, [eid])

    def _buy(self, a: dict, eid: int):
        n = a["stake"]["order"] if self.levers else 0
        if not n or a["place"] != "market":
            return super()._buy(a, eid)
        a["stake"]["order"] = 0
        price = self.state["stakes"]["price"]
        k = min(n, a["inv"]["coin"] // price, self.state["market_food"])
        self.state["market_food"] -= k
        a["inv"]["coin"] -= k * price
        a["inv"]["food"] += k
        self._note(a["id"], f"bought {k} food at the market for the winter")
        self._receipt(a, eid, "buy_food", True, "bought" if k else "had enough", extra={"food": k, "stock": 1})
        return True, "bought"


class StakesStub(StubMind):
    """The stub answers each ask by one rule on its own params, so it stays the format's reference."""
    model = "stub-stakes-v0"

    def _deliberate(self, v: dict) -> dict:
        why = (v.get("occasion") or {}).get("why")
        if why not in ASKS:
            return super()._deliberate(v)
        t = v["traits"]
        if why == "dues":  # pays if it can still eat, unless dishonest
            dues = max(DUES_MIN, v["coin"] // DUES_RATE)
            pick = "pay" if v["coin"] - dues >= v["food_price"] and t["honesty"] >= 350 else "skip"
        elif why == "stock":  # lays in when it plans ahead
            pick = "stock_up" if t["conscientiousness"] > 500 else "keep"
        else:  # sick: the agreeable stay home unless broke
            pick = "home" if t["agreeableness"] > 500 and v["coin"] >= 2 * v["food_price"] else "work"
        return {"action": pick, "target": None, "reason": "", "goal": "", "then": []}


class StakesChecker(Checker):
    """Checker with the stakes flows in both ledgers (prices read from the tick's price)."""

    def before(self, acts=()):
        super().before([x for x in acts if x["kind"] == "fire"])
        self._chest = self.sim.state.get("stakes", {}).get("chest", 0)

    def after(self) -> list[str]:
        sim, new = self.sim, self.sim.events[self._n0:]
        price = sim.state.get("stakes", {}).get("price", BASE_PRICE)
        g = lambda k, f: sum(e.data.get(f, 0) for e in new if e.kind == k)  # noqa: E731
        produced, eaten = g("produce", "food"), sum(1 for e in new if e.kind == "eat")
        spoiled = sum(e.data["market_food_lost"] for e in new if e.kind in ("spoil", "blight"))
        bought = sum(e.data["food"] for e in new if e.kind == "receipt" and e.data["action"] == "buy_food"
                     and e.data["ok"])
        want_food = produced - eaten - self._burned - spoiled - g("raid", "food")
        want_coin = (g("produce", "coin") + (price - 1) * g("sell", "food") - price * bought - g("dues", "coin")
                     - g("raid", "coin") - g("tithe", "coin"))
        (food0, coin0), (food1, coin1) = self._totals, totals(sim)
        out = []
        if food1 - food0 != want_food:
            out.append(f"food ledger: {food1 - food0} vs {want_food}")
        if coin1 - coin0 != want_coin:
            out.append(f"coin ledger: {coin1 - coin0} vs {want_coin}")
        chest = sim.state.get("stakes", {}).get("chest", 0)
        if chest - self._chest != (g("dues", "coin") + g("tithe", "coin") - g("produce", "chest")
                                   - g("chest_spent", "coin")):
            out.append(f"chest ledger: {chest - self._chest}")
        out += state_violations(sim) + log_violations(sim.events, self._n0)
        out += presence_violations(sim.events, self._n0, self.where)
        if [a["traits"] for a in sim.state["agents"]] != self.traits:
            out.append("a mind changed someone's traits")
        return [f"tick {self._tick}: {v}" for v in out]


def parse_levers(s: str) -> tuple[str, ...]:
    return () if s in ("", "none") else LEVERS if s == "all" else tuple(x for x in s.split(",") if x)


# ---------------------------------------------------------------------------- evaluation ($0, stub)
def run_world(levers, seed=1, agents=100, days=7, player=None, check=False):
    sim = StakesSim(seed, agents, StakesStub(), levers=levers)
    checker = StakesChecker(sim) if check else None
    player = player or {}
    bad, daily = [], []
    while sim.state["tick"] < days * TICKS_PER_DAY:
        acts = player.get(sim.state["tick"], [])
        if checker:
            bad += checker.step(acts)
        else:
            sim.step(acts)
        if sim.state["tick"] % TICKS_PER_DAY == 0:
            ag = sim.state["agents"]
            daily.append({"coin": sum(a["inv"]["coin"] for a in ag), "food": sum(a["inv"]["food"] for a in ag),
                          "market": sim.state["market_food"],
                          "chest": sim.state.get("stakes", {}).get("chest", 0),
                          "price": sim.state.get("stakes", {}).get("price", BASE_PRICE)})
    sim.finish()
    return sim, bad, daily


def gini(xs) -> float:
    xs = sorted(xs)
    n, s = len(xs), sum(xs)
    return 0.0 if not s else round(sum((2 * i - n + 1) * x for i, x in enumerate(xs)) / (n * s), 3)


def health(sim, daily) -> dict:
    ev, ag = sim.events, sim.state["agents"]
    days = sim.state["tick"] // TICKS_PER_DAY
    earners = [a for a in ag if a["role"] not in ("child", "elder")]
    deps = [a for a in ag if a["role"] in ("child", "elder")]
    meals = defaultdict(int)
    earned = defaultdict(int)
    sell_coin = 0
    price_at = {}
    for e in ev:
        if e.kind == "eat":
            meals[e.actor] += 1
        elif e.kind == "produce" and "coin" in e.data:
            earned[e.actor] += e.data["coin"]
        elif e.kind == "sell":
            earned[e.actor] += e.data["food"] * 2  # approximate (price-1 at the tick); spread reading only
    fails = sum(1 for e in ev if e.kind == "receipt" and e.data.get("action") == "buy_food" and not e.data["ok"]
                and e.data["why"] in ("no stock", "no coin"))
    per = sorted(earned[a["id"]] / days for a in earners)
    p10, p90 = per[len(per) // 10], per[len(per) * 9 // 10]
    stakes = defaultdict(lambda: defaultdict(int))
    for e in ev:
        if e.kind == "stake":
            stakes[e.data["what"]][str(e.data["pick"])] += 1
    raids = [e for e in ev if e.kind == "raid"]
    sick_days = 0
    out = {
        "meals_per_earner_day": round(sum(meals[a["id"]] for a in earners) / (len(earners) * days), 2),
        "meals_per_dependant_day": round(sum(meals[a["id"]] for a in deps) / max(1, len(deps) * days), 2),
        "earners_under_2_meals_a_day": sum(1 for a in earners if meals[a["id"]] < 2 * days),
        "failed_buys_per_earner_day": round(fails / (len(earners) * days), 3),
        "coin_per_agent_end": sum(a["inv"]["coin"] for a in ag) // len(ag),
        "coin_growth_per_agent_day_last3": round((daily[-1]["coin"] - daily[-4]["coin"]) / 3 / len(ag), 1)
        if len(daily) >= 4 else None,
        "gini_adult_coin": gini([a["inv"]["coin"] for a in earners]),
        "earnings_p10_p90_per_day": [round(p10, 1), round(p90, 1)],
        "pay_spread_p90_p10": round(p90 / p10, 1) if p10 else None,
        "price_by_day": [d["price"] for d in daily], "market_by_day": [d["market"] for d in daily],
        "chest_by_day": [d["chest"] for d in daily],
        "raids": len(raids), "raids_unwatched": sum(1 for r in raids if not r.data["watched"]),
        "sick_events": sum(1 for e in ev if e.kind == "sick"),
        "asks": {k: dict(v) for k, v in stakes.items()},
        "mind_calls_per_agent_day": round(sum(1 for e in ev if e.origin == "mind") / (len(ag) * days), 2),
    }
    return out


def twin_reach(levers, act_tick, act, seed=1, agents=100, days=9):
    from .propagation import report
    base, _, _ = run_world(levers, seed, agents, days)
    twin, _, _ = run_world(levers, seed, agents, days, {act_tick: [act]})
    r = report(base.events, twin.events)
    return {w: {k: r["by_day"][w][k] for k in ("acted_differently", "beyond_targets", "beyond_targets_besides_chats",
                                                "beyond_targets_traced")} for w in r["by_day"]}


def main():
    ap = argparse.ArgumentParser(description="B5 stakes sandbox: health and twin reach on the stub")
    ap.add_argument("--levers", default="all")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--check", action="store_true", help="ledgers and invariants every tick")
    ap.add_argument("--reach", action="store_true", help="fire 3@1 twin (plus sick/raid twins when on)")
    ap.add_argument("--out")
    ap.add_argument("--yields", help="field yield by day of year, e.g. 3,3,3,3,5,0,0")
    args = ap.parse_args()
    levers = parse_levers(args.levers)
    if args.yields:
        global YIELDS, YEAR
        YIELDS = tuple(int(x) for x in args.yields.split(","))
        YEAR = len(YIELDS)
    res = {"levers": list(levers), "days": args.days, "seeds": {}}
    for seed in map(int, args.seeds.split(",")):
        sim, bad, daily = run_world(levers, seed, days=args.days, check=args.check)
        res["seeds"][seed] = {"violations": len(bad), "first": bad[:5], "hash": sim.state_hash(), **health(sim, daily)}
    if args.reach:
        t = TICKS_PER_DAY + 60
        res["reach"] = {"fire 3@1": twin_reach(levers, t, {"kind": "fire", "family": 3})}
        if "sickness" in levers:
            res["reach"]["sick #10@1"] = twin_reach(levers, t, {"kind": "sick", "agent": 10})
        if "raids" in levers:
            res["reach"]["raid 3@1"] = twin_reach(levers, t, {"kind": "raid", "family": 3})
        if "seasons" in levers:  # a fire on harvest day, the night before winter
            res["reach"]["fire 3@3"] = twin_reach(levers, 3 * TICKS_PER_DAY + 60, {"kind": "fire", "family": 3})
    print(json.dumps(res, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
