"""Words for what a player can see (readable-world E2, plan/readable-world-spec.md, Steering 2026-11-14).

Talk and snubs at the player's place, refusals of the player's asks and visits to the player get a line or
two of speech after the tick. They go to narration.jsonl and to the inhabitant as `heard`, and never into
the world: the log is the same as a world run with no narrator. The player's own villager never speaks."""
import asyncio
import json
import os

from asiimov import live, narrate
from asiimov.events import read_jsonl
from asiimov.minds import StubMind
from asiimov.sim import TICKS_PER_DAY, Sim
from test_inhabitant import batch_of, intent
from test_live import WS


class Ev:
    def __init__(self, id, kind, actor, data):
        self.id, self.kind, self.actor, self.data = id, kind, actor, data


def world_with_body(seed=4, n=20):
    sim = Sim(seed, n, StubMind())
    sim.run_until(TICKS_PER_DAY // 2)
    ag = sim.state["agents"]
    for b in ag:
        if b["asleep"] or b["place"] is None:
            continue
        others = [o["id"] for o in ag if o is not b and o["place"] == b["place"] and not o["asleep"]]
        if len(others) >= 2:
            return sim, b["id"], others
    raise AssertionError("no crowd")


def test_talk_here_gets_words_and_the_players_villager_never_speaks():
    sim, me, others = world_with_body()
    x, y = others[:2]
    evs = [Ev(1, "talk", x, {"with": y, "gossip": []}), Ev(2, "talk", x, {"with": me, "gossip": []}),
           Ev(3, "talk", me, {"with": y, "gossip": []}), Ev(4, "snub", x, {"with": me}),
           Ev(5, "snub", me, {"with": y})]
    its = narrate.items(sim, evs, me)
    assert [(i["kind"], i["speakers"], i["to_player"]) for i in its] == [
        ("talk", [x, y], False), ("talk", [x], True), ("talk", [y], True), ("snub", [x], True), ("snub", [y], True)]
    for it in its:
        assert me not in it["speakers"]
        for line in narrate.TemplateNarrator().lines(it):
            assert line["who"] in it["speakers"] and 0 < len(line["text"]) <= narrate.MAX_CHARS
        p = narrate.prompt(it)
        assert all(sim.state["agents"][s]["name"] in p for s in it["speakers"])
        assert "soul" not in p.lower() and "trait" not in p.lower() and "Lately" not in p


def test_talk_elsewhere_or_while_asleep_gets_no_words():
    sim, me, others = world_with_body()
    far = next(a["id"] for a in sim.state["agents"] if a["place"] not in (None, sim.state["agents"][me]["place"]))
    assert narrate.items(sim, [Ev(1, "talk", far, {"with": others[0], "gossip": []})], me) == []
    sim.state["agents"][me]["asleep"] = True
    assert narrate.items(sim, [Ev(1, "talk", others[0], {"with": others[1], "gossip": []})], me) == []


def test_a_refusal_of_the_players_ask_and_a_visit_get_the_other_villagers_line():
    sim, me, (x, *_) = world_with_body()
    evs = [Ev(1, "receipt", me, {"action": "ask_food", "ok": False, "why": "refused", "to": x}),
           Ev(2, "receipt", me, {"action": "ask_kin", "ok": False, "why": "kin has nothing to spare", "to": x}),
           Ev(3, "receipt", me, {"action": "ask_food", "ok": False, "why": "target asleep", "to": x}),
           Ev(4, "receipt", me, {"action": "ask_food", "ok": True, "why": "gave", "to": x}),
           Ev(5, "meet", x, {"with": me}), Ev(6, "meet", me, {"with": x})]
    its = narrate.items(sim, evs, me)
    assert [(i["event"], i["kind"], i["speakers"]) for i in its] == [(1, "refusal", [x]), (2, "refusal", [x]), (5, "visit", [x])]
    assert "nothing to spare" in narrate.prompt(its[1])


def test_the_model_narrator_keeps_good_lines_and_falls_back_on_anything_else():
    sim, me, others = world_with_body()
    x, y = others[:2]
    it = narrate.items(sim, [Ev(1, "talk", x, {"with": y, "gossip": []})], me)[0]
    nx, ny = sim.state["agents"][x]["name"], sim.state["agents"][y]["name"]
    n = narrate.LLMNarrator("http://127.0.0.1:9", "m")
    replies = [json.dumps({"lines": [{"who": nx, "text": "Morning."}, {"who": ny, "text": "Morning to you."}]}),
               json.dumps({"lines": [{"who": "Somebody Else", "text": "Hi"}]}),  # not a speaker
               "not json", json.dumps({"lines": []})]
    for i, text in enumerate(replies):
        n.client.post = lambda path, body, t=text: {"choices": [{"message": {"content": t}}]}
        got = n.lines(dict(it))
        if i == 0:
            assert got == [{"who": x, "text": "Morning."}, {"who": y, "text": "Morning to you."}]
        else:
            assert got == narrate.TemplateNarrator().lines(it)

    def down(path, body):
        raise RuntimeError("LLM server unreachable")
    n.client.post = down
    got = dict(it)
    assert n.lines(got) == narrate.TemplateNarrator().lines(it) and got["fallback"] is True


def test_an_inhabitant_hears_words_and_the_world_log_is_unchanged(tmp_path):
    out = str(tmp_path / "w")
    runner = live.LiveRunner(out, StubMind(), 4, 20)
    for _ in range(TICKS_PER_DAY // 2 - 10):
        runner.step()
    aid = 3
    heard = []

    async def main():
        ready = asyncio.get_running_loop().create_future()
        task = asyncio.ensure_future(live.serve(runner, port=0, tps=1000, paused=True, narrator=narrate.TemplateNarrator(),
                                                started=lambda port, hub: ready.set_result(port)))
        port = await ready
        c = await WS.connect(port)
        c.send({"type": "hello", "protocol": 2, "scope": "inhabitant", "agent": aid})
        await c.until("welcome")
        god = await WS.connect(port)
        god.send({"type": "hello", "protocol": 2})
        await god.until("welcome")
        god.send({"type": "control", "op": "run"})
        k = 0
        while True:
            m = await c.recv()
            if m["type"] == "heard":
                heard.append(m)
            if m["type"] == "tick":
                v = m["view"]
                k += 1
                if v["present"] and not v["asleep"] and k % 3 == 0:  # ask someone here for food now and then
                    c.send(intent(f"a{k}", "act", {"action": "ask_food", "target": v["present"][0]["id"]}, v["tick"]))
                if v["tick"] >= TICKS_PER_DAY + 30:
                    break
        god.send({"type": "control", "op": "pause"})
        await god.until("status")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    runner.finish()
    assert len(heard) >= 2 and {h["kind"] for h in heard} <= {"talk", "snub", "refusal", "visit"}
    for h in heard:
        assert h["lines"] and all(x["who"] != aid and x["name"] and x["text"] for x in h["lines"])
    side = [json.loads(l) for l in open(os.path.join(out, "narration.jsonl"))]
    assert len(side) >= len(heard) and all(r["narrator"] == "template-v0" and r["player"] == aid for r in side)
    assert [l.rstrip("\n") for l in open(os.path.join(out, "events.jsonl"))] == batch_of(out)
    assert not any("heard" in e.kind or "narrat" in e.kind for e in read_jsonl(os.path.join(out, "events.jsonl")))


def test_the_prompt_holds_nothing_the_player_could_not_see():
    """Review of #61: a speaker's memory (its notes, say of a theft elsewhere) and needs stay out of the prompt."""
    sim, me, (x, y, *_) = world_with_body()
    sim._note(x, "stole food from Somebody Elsewhere")
    sim.state["agents"][x]["needs"]["hunger"] = 600
    sim.state["agents"][y]["needs"]["hunger"] = 900
    its = narrate.items(sim, [Ev(1, "talk", x, {"with": y, "gossip": []}), Ev(2, "meet", x, {"with": me})], me)
    for it in its:
        p = narrate.prompt(it)
        assert "stole" not in p and "Somebody Elsewhere" not in p and "hungry" not in p
    assert "looks starving" in narrate.prompt(its[0])
    for facts in its[0]["people"].values():
        assert set(facts) == {"name", "role", "looks", "doing"}


def test_the_prompt_names_the_hour_the_season_and_what_each_speaker_is_doing():
    """E2b (Steering 10:09Z): the 8B dry run's lines invented rain, corn and prices and said "Mornin'" at noon,
    because the prompt gave only "a small farming village". It now gives the hour, the part of day, the season,
    what each speaker is visibly doing, and says to speak only of these facts."""
    sim, me, (x, y, *_) = world_with_body()
    it = narrate.items(sim, [Ev(1, "talk", x, {"with": y, "gossip": []})], me)[0]
    p = narrate.prompt(it)
    h = (sim.state["tick"] - 1) % TICKS_PER_DAY * 10 // 60
    assert f"Time: {h:02d}:00, {narrate.part_of_day(h)}" in p
    assert it["season"] in ("winter", "the growing season", "harvest day, the last before winter") and it["season"] in p
    for s in (x, y):
        a = sim.state["agents"][s]
        assert f"{a['name']}, {a['role']}, {narrate.DOING[a['activity']]}" in p
    assert "Speak only of these facts and the people here" in p
    assert [narrate.part_of_day(h) for h in (3, 6, 10, 14, 19, 22)] == \
        ["night", "early morning", "morning", "afternoon", "evening", "night"]


def test_the_season_follows_the_year():
    sim, _, _ = world_with_body()
    y = sim.state["stakes"]["year"]["yields"]
    names = [narrate.season(sim, d) for d in range(len(y))]
    for d, n in enumerate(names):
        nxt = y[(d + 1) % len(y)]
        assert n == ("winter" if y[d] == 0 else "harvest day, the last before winter" if nxt == 0 else "the growing season")
    assert "winter" in names and names.count("harvest day, the last before winter") == 1


def test_the_prompt_says_what_the_place_holds_and_the_greeting_for_the_hour():
    """E2c (Steering 13:07Z): dry run 2's lines named beans, stalls and a cow, and said "Evenin'" at 14:00. The
    prompt now says outright what each place holds and gives the greeting word for the hour."""
    sim, me, (x, y, *_) = world_with_body()
    it = narrate.items(sim, [Ev(1, "talk", x, {"with": y, "gossip": []})], me)[0]
    for place, holds in list(narrate.PLACE_HOLDS.items()) + [("home_3", narrate.HOME_HOLDS)]:
        for h, word in ((7, "Good morning"), (14, "Good afternoon"), (19, "Good evening"), (2, "Good evening")):
            p = narrate.prompt(dict(it, place=place, hour=h))
            assert holds in p and narrate.VILLAGE in p and f'A greeting now is "{word}"' in p
            assert "home_3" not in p
    assert 'just "food"' in narrate.PLACE_HOLDS["market"] and "no stalls" in narrate.PLACE_HOLDS["market"]


def test_the_deny_list_catches_kinds_of_food_stalls_animals_weather_and_wrong_greetings():
    assert narrate.invented("Got any beans at the east stall?", 10) == ["beans", "stall"]
    assert narrate.invented("Saw fox tracks by your cow. Looks like rain.", 10) == ["fox", "cow", "rain"]
    assert narrate.invented("Evenin', Mara.", 14) == ["evening greeting"]
    assert narrate.invented("Mornin'. Good morning to you.", 14) == ["morning greeting"]
    assert narrate.invented("Good afternoon. Bought some food?", 14) == []
    assert narrate.invented("Good evening.", 23) == [] and narrate.invented("Mornin'!", 7) == []
    assert narrate.invented("I need tools from Oren, and the plaza's busy.", 10) == []


def test_a_deny_list_hit_is_asked_once_more_then_falls_back():
    sim, me, (x, y, *_) = world_with_body()
    it = narrate.items(sim, [Ev(1, "talk", x, {"with": y, "gossip": []})], me)[0]
    it["hour"] = 14
    nx, ny = sim.state["agents"][x]["name"], sim.state["agents"][y]["name"]
    bad = json.dumps({"lines": [{"who": nx, "text": "Fresh beans today."}, {"who": ny, "text": "Good afternoon."}]})
    good = json.dumps({"lines": [{"who": nx, "text": "Good afternoon."}, {"who": ny, "text": "Afternoon to you."}]})
    n = narrate.LLMNarrator("http://127.0.0.1:9", "m")
    for replies, fallback in (([bad, good], False), ([bad, bad], True), ([good], False)):
        asked = []
        q = list(replies)
        n.client.post = lambda path, body: (asked.append(body["messages"][0]["content"]),
                                            {"choices": [{"message": {"content": q.pop(0)}}]})[1]
        got = dict(it)
        lines = n.lines(got)
        assert got["fallback"] is fallback and len(asked) == len(replies)
        assert lines == (narrate.TemplateNarrator().lines(it) if fallback else
                         [{"who": x, "text": "Good afternoon."}, {"who": y, "text": "Afternoon to you."}])
        if len(asked) == 2:
            assert "Not beans" in asked[1] and asked[1].startswith(asked[0])
            assert got["denied"][0] == ["beans"]


def test_topics_come_from_the_speakers_own_events_of_the_last_day():
    """E2c: one or two topics from the speakers' own real events: what they bought, asked for, gave or were
    refused, the trade input they need, and sickness in their family. Never the player's or a bystander's."""
    sim, me, (x, y, *_) = world_with_body()
    ag = sim.state["agents"]
    other = next(a["id"] for a in ag if a["id"] not in (me, x, y) and a["family"] != ag[x]["family"])
    sup = next(a["id"] for a in ag if a["role"] == "crafter" and a["family"] != ag[x]["family"])
    t = sim.state["tick"]
    r = narrate.Recent()

    class E(Ev):
        def __init__(self, id, kind, actor, data, tick=t - 1):
            super().__init__(id, kind, actor, data)
            self.tick = tick
    r.feed(sim, [E(1, "receipt", x, {"action": "buy_food", "ok": True, "why": "bought"}, t - 200),  # over a day ago
                 E(2, "receipt", x, {"action": "ask_food", "ok": False, "why": "refused", "to": other}),
                 E(3, "receipt", y, {"action": "give_food", "ok": True, "why": "gave", "to": other}),
                 E(4, "receipt", me, {"action": "buy_food", "ok": True, "why": "bought"}),
                 E(5, "receipt", other, {"action": "buy_food", "ok": True, "why": "bought"}),
                 E(6, "trade_due", y, {"need": "crafter", "to": sup})])
    it = narrate.items(sim, [Ev(9, "talk", x, {"with": y, "gossip": []})], me, r)[0]
    nx, ny, no, ns = (ag[i]["name"] for i in (x, y, other, sup))
    assert it["topics"] == [f"{nx} asked {no} for food and was turned down", f"{ny} needs tools from {ns}, the crafter, today"]
    p = narrate.prompt(it)
    assert "True things from their day they might mention: " + "; ".join(it["topics"]) in p
    assert "bought food" not in p  # the old buy, the player's own and a bystander's stay out
    r.feed(sim, [E(7, "receipt", y, {"action": "trade", "ok": True, "why": "supplied", "to": sup, "coin": 1})])
    assert r.topics(sim, y) == [f"{ny} got tools from {ns}, the crafter", f"{ny} gave food to {no}"]
    kin = [m for m in sim.state["families"][ag[y]["family"]]["members"] if m != y][0]
    sim.state["sick"]["ill"] = {str(kin): t + 99, str(other): t + 99}  # a bystander's sickness stays out
    assert r.topics(sim, y) == [f"{ag[kin]['name']} in {ny}'s family is sick", f"{ny} got tools from {ns}, the crafter"]
    assert narrate.items(sim, [Ev(9, "talk", x, {"with": y, "gossip": []})], me)[0]["topics"] == []
