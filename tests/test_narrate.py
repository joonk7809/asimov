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
        assert "soul" not in p.lower() and "trait" not in p.lower()


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
