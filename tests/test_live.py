"""Live runner (B10 PR 1, plan/client-protocol.md): a live world is a batch world whose player actions
arrived late. Live equals batch hash for hash, kills lose no intent the sidecar received (S8), and a
hostile client changes nothing except through recorded player events (S1)."""
import asyncio
import base64
import hashlib
import json
import os
import random
import struct

import pytest

from asiimov import live
from asiimov.events import read_jsonl
from asiimov.minds import StubMind
from asiimov.replay import replay
from asiimov.sim import TICKS_PER_DAY, Sim

AGENTS, SEED, END = 8, 3, TICKS_PER_DAY + 20
# (tick, intent id, family): two fires in one tick, one on a family that just burned, one late on day 2
SCRIPT = [(12, "a", 1), (12, "b", 2), (70, "c", 1), (150, "d", 0)]


def batch_log(script=SCRIPT, end=END):
    player = {}
    for tick, _, fam in script:
        player.setdefault(tick, []).append({"kind": "fire", "family": fam})
    sim = Sim(SEED, AGENTS, StubMind())
    sim.run_until(end, player)
    sim.finish()
    return [e.to_json() for e in sim.events], sim.state_hash()


def intent(iid, fam, seen):
    return {"type": "intent", "id": iid, "verb": "fire", "args": {"family": fam}, "seen_tick": seen}


def lines(path):
    with open(path) as f:
        return [l.rstrip("\n") for l in f]


def drive(runner, script, end, sent=None):
    """Step to `end`, submitting each scripted intent at its tick. Returns every ack seen."""
    sent = sent if sent is not None else set()
    acks = []
    while runner.sim.state["tick"] < end:
        now = runner.sim.state["tick"]
        for tick, iid, fam in script:
            if tick == now and iid not in sent:
                sent.add(iid)
                assert runner.submit(intent(iid, fam, now)) is None
        acks += runner.step()[2]
    return acks


def test_live_run_equals_batch_run_hash_for_hash(tmp_path):
    out = str(tmp_path / "w")
    runner = live.LiveRunner(out, StubMind(), SEED, AGENTS, mind_kind="stub")
    acks = drive(runner, SCRIPT, END)
    runner.finish()
    want, final = batch_log()
    assert lines(os.path.join(out, "events.jsonl")) == want
    events = read_jsonl(os.path.join(out, "events.jsonl"))
    assert replay(events).state_hash() == final == events[-1].data["hash"]
    assert [(a["id"], a["status"], a["tick"]) for a in acks] == [(i, "applied", t) for t, i, _ in SCRIPT]
    for a in acks:  # each ack names its own fire event
        e = events[a["event"]]
        assert (e.kind, e.origin, e.tick) == ("fire", "player", a["tick"])
    # run.json is a scripted world: the batch runner's resume and replay read it the same way
    cfg = json.load(open(os.path.join(out, "run.json")))
    assert cfg["player"] == [[t, {"kind": "fire", "family": f}] for t, _, f in SCRIPT]
    side = [json.loads(l) for l in lines(os.path.join(out, "runner.jsonl"))]
    assert {r["kind"] for r in side} >= {"open", "received", "applied"}
    assert all("wall" in r for r in side) and not any("wall" in l for l in want)


def test_view_positions_are_the_engines_own_at_every_tick_boundary(tmp_path):
    runner = live.LiveRunner(str(tmp_path / "w"), StubMind(), SEED, AGENTS)
    walkers = 0
    while runner.sim.state["tick"] < TICKS_PER_DAY:
        view = runner.view()
        assert view["tick"] == runner.sim.state["tick"]
        for v, a in zip(view["agents"], runner.sim.state["agents"]):
            assert v["pos"] == list(runner.sim._pos(a)) and v["place"] == a["place"]
            walkers += v["route"] is not None
        runner.step()
    assert walkers > 10  # the day has walking in it, so routes were exercised


class Killed(Exception):
    pass


class DiesAt(StubMind):
    def __init__(self, tick):
        self.tick = tick

    def decide(self, req):
        if req.tick >= self.tick:
            raise Killed("mind died mid-tick")
        return super().decide(req)


def _kill(out, rng, kill_tick, phase):
    """Run the script until a kill at `kill_tick` in `phase`, leaving the directory as a crash would."""
    mind = DiesAt(kill_tick) if phase == "mid_tick" else StubMind()
    r = live.LiveRunner(out, mind, SEED, AGENTS)
    sent = set()
    try:
        while r.sim.state["tick"] < END:
            now = r.sim.state["tick"]
            for tick, iid, fam in SCRIPT:
                if tick == now and iid not in sent:
                    if now == kill_tick and phase == "before_schedule":
                        r._schedule = lambda rec: (_ for _ in ()).throw(Killed())
                    sent.add(iid)
                    r.submit(intent(iid, fam, now))
            if now == kill_tick and phase == "before_step":
                raise Killed()
            if now == kill_tick and phase == "before_ack":
                side = r._side
                r._side = lambda kind, sync=False, **d: (_ for _ in ()).throw(Killed()) if kind == "applied" \
                    else side(kind, sync, **d)
            r.step()
        raise AssertionError("the kill never happened")
    except Killed:
        pass
    for name in ("events.jsonl", "runner.jsonl"):  # sometimes a torn last line
        if rng.random() < 0.5:
            with open(os.path.join(out, name), "a") as f:
                f.write('{"torn": ')
    return sent


@pytest.mark.parametrize("trial", range(40))
def test_kill_anywhere_and_reopen_loses_no_received_intent(tmp_path, trial):
    rng = random.Random(trial)
    phase = rng.choice(["before_schedule", "before_step", "mid_tick", "before_ack"])
    ticks = [t for t, _, _ in SCRIPT]  # the two phases around an intent need one in that tick
    kill_tick = rng.choice(ticks if phase in ("before_schedule", "before_ack") else ticks + [rng.randrange(1, END)])
    out = str(tmp_path / "w")
    sent = _kill(out, rng, kill_tick, phase)
    r = live.LiveRunner(out, StubMind())
    for t, iid, fam in SCRIPT:  # a client resends what it never saw acked
        if iid in sent:
            ack = r.submit(intent(iid, fam, r.sim.state["tick"]))
            assert ack is None or (ack["status"], ack["tick"]) == ("applied", t)
    drive(r, SCRIPT, END, sent)
    r.finish()
    assert lines(os.path.join(out, "events.jsonl")) == batch_log()[0], (phase, kill_tick)
    assert all(r.intents[i]["status"] == "applied" for _, i, _ in SCRIPT)


def test_hostile_client_changes_nothing_but_recorded_player_events(tmp_path):
    rng = random.Random(7)
    runner = live.LiveRunner(str(tmp_path / "w"), StubMind(), SEED, AGENTS)
    nfam = len(runner.sim.state["families"])
    bad_args = [None, [], {}, {"family": True}, {"family": 1.0}, {"family": -1}, {"family": nfam},
                {"family": 10 ** 30}, {"family": "1"}, {"family": 1, "extra": 2}, {"fam": 1}]
    valid, seq = [], 0
    while runner.sim.state["tick"] < END:
        now = runner.sim.state["tick"]
        for _ in range(rng.randrange(4)):
            seq += 1
            iid = f"h{seq}"
            m = rng.choice([
                lambda: intent(iid, rng.randrange(nfam), now + rng.randrange(1, 9)),       # from the future
                lambda: intent(iid, rng.randrange(nfam), now - live.MAX_LAG - 1),          # stale
                lambda: {**intent(iid, 0, now), "seen_tick": rng.choice([None, True, "3", 1.5])},
                lambda: {**intent(iid, 0, now), "verb": rng.choice(["steal", "", None, "FIRE"])},
                lambda: {**intent(iid, 0, now), "args": rng.choice(bad_args)},
                lambda: {**intent(iid, 0, now), "id": rng.choice([None, 3, "", "x" * 65])},
                lambda: intent(rng.choice(valid)[1], rng.randrange(nfam), now) if valid else intent(iid, 0, -1),
                lambda: intent(iid, rng.randrange(nfam), max(0, now - rng.randrange(live.MAX_LAG))),  # valid
            ])()
            fresh = isinstance(m.get("id"), str) and m["id"] not in runner.intents
            ack = runner.submit(m)
            if ack is None:
                assert fresh
                valid.append((now, m["id"], m["args"]["family"]))
            else:
                assert ack["status"] == "rejected" or not fresh
        runner.step()
    runner.finish()
    assert len(valid) > 10 and seq > 100
    assert lines(os.path.join(runner.out, "events.jsonl")) == batch_log(valid)[0]


# ---------------------------------------------------------------------- over the wire
class WS:
    """A bare RFC 6455 client, masked as clients must be."""

    def __init__(self, reader, writer):
        self.r, self.w = reader, writer

    @classmethod
    async def connect(cls, port):
        r, w = await asyncio.open_connection("127.0.0.1", port)
        key = base64.b64encode(os.urandom(16)).decode()
        w.write(f"GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n".encode())
        head = await r.readuntil(b"\r\n\r\n")
        accept = base64.b64encode(hashlib.sha1((key + live.GUID).encode()).digest()).decode()
        assert head.startswith(b"HTTP/1.1 101") and accept.encode() in head
        return cls(r, w)

    def raw(self, op, payload, masked=True, fin=True):
        mask = os.urandom(4) if masked else b""
        n = len(payload)
        head = bytes([(0x80 if fin else 0) | op]) + (
            bytes([(0x80 if masked else 0) | n]) if n < 126 else bytes([(0x80 if masked else 0) | 126]) + struct.pack(">H", n))
        self.w.write(head + mask + (live._unmask(payload, mask) if masked else payload))

    def send(self, obj):
        self.raw(1, json.dumps(obj).encode())

    async def recv(self):
        b1, b2 = await asyncio.wait_for(self.r.readexactly(2), 10)
        n = b2 & 0x7F
        if n == 126:
            (n,) = struct.unpack(">H", await self.r.readexactly(2))
        elif n == 127:
            (n,) = struct.unpack(">Q", await self.r.readexactly(8))
        data = await self.r.readexactly(n)
        return None if b1 & 0x0F == 8 else json.loads(data)

    async def until(self, type_):
        while (m := await self.recv())["type"] != type_:
            pass
        return m


def test_websocket_session_hello_backlog_ticks_acks_and_reattach(tmp_path):
    runner = live.LiveRunner(str(tmp_path / "w"), StubMind(), SEED, AGENTS)
    for _ in range(30):
        runner.step()

    async def main():
        ready = asyncio.get_running_loop().create_future()
        task = asyncio.ensure_future(live.serve(runner, port=0, paused=True,
                                                started=lambda port, hub: ready.set_result(port)))
        port = await ready
        c = await WS.connect(port)
        c.send({"type": "hello", "protocol": 1, "scope": "god"})
        w = await c.recv()
        assert (w["type"], w["protocol"], w["view_tick"], w["paused"]) == ("welcome", 1, 30, True)
        assert w["engine"] == runner.sim.events[0].data["engine"] and w["world"]["map"] == runner.sim.map
        got = []
        while len(got) < w["n_events"]:
            got += (await c.recv())["events"]
        assert got == live.event_dicts(runner.sim.events)

        c.send(intent("w1", 2, 30))
        c.send(intent("w1", 2, 30))           # a resend before the ack: still one fire
        c.send(intent("w2", 99, 30))
        assert await c.recv() == live.msg("ack", id="w2", status="rejected", reason="bad_args")
        c.send({"type": "control", "op": "step"})
        await c.until("status")
        t = await c.recv()
        assert t["type"] == "tick" and t["tick"] == 30 and t["view"] == runner.view()
        fires = [e for e in t["events"] if e["kind"] == "fire"]
        assert len(fires) == 1 and fires[0]["data"] == {"family": 2}
        ack = await c.recv()
        assert (ack["id"], ack["status"], ack["event"]) == ("w1", "applied", fires[0]["id"])

        c.send({"type": "control", "op": "speed", "tps": 200})
        c.send({"type": "control", "op": "run"})
        while (await c.recv()).get("tick", 0) < 40:
            pass
        c.send({"type": "control", "op": "pause"})
        s = await c.until("status")
        while s["paused"] is not True:
            s = await c.until("status")

        since = len(runner.sim.events) // 2  # paused, so the log is still
        d = await WS.connect(port)            # reattach mid-world: the gap, then live ticks, nothing twice
        d.send({"type": "hello", "protocol": 1, "since": since})
        w2 = await d.recv()
        back = []
        while len(back) < w2["n_events"] - since:
            back += (await d.recv())["events"]
        d.send({"type": "control", "op": "step"})
        await d.until("status")
        t2 = await d.recv()
        assert [e["id"] for e in back + t2["events"]] == list(range(since, len(runner.sim.events)))

        e = await WS.connect(port)
        e.send({"type": "hello", "protocol": 1, "scope": "inhabitant", "agent": 0})
        assert (await e.recv())["reason"] == "unsupported_scope" and await e.recv() is None
        f = await WS.connect(port)
        f.raw(1, b'{"type": "hello", "protocol": 1}', masked=False)  # clients must mask
        assert (await f.recv())["reason"].startswith("bad_frame") and await f.recv() is None
        g = await WS.connect(port)
        g.send({"type": "hello", "protocol": 2})
        assert (await g.recv())["reason"] == "unknown_protocol"
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(main())
    assert [r.get("id") for r in map(json.loads, lines(os.path.join(runner.out, "runner.jsonl")))
            if r["kind"] == "received"] == ["w1"]
