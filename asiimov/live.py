"""Live runner: step a world in real time and serve it to clients over WebSocket (plan/client-protocol.md).

    python -m asiimov.live --out runs/live --agents 25 --seed 1 --tps 2      # new world, or reopen runs/live
    python -m asiimov.live --out runs/live --paused                          # reopen paused

The engine stays the only source of truth. Clients get events and views, and send intents. An intent is
checked here, scheduled in run.json's player list (the same place a scripted --fire goes), and applied by
the engine at the start of the next tick, so a live world is a batch world whose player actions arrived
late. Wall-clock time, clients, controls and acks go to the sidecar runner.jsonl, never to events.jsonl
or the state hash.

A live run against a served model (--mind llm) needs its own charter and its own pod, never one serving a
canonical run (plan/2d-env-eval.md). Test 16b's session is one (plan/next-push-2026-11-09.md, track 3):

    python -m asiimov.live --out runs/s16b --agents 25 --mind llm --llm-url ... --llm-model ... --llm-key ... \
        --host 0.0.0.0 --token SECRET --tps 0.05

--token makes every hello carry that token: on a public port, nobody else attaches or steers the world.
Each tick's wall time and mind-call latencies go to the sidecar (`step`), next to each intent's received
and applied times, so a session's reply latency is measured from runner.jsonl alone.
"""
import argparse
import asyncio
import base64
import hashlib
import json
import os
import struct
import sys
import time
from collections import defaultdict

from . import inhabitant, persist
from .minds import NoisyMind, StubMind
from .sim import DELIBERATE_ACTIONS, ENGINE_VERSION, TICKS_PER_DAY
from .worldmap import position

PROTOCOL = 2               # 2 adds the inhabitant scope (plan/player-body-spec.md)
SCOPES = ("god", "inhabitant")
MAX_LAG = TICKS_PER_DAY    # an intent seen more than a day ago is stale
ORPHAN_TICKS = 36          # a body still held on reopen is released this long after, unless an inhabitant claims it
MAX_FRAME = 1 << 20        # client frames
BACKLOG_CHUNK = 500        # events per backlog message
SEND_QUEUE = 4096          # messages a slow client may fall behind before it is dropped


def _atomic_write(path: str, text: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


class LiveRunner:
    """One world in a run directory, stepped one tick at a time, with intents scheduled between ticks.
    Not thread-safe: submit(), step() and view() must not overlap."""

    def __init__(self, out: str, mind, seed: int = 1, agents: int = 25, order: str = "id",
                 snapshot_every: int = TICKS_PER_DAY, mind_kind: str | None = None, workers: int = 1):
        self.out = out
        self.mind = mind
        if os.path.exists(os.path.join(out, "events.jsonl")):
            self.run = persist.resume(out, mind, workers, snapshot_every=snapshot_every)
            self.reopened = True
        else:
            self.run = persist.start(out, seed, agents, mind, order, {}, workers, snapshot_every=snapshot_every,
                                     mind_kind=mind_kind)
            self.reopened = False
        self.sim = self.run.sim
        self.intents: dict[str, dict] = {}  # intent id -> {status, tick, slot, act, event | reason}
        self.side_path = os.path.join(out, "runner.jsonl")
        lines = []
        if os.path.exists(self.side_path):
            with open(self.side_path, "rb") as f:
                lines = f.read().split(b"\n")[:-1]  # a torn last line was never acted on
            with open(self.side_path, "r+b") as f:
                f.truncate(sum(len(l) + 1 for l in lines))
        self.side = open(self.side_path, "a")
        self.inhabitant: dict | None = None  # {"client", "agent"} while an inhabitant is connected
        self._reconcile(lines)
        # An inhabitant's body still held on reopen (a crash, or a stop while it was connected) is released
        # ORPHAN_TICKS later unless an inhabitant says hello for it. A god-scope possession is left alone,
        # as it is on a god client's disconnect.
        (held, since), last = self._held_from(), self._last_inhabit
        orphan = held is not None and last is not None and last["kind"] == "inhabit" and last["agent"] == held
        self.orphan_at = since + ORPHAN_TICKS if orphan else None
        self._side("open", reopened=self.reopened, view_tick=self.sim.state["tick"], n_events=len(self.sim.events))

    # ------------------------------------------------------------------ sidecar
    def _side(self, kind: str, sync: bool = False, **data):
        self.side.write(json.dumps({"wall": round(time.time(), 3), "kind": kind, **data}, sort_keys=True) + "\n")
        self.side.flush()
        if sync:
            os.fsync(self.side.fileno())

    def _reconcile(self, lines: list[bytes]):
        """Rebuild intent outcomes from the sidecar, and finish what a kill interrupted: an intent the
        sidecar received but run.json doesn't schedule is scheduled now; one whose tick has run is applied."""
        self._last_inhabit = None  # the newest inhabit or leave record
        for line in lines:
            r = json.loads(line)
            if r["kind"] == "received":
                self.intents[r["id"]] = {"status": "pending", "tick": r["tick"], "slot": r["slot"], "act": r["act"]}
            elif r["kind"] == "applied":
                self.intents[r["id"]].update(status="applied", event=r["event"])
            elif r["kind"] == "rejected":
                self.intents[r["id"]] = {"status": "rejected", "reason": r["reason"]}
            elif r["kind"] in ("inhabit", "leave"):
                self._last_inhabit = r
        now = self.sim.state["tick"]
        for iid, rec in self.intents.items():
            if rec["status"] == "rejected":
                continue
            acts = self.run.player.get(rec["tick"], [])
            if rec["slot"] == len(acts) and rec["tick"] >= now:
                self._schedule(rec)  # killed between the sidecar and run.json
            elif rec["slot"] >= len(acts) or acts[rec["slot"]] != rec["act"]:
                raise ValueError(f"{self.out}: run.json doesn't schedule intent {iid!r} where the sidecar says")
            if rec["tick"] < now:
                ev = self._player_events(rec["tick"])[rec["slot"]]
                if rec["status"] == "pending":
                    rec.update(status="applied", event=ev.id)
                    self._side("applied", id=iid, tick=rec["tick"], event=ev.id)
                elif ev.id != rec["event"]:
                    raise ValueError(f"{self.out}: intent {iid!r} was acked as event {rec['event']}, the log has {ev.id}")

    def _player_events(self, tick: int) -> list:
        return [e for e in self.sim.events if e.tick == tick and e.origin == "player"]

    def _schedule(self, rec: dict):
        self.run.player.setdefault(rec["tick"], []).append(rec["act"])
        cfg = persist.config(self.out)
        cfg["player"] = [[t, act] for t in sorted(self.run.player) for act in self.run.player[t]]
        _atomic_write(os.path.join(self.out, "run.json"), json.dumps(cfg, indent=1))

    # ------------------------------------------------------------------ intents
    def verbs(self) -> dict:
        n = len(self.sim.state["agents"])
        return {"fire": {"family": [0, len(self.sim.state["families"]) - 1]},
                "possess": {"agent": [0, n - 1]}, "release": {},
                "act": {"action": list(DELIBERATE_ACTIONS), "target": [0, n - 1]}}

    def _held_agent(self) -> int | None:
        return self._held_from()[0]

    def _held_from(self) -> tuple[int | None, int]:
        """The body possessed once every action already scheduled applies, and the tick that state takes
        effect. Scheduled actions can lie past the view: a crash reopen re-runs ticks whose actions run.json
        already holds. Possession changes only through player actions, which all pass through here, so this
        is exact."""
        now = self.sim.state["tick"]
        held, since = self.sim.state.get("player", {}).get("agent"), now
        for t in sorted(t for t in self.run.player if t >= now):
            for act in self.run.player[t]:
                if act["kind"] in ("possess", "release"):
                    held, since = (act["agent"] if act["kind"] == "possess" else None), t + 1
        return held, since

    def _held(self) -> bool:
        return self._held_agent() is not None

    def _internal(self, act: dict) -> None:
        """A player action the runner itself sends (an inhabitant's possess or release, an orphan's release),
        through the same durable path as a client's intent. Its id starts with "~", which clients can't use."""
        tick = self.sim.state["tick"]
        iid = f"~{act['kind']}-{tick}-{len(self.run.player.get(tick, []))}"
        rec = {"status": "pending", "tick": tick, "slot": len(self.run.player.get(tick, [])), "act": act}
        self._side("received", sync=True, id=iid, client="runner", tick=tick, slot=rec["slot"], act=act,
                   seen_tick=tick)
        self.intents[iid] = rec
        self._schedule(rec)

    def inhabit(self, agent, client: str) -> str | None:
        """An inhabitant's hello for `agent`: possess it, or claim it if a reopened world still holds it.
        Returns why not, or None."""
        if not _is_int(agent) or not 0 <= agent < len(self.sim.state["agents"]):
            return "bad_agent"
        held = self._held_agent()
        if self.inhabitant is not None or (held is not None and (held != agent or self.orphan_at is None)):
            return "body_taken"
        self._side("inhabit", sync=True, client=client, agent=agent, claimed=held is not None)  # before the possess
        if held is None:
            self._internal({"kind": "possess", "agent": agent})
        self.orphan_at = None
        self.inhabitant = {"client": client, "agent": agent}
        return None

    def leave(self, client: str):
        """An inhabitant disconnected: release its body, and the mind's own deliberations resume."""
        if self.inhabitant is None or self.inhabitant["client"] != client:
            return
        agent, self.inhabitant = self.inhabitant["agent"], None
        if self._held_agent() == agent:
            self._internal({"kind": "release"})
        self._side("leave", client=client, agent=agent)

    def _check(self, msg: dict, scope: str) -> tuple[dict | None, str | None]:
        """The engine action for an intent, or why it is refused. This checks shape and the one piece of
        state the runner owns (who is possessed); whether the action works is the engine's call when it
        applies (L2), with a receipt."""
        now = self.sim.state["tick"]
        seen = msg.get("seen_tick")
        verb = msg.get("verb")
        if not isinstance(verb, str) or verb not in self.verbs():
            return None, "unknown_verb"
        if scope == "inhabitant" and verb != "act":
            return None, "scope"  # divine verbs and the body's own possession are not an inhabitant's
        args = msg.get("args")
        n = len(self.sim.state["agents"])
        if not isinstance(args, dict):
            return None, "bad_args"
        if verb == "fire":
            ok = set(args) == {"family"} and _is_int(args["family"]) and 0 <= args["family"] < len(self.sim.state["families"])
            act = {"kind": "fire", "family": args.get("family")}
        elif verb == "possess":
            ok = set(args) == {"agent"} and _is_int(args["agent"]) and 0 <= args["agent"] < n
            act = {"kind": "possess", "agent": args.get("agent")}
        elif verb == "release":
            ok, act = args == {}, {"kind": "release"}
        else:
            t = args.get("target")
            ok = set(args) <= {"action", "target"} and isinstance(args.get("action"), str) \
                and args["action"] in DELIBERATE_ACTIONS and (t is None or (_is_int(t) and 0 <= t < n))
            act = {"kind": "act", "action": args.get("action")} if t is None else \
                {"kind": "act", "action": args.get("action"), "target": t}
        if not ok:
            return None, "bad_args"
        if not _is_int(seen) or not 0 <= seen <= now:
            return None, "bad_seen_tick"
        if now - seen > MAX_LAG:
            return None, "stale"
        if scope == "god" and verb != "fire" and self.inhabitant is not None:
            return None, "inhabited"  # the body is an inhabitant's: god scope keeps fire only
        if scope == "inhabitant" and (self.inhabitant is None or self._held_agent() != self.inhabitant["agent"]):
            return None, "no_body"
        held = self._held()
        if verb == "possess" and held:
            return None, "body_taken"  # release first: one action, one player event (see step)
        if verb in ("act", "release") and not held:
            return None, "no_body"
        return act, None

    def submit(self, msg: dict, client: str = "local", scope: str = "god") -> dict | None:
        """Take an intent. Returns its ack now if it is refused or already decided, or None if it will be
        applied by the next step() (which returns its ack once its event is durable)."""
        iid = msg.get("id")
        if not isinstance(iid, str) or not 1 <= len(iid) <= 64 or iid.startswith("~"):
            return self._ack(iid if isinstance(iid, str) else None, {"status": "rejected", "reason": "bad_id"})
        if iid in self.intents:
            rec = self.intents[iid]
            return None if rec["status"] == "pending" else self._ack(iid, rec)
        act, reason = self._check(msg, scope)
        if reason:
            rec = self.intents[iid] = {"status": "rejected", "reason": reason}
            self._side("rejected", id=iid, client=client, reason=reason, msg=_clip(msg))
            return self._ack(iid, rec)
        tick = self.sim.state["tick"]
        rec = {"status": "pending", "tick": tick, "slot": len(self.run.player.get(tick, [])), "act": act}
        self._side("received", sync=True, id=iid, client=client, tick=tick, slot=rec["slot"], act=act,
                   seen_tick=msg["seen_tick"])
        self.intents[iid] = rec
        self._schedule(rec)
        return None

    @staticmethod
    def _ack(iid, rec: dict) -> dict:
        out = msg("ack", id=iid, status=rec["status"])
        if rec["status"] == "applied":
            out.update(tick=rec["tick"], event=rec["event"])
        else:
            out["reason"] = rec["reason"]
        return out

    # ------------------------------------------------------------------ stepping
    def step(self) -> tuple[int, list, list[dict]]:
        """Run one tick. Returns (the tick that ran, its events, acks for the intents it applied)."""
        sim, tick = self.sim, self.sim.state["tick"]
        if self.orphan_at is not None and tick >= self.orphan_at:
            self.orphan_at = None
            if self.inhabitant is None and self._held():
                self._internal({"kind": "release"})  # reopened with a body held and nobody came back for it
        n0, c0, t0 = len(sim.events), len(getattr(self.mind, "calls", ())), time.time()
        self.run.advance(tick + 1)
        events = sim.events[n0:]
        lat = [c["latency_s"] for c in getattr(self.mind, "calls", [])[c0:]]
        self._side("step", tick=tick, wall_s=round(time.time() - t0, 3), calls=len(lat),
                   lat_max_s=max(lat, default=0), lat_sum_s=round(sum(lat), 3))
        due = sorted((rec["slot"], iid) for iid, rec in self.intents.items()
                     if rec["status"] == "pending" and rec["tick"] == tick)
        acks = []
        if due:
            os.fsync(self.run.log.fileno())  # acked only once durable
            player = [e for e in events if e.origin == "player"]
            if len(player) != len(self.run.player.get(tick, [])):
                raise RuntimeError(f"tick {tick}: {len(player)} player events for "
                                   f"{len(self.run.player[tick])} scheduled actions")
            for slot, iid in due:
                rec = self.intents[iid]
                rec.update(status="applied", event=player[slot].id)
                self._side("applied", id=iid, tick=tick, event=player[slot].id)
                acks.append(self._ack(iid, rec))
        return tick, events, acks

    def finish(self):
        """End the recording here (an end event). A live world normally stays open: use close()."""
        self.run.finish()
        self.side.close()

    def close(self):
        """Stop serving; the world stays open. A snapshot at the current tick lets reopen continue right here
        instead of taking resume's crash path, which re-runs the last recorded tick."""
        if not os.path.exists(self.run._snap_path(self.sim.state["tick"])):
            self.run._snapshot()
        os.fsync(self.run.log.fileno())
        self.run.log.close()
        self._side("close", view_tick=self.sim.state["tick"])
        self.side.close()

    # ------------------------------------------------------------------ what clients see
    def view(self) -> dict:
        """Every villager at the current view tick, with the engine's own position."""
        sim, tick = self.sim, self.sim.state["tick"]
        places = sim.map["places"]
        agents = []
        for a in sim.state["agents"]:
            r = a["route"]
            pos = places[a["place"]]["anchor"] if r is None else position(r, places[r["to"]]["anchor"], tick)
            agents.append({"id": a["id"], "place": a["place"], "activity": a["activity"], "asleep": a["asleep"],
                           "pos": list(pos), "route": None if r is None else
                           {"from": r["from"], "to": r["to"], "depart": r["depart"], "arrive": r["arrive"]}})
        return {"tick": tick, "agents": agents}

    def inhabitant_tick(self, tick: int, events: list[dict], aid: int) -> dict:
        """A tick as an inhabitant gets it: the events it witnessed and its own whitelisted view."""
        here = set(inhabitant.present_ids(self.sim, aid))
        return msg("tick", tick=tick, events=[inhabitant.scrub(e, aid) for e in events if inhabitant.witnessed(e, aid, here, tick)],
                   view=inhabitant.view(self.sim, aid))

    def welcome(self, scope: str, paused: bool, tps: float) -> dict:
        if scope == "inhabitant":
            aid = self.inhabitant["agent"]
            return msg("welcome", scope=scope, view_tick=self.sim.state["tick"], paused=paused, tps=tps,
                       verbs={"act": self.verbs()["act"]}, world=inhabitant.welcome_world(self.sim, aid),
                       view=inhabitant.view(self.sim, aid))
        g = self.sim.events[0].data
        return msg("welcome", scope=scope, view_tick=self.sim.state["tick"], n_events=len(self.sim.events),
                   paused=paused, tps=tps, verbs=self.verbs(),
                   world={"seed": g["seed"], "order": g["order"], "map": g["map"], "population": g["population"],
                          "ticks_per_day": TICKS_PER_DAY})


def msg(type_: str, **fields) -> dict:
    return {"type": type_, "protocol": PROTOCOL, "engine": ENGINE_VERSION, **fields}


def event_dicts(events) -> list[dict]:
    return [json.loads(e.to_json()) for e in events]


def _clip(m: dict) -> str:
    return json.dumps(m, sort_keys=True)[:300]


# ---------------------------------------------------------------------- WebSocket (RFC 6455, server side)
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WSError(Exception):
    pass


def frame(op: int, payload: bytes) -> bytes:
    n = len(payload)
    if n < 126:
        head = struct.pack(">BB", 0x80 | op, n)
    elif n < 1 << 16:
        head = struct.pack(">BBH", 0x80 | op, 126, n)
    else:
        head = struct.pack(">BBQ", 0x80 | op, 127, n)
    return head + payload


def _unmask(data: bytes, mask: bytes) -> bytes:
    n = len(data)
    m = (mask * (n // 4 + 1))[:n]
    return (int.from_bytes(data, "big") ^ int.from_bytes(m, "big")).to_bytes(n, "big")


async def handshake(reader, writer) -> bool:
    try:
        req = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
    except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
        return False
    lines = req.decode("latin-1").split("\r\n")
    headers = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:] if l)}
    key = headers.get("sec-websocket-key")
    if not lines[0].startswith("GET ") or "websocket" not in headers.get("upgrade", "").lower() or not key:
        writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
        return False
    accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    writer.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
    return True


async def read_message(reader, send) -> str | None:
    """The next text message, or None when the client closes. Answers pings through `send`."""
    parts, opcode = [], None
    while True:
        b1, b2 = await reader.readexactly(2)
        fin, op, n = b1 & 0x80, b1 & 0x0F, b2 & 0x7F
        if b1 & 0x70 or not b2 & 0x80:
            raise WSError("reserved bits set or unmasked client frame")
        if n == 126:
            (n,) = struct.unpack(">H", await reader.readexactly(2))
        elif n == 127:
            (n,) = struct.unpack(">Q", await reader.readexactly(8))
        if op >= 8 and (not fin or n > 125):
            raise WSError("bad control frame")
        if n + sum(map(len, parts)) > MAX_FRAME:
            raise WSError("message too large")
        mask = await reader.readexactly(4)
        data = _unmask(await reader.readexactly(n), mask)
        if op == 8:
            return None  # Conn.close() answers it
        if op == 9:
            send(frame(10, data))
            continue
        if op == 10:
            continue
        if op in (1, 2):
            if opcode is not None:
                raise WSError("new message inside a fragmented one")
            opcode = op
        elif op != 0 or opcode is None:
            raise WSError(f"unexpected opcode {op}")
        parts.append(data)
        if fin:
            if opcode != 1:
                raise WSError("binary messages are not part of the protocol")
            try:
                return b"".join(parts).decode("utf-8")
            except UnicodeDecodeError as e:
                raise WSError("text frame is not UTF-8") from e


class Conn:
    """One client: its scope once it has said hello, and a send queue drained by its own task."""
    count = 0

    def __init__(self, writer):
        Conn.count += 1
        self.name = f"c{Conn.count}"
        self.writer, self.queue = writer, asyncio.Queue()
        self.scope, self.ready, self.closed, self.agent = None, False, False, None
        self.task = asyncio.ensure_future(self._pump())

    def send_raw(self, data: bytes):
        if self.closed:
            return
        if self.queue.qsize() >= SEND_QUEUE:
            self.close()
            return
        self.queue.put_nowait(data)

    def send(self, m: dict | str):
        self.send_raw(frame(1, (m if isinstance(m, str) else json.dumps(m)).encode()))

    def close(self, error: str | None = None):
        if error and not self.closed:
            self.queue.put_nowait(frame(1, json.dumps(msg("error", reason=error)).encode()))
        if not self.closed:  # the close frame repeats the reason (code 4000): some clients drop data at close
            self.queue.put_nowait(frame(8, struct.pack(">H", 4000) + error.encode()[:120] if error else struct.pack(">H", 1000)))
        self.closed = True
        self.queue.put_nowait(None)

    async def _pump(self):
        try:
            while (data := await self.queue.get()) is not None:
                self.writer.write(data)
                await self.writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            self.closed = True
            self.writer.close()


class Hub:
    """Paces the runner and fans out to clients. Everything that touches the runner happens between ticks,
    in the tick loop; a tick itself runs in a worker thread so the server keeps reading while it does."""

    def __init__(self, runner: LiveRunner, tps: float = 1.0, paused: bool = False, token: str | None = None):
        self.runner, self.tps, self.paused, self.token = runner, tps, paused, token
        self.conns: set[Conn] = set()
        self.inbox: list[tuple[Conn, dict]] = []
        self.wake = asyncio.Event()
        self.waiters: dict[str, Conn] = {}
        self.step_once, self.stopping = False, False

    async def handle(self, reader, writer):
        if not await handshake(reader, writer):
            writer.close()
            return
        conn = Conn(writer)
        self.conns.add(conn)
        try:
            while (text := await read_message(reader, conn.send_raw)) is not None:
                try:
                    m = json.loads(text)
                    if not isinstance(m, dict) or not isinstance(m.get("type"), str):
                        raise ValueError
                except (ValueError, RecursionError):
                    m = None
                if conn.scope is None:
                    err = _hello_error(m) or (self.token is not None and m.get("token") != self.token and "bad_token")
                    if err:
                        conn.close(err)
                        break
                    conn.scope = m.get("scope", "god")
                elif m is None or m.get("type") not in ("intent", "control"):
                    conn.send(msg("ack", id=None, status="rejected", reason="bad_message"))
                    continue
                self.inbox.append((conn, m))
                self.wake.set()
        except (WSError, asyncio.IncompleteReadError, ConnectionError, OSError) as e:
            conn.close(f"bad_frame: {e}" if isinstance(e, WSError) else None)
        finally:
            conn.close()
            self.conns.discard(conn)
            if conn.scope == "inhabitant":  # its body is released between ticks, like any intent
                self.inbox.append((conn, {"type": "leave"}))
                self.wake.set()

    def _drain(self):
        items, self.inbox = self.inbox, []
        for conn, m in items:
            t = m["type"]
            if t == "hello":
                if conn.scope == "inhabitant":
                    why = self.runner.inhabit(m.get("agent"), conn.name)
                    if why:
                        conn.scope = None  # nothing to release
                        conn.close(why)
                        continue
                    conn.agent = m["agent"]
                self._welcome(conn, m.get("since", 0))
            elif t == "leave":
                self.runner.leave(conn.name)
            elif t == "intent":
                ack = self.runner.submit(m, client=conn.name, scope=conn.scope)
                if ack is None:
                    self.waiters[m["id"]] = conn
                else:
                    conn.send(ack)
            elif t == "control":
                self._control(conn, m)

    def _welcome(self, conn: Conn, since: int):
        events = self.runner.sim.events
        if since > len(events):
            conn.close("bad_since")
            return
        conn.send(self.runner.welcome(conn.scope, self.paused, self.tps))
        if conn.scope == "inhabitant":  # its own, the ones naming it and public ones; no co-presence history
            mine = [inhabitant.scrub(e, conn.agent) for e in event_dicts(events[since:])
                    if inhabitant.witnessed(e, conn.agent, set(), -1)]
            for i in range(0, len(mine), BACKLOG_CHUNK):
                conn.send(msg("events", events=mine[i:i + BACKLOG_CHUNK]))
        else:
            for i in range(since, len(events), BACKLOG_CHUNK):
                conn.send(msg("events", events=event_dicts(events[i:i + BACKLOG_CHUNK])))
        conn.ready = True
        self.runner._side("client", client=conn.name, scope=conn.scope, since=since)

    def _control(self, conn: Conn, m: dict):
        op, tps = m.get("op"), m.get("tps")
        if conn.scope != "god" or op not in ("pause", "run", "step", "speed") or \
                (op == "speed" and not (isinstance(tps, (int, float)) and not isinstance(tps, bool) and 0.1 <= tps <= 1000)):
            conn.send(msg("error", reason="bad_control"))
            return
        if op == "pause":
            self.paused = True
        elif op == "run":
            self.paused = False
        elif op == "step":
            self.step_once = True
        else:
            self.tps = float(tps)
        self.runner._side("control", client=conn.name, op=op, tps=self.tps)
        status = json.dumps(msg("status", paused=self.paused, tps=self.tps, view_tick=self.runner.sim.state["tick"]))
        for c in self.conns:
            if c.ready:
                c.send(status)

    async def tick_loop(self):
        loop = asyncio.get_running_loop()
        due = loop.time()
        while not self.stopping:
            self._drain()
            if self.paused and not self.step_once:
                self.wake.clear()
                await self.wake.wait()
                due = loop.time()
                continue
            self.step_once = False
            tick, events, acks = await asyncio.to_thread(self.runner.step)
            dicts = event_dicts(events)
            text = json.dumps(msg("tick", tick=tick, events=dicts, view=self.runner.view()))
            for c in self.conns:
                if c.ready and c.scope == "inhabitant":
                    c.send(self.runner.inhabitant_tick(tick, dicts, c.agent))
                elif c.ready:
                    c.send(text)
            for ack in acks:
                c = self.waiters.pop(ack["id"], None)
                if c:
                    c.send(ack)
            due = max(due + 1 / self.tps, loop.time())
            self.wake.clear()
            try:  # a control or hello cuts the wait short
                await asyncio.wait_for(self.wake.wait(), due - loop.time())
            except asyncio.TimeoutError:
                pass


def _hello_error(m) -> str | None:
    if m is None or m.get("type") != "hello":
        return "expected_hello"
    if not _is_int(m.get("protocol")) or m["protocol"] != PROTOCOL:
        return "unknown_protocol"
    scope = m.get("scope", "god")
    if scope not in SCOPES:
        return "bad_scope"
    if scope == "inhabitant" and not _is_int(m.get("agent")):
        return "bad_agent"  # its range is the runner's to check
    since = m.get("since", 0)
    if not _is_int(since) or since < 0:
        return "bad_since"
    return None


async def serve(runner: LiveRunner, host: str = "127.0.0.1", port: int = 8765, tps: float = 1.0,
                paused: bool = False, started=None, token: str | None = None):
    """Run the hub until cancelled. `started(port, hub)` is called once the socket listens."""
    hub = Hub(runner, tps, paused, token)
    server = await asyncio.start_server(hub.handle, host, port, limit=1 << 16)
    if started:
        started(server.sockets[0].getsockname()[1], hub)
    try:
        await hub.tick_loop()
    finally:
        hub.stopping = True
        server.close()  # not wait_closed(): it would wait on every open client's read
        for c in list(hub.conns):
            c.close()
            c.writer.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="runs/live", help="run directory; reopened if it holds a world")
    ap.add_argument("--agents", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mind", choices=["stub", "noisy", "llm"], default=None, help="default stub; on reopen, the run's")
    ap.add_argument("--pod", help="pod id from scripts/runpod_pod.py (url, model and key)")
    ap.add_argument("--llm-url")
    ap.add_argument("--llm-model")
    ap.add_argument("--llm-key")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=None, help="concurrent mind calls (default 64 for llm, else 1)")
    ap.add_argument("--token", default=None, help="every hello must carry this token (use it on a public port)")
    ap.add_argument("--tps", type=float, default=1.0, help="ticks per wall-clock second")
    ap.add_argument("--paused", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    kind = args.mind
    if kind is None:
        kind = (persist.config(args.out).get("mind") if os.path.exists(os.path.join(args.out, "run.json")) else None) or "stub"
    if kind not in ("stub", "noisy", "llm"):
        raise SystemExit(f"{args.out} was run with mind {kind!r}")
    if kind == "llm":
        from .run import llm_from_args
        mind = llm_from_args(args)[0]
    else:
        mind = StubMind() if kind == "stub" else NoisyMind()
    runner = LiveRunner(args.out, mind, args.seed, args.agents, mind_kind=kind,
                        workers=args.workers or (64 if kind == "llm" else 1))
    print(f"{'reopened' if runner.reopened else 'new world'} {args.out} at tick {runner.sim.state['tick']}, "
          f"engine {ENGINE_VERSION}, ws://{args.host}:{args.port}/", file=sys.stderr, flush=True)
    try:
        asyncio.run(serve(runner, args.host, args.port, args.tps, args.paused, token=args.token))
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()


if __name__ == "__main__":
    main()
