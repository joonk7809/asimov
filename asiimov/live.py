"""Live runner: step a world in real time and serve it to clients over WebSocket (plan/client-protocol.md).

    python -m asiimov.live --out runs/live --agents 25 --seed 1 --tps 2      # new world, or reopen runs/live
    python -m asiimov.live --out runs/live --paused                          # reopen paused

The engine stays the only source of truth. Clients get events and views, and send intents. An intent is
checked here, scheduled in run.json's player list (the same place a scripted --fire goes), and applied by
the engine at the start of the next tick, so a live world is a batch world whose player actions arrived
late. Wall-clock time, clients, controls and acks go to the sidecar runner.jsonl, never to events.jsonl
or the state hash.

Stub and noisy minds only: a live run against a served model needs its own charter, and never a pod that
is serving a canonical run (plan/2d-env-eval.md).
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

from . import persist
from .minds import NoisyMind, StubMind
from .sim import ENGINE_VERSION, TICKS_PER_DAY
from .worldmap import position

PROTOCOL = 1
SCOPES = ("god",)          # "inhabitant" is reserved for Track B
MAX_LAG = TICKS_PER_DAY    # an intent seen more than a day ago is stale
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
                 snapshot_every: int = TICKS_PER_DAY, mind_kind: str | None = None):
        self.out = out
        if os.path.exists(os.path.join(out, "events.jsonl")):
            self.run = persist.resume(out, mind, snapshot_every=snapshot_every)
            self.reopened = True
        else:
            self.run = persist.start(out, seed, agents, mind, order, {}, snapshot_every=snapshot_every,
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
        self._reconcile(lines)
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
        for line in lines:
            r = json.loads(line)
            if r["kind"] == "received":
                self.intents[r["id"]] = {"status": "pending", "tick": r["tick"], "slot": r["slot"], "act": r["act"]}
            elif r["kind"] == "applied":
                self.intents[r["id"]].update(status="applied", event=r["event"])
            elif r["kind"] == "rejected":
                self.intents[r["id"]] = {"status": "rejected", "reason": r["reason"]}
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
        return {"fire": {"family": [0, len(self.sim.state["families"]) - 1]}}

    def _check(self, msg: dict, scope: str) -> tuple[dict | None, str | None]:
        """The engine action for an intent, or why it is refused. Nothing here reads mutable world state:
        what the engine checks when the action applies stays the engine's (L2)."""
        now = self.sim.state["tick"]
        seen = msg.get("seen_tick")
        if scope != "god":
            return None, "scope"
        if not isinstance(msg.get("verb"), str) or msg["verb"] not in self.verbs():
            return None, "unknown_verb"
        args = msg.get("args")
        if not isinstance(args, dict) or set(args) != {"family"} or not _is_int(args["family"]) \
                or not 0 <= args["family"] < len(self.sim.state["families"]):
            return None, "bad_args"
        if not _is_int(seen) or not 0 <= seen <= now:
            return None, "bad_seen_tick"
        if now - seen > MAX_LAG:
            return None, "stale"
        return {"kind": "fire", "family": args["family"]}, None

    def submit(self, msg: dict, client: str = "local", scope: str = "god") -> dict | None:
        """Take an intent. Returns its ack now if it is refused or already decided, or None if it will be
        applied by the next step() (which returns its ack once its event is durable)."""
        iid = msg.get("id")
        if not isinstance(iid, str) or not 1 <= len(iid) <= 64:
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
        n0 = len(sim.events)
        self.run.advance(tick + 1)
        events = sim.events[n0:]
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

    def welcome(self, scope: str, paused: bool, tps: float) -> dict:
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
        self.scope, self.ready, self.closed = None, False, False
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

    def __init__(self, runner: LiveRunner, tps: float = 1.0, paused: bool = False):
        self.runner, self.tps, self.paused = runner, tps, paused
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
                    err = _hello_error(m)
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

    def _drain(self):
        items, self.inbox = self.inbox, []
        for conn, m in items:
            t = m["type"]
            if t == "hello":
                self._welcome(conn, m.get("since", 0))
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
            text = json.dumps(msg("tick", tick=tick, events=event_dicts(events), view=self.runner.view()))
            for c in self.conns:
                if c.ready:
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
        return "unsupported_scope" if scope == "inhabitant" else "bad_scope"
    since = m.get("since", 0)
    if not _is_int(since) or since < 0:
        return "bad_since"
    return None


async def serve(runner: LiveRunner, host: str = "127.0.0.1", port: int = 8765, tps: float = 1.0,
                paused: bool = False, started=None):
    """Run the hub until cancelled. `started(port, hub)` is called once the socket listens."""
    hub = Hub(runner, tps, paused)
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
    ap.add_argument("--mind", choices=["stub", "noisy"], default=None, help="default stub; on reopen, the run's")
    ap.add_argument("--tps", type=float, default=1.0, help="ticks per wall-clock second")
    ap.add_argument("--paused", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    kind = args.mind
    if kind is None:
        kind = (persist.config(args.out).get("mind") if os.path.exists(os.path.join(args.out, "run.json")) else None) or "stub"
    if kind not in ("stub", "noisy"):
        raise SystemExit(f"{args.out} was run with mind {kind!r}; the live runner serves stub and noisy minds only")
    runner = LiveRunner(args.out, StubMind() if kind == "stub" else NoisyMind(), args.seed, args.agents,
                        mind_kind=kind)
    print(f"{'reopened' if runner.reopened else 'new world'} {args.out} at tick {runner.sim.state['tick']}, "
          f"engine {ENGINE_VERSION}, ws://{args.host}:{args.port}/", file=sys.stderr, flush=True)
    try:
        asyncio.run(serve(runner, args.host, args.port, args.tps, args.paused))
    except KeyboardInterrupt:
        pass
    finally:
        runner.close()


if __name__ == "__main__":
    main()
