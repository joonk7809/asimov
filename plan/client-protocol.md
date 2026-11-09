# Client protocol

2026-11-09. Status: proposal, protocol 1 (B10, Track A, PR 1). Implemented by `asiimov/live.py`.
Specs: plan/2d-env-eval.md and plan/client-form-factor.md in the shared folder.

The engine is the only source of truth. A client renders what the engine sends and sends intents.
It never sends state, and nothing it computes (interpolation, its own clock) is stored or sent back.

## Transport
WebSocket (RFC 6455), text frames, one JSON object per frame, UTF-8. Default `ws://127.0.0.1:8765/`.
A frame over 1 MiB from a client closes the connection. Server frames can be larger (a backlog chunk
with long mind outputs), so a client raises its inbound buffer (Godot: `inbound_buffer_size`).

Every server message carries `"protocol": 1` and `"engine": "<ENGINE_VERSION>"`. A client that sees a
protocol it doesn't know disconnects. A new engine version needs no protocol change unless a message
changes shape; new event kinds can appear in any engine, and a client ignores kinds it doesn't draw.

## Ticks and views
The world steps in ticks of 10 sim-minutes (144 a day). "View tick N" is the state between ticks:
tick N-1 has run, tick N hasn't. An intent received while the view is at N is applied when tick N
runs. Wall-clock time paces the runner but never enters the engine, the log or the state hash.

## Scopes
The first message names a scope. The scope decides what the server sends; it is fixed per connection.

| scope | protocol 1 | sends |
|---|---|---|
| `god` | yes | everything: genesis (population with souls, map), every event with its mind output, every villager's view. Dev and review. Accepts `control`. |
| `inhabitant` | reserved, refused with `unsupported_scope` | Track B. Only what the chosen villager perceives: its own state, the people at its place, the map. Never another villager's soul, mind text, needs or inventory. Its contents get pinned with Track B's player-body spec. |

## Client to server

`hello` must be the first message:
```json
{"type": "hello", "protocol": 1, "scope": "god", "since": 0}
```
`since` is the first event id the client still needs (default 0). A reattaching client passes the id
after the last event it has; the server sends the gap as backlog and then live ticks, with no event
sent twice and none skipped.

`intent`, a player verb:
```json
{"type": "intent", "id": "c1-0007", "verb": "fire", "args": {"family": 3}, "seen_tick": 210}
```
- `id`: the client's string, 1 to 64 characters, unique per world. Resending an id gets the recorded
  outcome again and never applies twice.
- `seen_tick`: the view tick the client was looking at. It's checked, not used to schedule.
- Verbs in protocol 1 (god scope), each a player action in sim.py `_player`:
  - `fire` with `{"family": int}`.
  - `possess` with `{"agent": int}`. Refused with `body_taken` while a body is held (release first).
  - `release` with `{}`. Refused with `no_body` if none is held.
  - `act` with `{"action": one of DELIBERATE_ACTIONS, "target": int?}`, on the possessed body. Refused
    with `no_body` if none is held. Whether the act works is the engine's call in that tick's commit
    phase, and the outcome is a `receipt` event citing the `act` event (plan/player-body-spec.md). The
    ack names the `act` event, not the receipt.
  Protocol 2 adds the inhabitant scope, which sends these on the client's behalf.

`control` (god scope only): `{"type": "control", "op": "pause" | "run" | "step" | "speed", "tps": 2.0}`.
`step` runs one tick while paused; `speed` takes `tps` between 0.1 and 1000.

## Server to client

`welcome`, once, answering `hello`:
```json
{"type": "welcome", "protocol": 1, "engine": "15", "scope": "god",
 "world": {"seed": 1, "order": "id", "map": {...}, "population": {...}, "ticks_per_day": 144},
 "view_tick": 211, "n_events": 6012, "paused": false, "tps": 1.0, "verbs": {"fire": {"family": [0, 24]}}}
```
`verbs` gives each verb's argument ranges (here, families 0 to 24 inclusive).

`events`, the backlog, in order, in chunks: `{"type": "events", "events": [...]}`. Each event is
exactly a line of `events.jsonl` (`id, tick, kind, origin, actor, data, parents, mind`). Backlog ends
before the first `tick` message.

`tick`, after each tick runs:
```json
{"type": "tick", "tick": 211, "events": [...],
 "view": {"tick": 212, "agents": [{"id": 0, "place": "home_0", "activity": "home", "asleep": false,
          "pos": [20, -20], "route": null}, ...]}}
```
`tick` is the tick that ran, `events` are the events it emitted, `view` is the state at view tick 212.
For a walker, `place` is null and `route` is `{"from": [x, y], "to": "market", "depart": 205,
"arrive": 214}`. `pos` is the engine's own position (worldmap.position) at the view tick, in tiles.
A client interpolates between views for display only.

`ack`, an intent's outcome, sent to the client that sent it:
- `{"type": "ack", "id": "c1-0007", "status": "applied", "tick": 211, "event": 6013}`, sent only after
  the intent's player event is fsynced in `events.jsonl`.
- `{"type": "ack", "id": "c1-0007", "status": "rejected", "reason": "bad_args"}`, sent at the next tick
  boundary. A rejected intent changes nothing in the world. Reasons: `bad_message`, `bad_id`,
  `unknown_verb`, `bad_args`, `bad_seen_tick` (missing, or ahead of the view), `stale` (more than a
  day behind the view), `scope`.

`status`, broadcast after a control: `{"type": "status", "paused": true, "tps": 1.0, "view_tick": 212}`.

`error`, then the server closes: a first message that isn't a valid `hello`, an unknown protocol, an
unsupported scope (`unsupported_scope`), a `since` past the log (`bad_since`), or a malformed frame. The
close frame repeats the reason with code 4000, since a client can drop a message that arrives just
before a close (Godot does).

## Durability
A run directory is a `persist.py` run plus one sidecar file:
- `run.json` holds the world's inputs. Its `player` list is the schedule of player actions. The live
  runner appends an accepted intent there (atomic write) before the tick it applies at, so
  `persist.resume`, `python -m asiimov.run --resume` and `replay()` all see it like a scripted `--fire`.
- `runner.jsonl` is the sidecar: wall-clock times, clients, controls and each intent's id, scope and
  outcome. It's runner metadata and never enters `events.jsonl` or the state hash.

An intent's path is: `received` fsynced in the sidecar, then scheduled in `run.json`, then the tick
runs, then `events.jsonl` is fsynced, then `applied` in the sidecar and the ack. A kill at any point
leaves the intent applied (it's in `run.json`), rescheduled from its `received` line on reopen, or
not acknowledged at all (it never reached the sidecar, and the client resends it by id). An intent
the sidecar received is never dropped.

A clean stop (Ctrl-C) snapshots the current tick, so reopening continues at the same view tick with
every event and ack, and a client reattaches with its own `since`.

After a crash, persist resumes from the newest snapshot and re-runs the recorded tail, dropping the last
recorded tick with events (and any empty ticks after it), so the view can move back more than one tick.
A client whose `since` is past the reopened log gets `bad_since` and reloads with `since: 0`. A
player action applies at the start of its tick, before any mind is asked, so its event id survives the
re-run and its ack stands. With a live LLM mind, the rest of a re-run tick can differ from what was
streamed before the crash.
