# asiimov Godot client

2026-11-09. Block B10, Track A, PR 2. Godot 4.3-stable, GDScript only. The client draws and sends intents.
The Python engine is the only source of truth, over protocol 2 (`plan/client-protocol.md`).

    python -m asiimov.live --out runs/live --agents 25 --tps 2        # the live world
    godot --path client/godot -- --url ws://127.0.0.1:8765/           # live client
    godot --path client/godot -- --log runs/demo/events.jsonl         # a recorded log, with a scrubber
    xvfb-run -a godot --path client/godot --rendering-driver opengl3 -- --log LOG --at 230 --screenshot out.png

- `scripts/world_model.gd` rebuilds who is where from events alone. The engine moves villagers only in
  `activity` and `arrive` events, and positions use worldmap.position's integer math. Interpolation between
  ticks is for drawing only.
- `scripts/live_client.gd` speaks protocol 2: in god scope hello, backlog, ticks, acks, `fire` and controls; in inhabitant scope (`--inhabit N`) its villager's view, witnessed events and `act`. The action menu is the server's `offer`, one button per action (and person it may name), never computed in GDScript.
- `tests/test_inhabit.gd` (bar 6): inhabits a villager while a god client steps the world, presses menu buttons and checks every applied act's receipt arrives in its own stream.
- `scripts/main.gd` draws the map from genesis, villagers colored by activity, a timeline with a scrubber,
  play/pause and fire.

## Tests (headless, also in CI: `.github/workflows/godot.yml`, the pinned binary, SHA-512 checked)

    python client/godot/tests/make_fixture.py /tmp/fx --agents 100 --days 3 --seed 4
    godot --headless --path client/godot --script res://tests/test_positions.gd -- /tmp/fx
    python -m asiimov.live --out /tmp/live --paused &   # then:
    godot --headless --path client/godot --script res://tests/test_live.gd -- ws://127.0.0.1:8765/ 300

`test_positions.gd` checks every villager at every tick boundary against `Sim._pos`, read from the live sim
by `make_fixture.py`, never against the viewer. `test_live.gd` steps a paused live world and checks each
tick's rebuilt positions against the engine's `pos` in that tick's view, and that one `fire` is acked as
applied with its event's id.

Not done yet: web export (it needs export templates, which aren't installed) and sprites.
