# The player's body (B10 PR 3, Track B spec)

2026-11-09. Status: proposal. This is the spec, written before any engine code. The build is an engine
bump under plan/engine-numbering-rule.md (shared folder): branch label `pb-1`, and Steering assigns the
integer at merge. It needs review and Steering's sign-off before the build starts.

joon, 2026-11-09 02:53Z: "player could be both, most likely an inhabitant." So the main mode is
inhabiting one villager, and divine verbs are a second, separate family. Open for joon: what the body
does when the player gives no orders (the decision card in thread "Build the Godot client for the live
world"). This spec is written for the recommended option, autopilot by the villager's own mind, and
notes what changes under the others.

## Principle
The player takes over the villager's **in-the-moment decisions**, and nothing else. The engine already
has exactly one place where those are made: a deliberation, whose output `_resolve(a, out, eid)` turns
into an action now and maybe an override step (sim.py `_resolve`). A player order is a deliberation
whose output comes from a person instead of a model. It's recorded as an `origin="player"` event, and
`_resolve` runs it with every check a mind's choice gets: `_target_error`, the live place checks in
`_person_act` (L2), receipts and chance rolls. One thing a mind has that `_resolve` doesn't check is the
offer: a mind is held to `_offer` only by its reply schema, and `_resolve` itself never reads the offer
(review of #38: with 0 coin, `_resolve(go_market)` still starts a shop step). So the player path adds
an explicit offer check (change 2). With it, the player can't do anything a villager couldn't choose,
and replay holds by S2: player events are recorded inputs, like `fire`.

## Engine changes (in `Sim._player`, plus the deliberation point)
1. `possess {agent}` (player event): sets `state["player"] = {"agent": id, "event": eid}`. The key is
   added only on the first possess, so a world with no player has the same state, hash and log as
   today, apart from the genesis engine field. Re-possessing moves the body, and `release` removes the key.
2. `act {action, target?}` (player event, on the possessed villager only). **When it resolves:** in the
   same phase as a mind's deliberation, not at the start of the tick. `_player` only queues the act
   (`fire` keeps applying at the start). In the agent loop, where `_maybe_deliberate` would run for this
   villager (after arrivals, occupancy, `_body`, `_act` and `_eat`), the engine:
   - emits the `act` player event;
   - computes `view = _view(a, occupancy)`, `_situation` (which fills `view["people"]`) and
     `offer = _offer(a, view)`, exactly as a deliberation request would;
   - registers it in `pending` with the player's output as its reply.

   `_commit` then applies it in `_order` among the minds' replies. So its receipts, rolls and ordering
   against other villagers are those a mind's identical deliberation would get on that tick. At commit
   it runs these checks in order, each a failed receipt with a named reason and no other effect:
   - `action not in offer` gives `"not offered"`, with the offer recorded in the receipt;
   - a target outside `view["people"]` for a person action gives `"unknown target"`;
   - then `_resolve(a, {"action": action, "target": target}, eid)`.

   If the body is asleep, or walking home for bed, on the tick the act arrives, the act event is
   emitted with a failed receipt, `"asleep"` or `"bedtime"`. Queued acts never carry over to the next
   tick: the client resends after reading the receipt. The `goal`/`then` intent fields aren't exposed,
   because the player is the intent. Replay is unchanged: `replay()` passes the player events of each
   tick back to `step`, and `_player` queues the acts again.
3. Guard: `_maybe_deliberate` skips the possessed villager. Its occasions (hungry, robbed, refused, fire and the rest of `OCCASIONS`) still
   fire, and they're shown to the player as prompts to act (the inhabitant view's `occasion`).
4. Under the recommended option, wake plans and night reflection still go to the villager's mind, so
   it keeps one coherent memory and a day plan. The habit/plan layer runs between orders, and an order
   overrides the current step as a deliberation's override does today. (Option 1, standing still, also
   skips plan and reflect and leaves `plan` empty. Option 2 adds an idle counter that lifts the step-3
   guard after 6 ticks without an order.)

No prompt text changes. Other villagers see the possessed villager as an ordinary person. So L11, L18
and L19 have nothing to re-measure, and no test reads a decision this changes, except the possessed
villager's own. The scorecard has no way to exclude a villager yet, so the build adds one: any villager
with a `possess` event in the log is dropped from every scorecard and diagnostic reading, with the count
dropped reported. B5/B6's test-11 arm
doesn't apply: this adds no mind choices.

D2's rule, that a reply lands on the tick it's asked, holds. Everything another villager asks of the
player's body (ask_kin, ask_food, buy_from, steal against it) is resolved by rule rolls today, so it
never waits on the player. Free speech, when B6 adds it, is the one place a mind would wait for the
player: the player's reply lands on the next tick as a player event, never blocking a tick. That's an
explicit exception for the player's body only, and it needs Steering's sign-off when B6 specs it.

## Protocol 2: the inhabitant scope
`hello {"protocol": 2, "scope": "inhabitant", "agent": id}`. The runner sends `possess` as an intent
on hello. Two inhabitants can't hold one body: the second gets `body_taken`. When an inhabitant
disconnects, the runner sends `release`, and the mind's normal deliberations resume.

What an inhabitant gets, and nothing more (the server filters; the client is never trusted to hide):
- `welcome`: the map, its own villager (name, role, family, home), its family's ids and names. Not the
  population's souls, traits or params, not even its own villager's numbers. The soul shows in behavior.
- Per tick, a `view` built from `Sim._view` plus `_situation`'s facts: its own place or route, pos,
  activity, hunger, food, coin, plan blocks, recent notes, the people present (id, name, role, kin,
  has_food, the same fields a mind sees), market food if at the market, the current occasion, and the
  actions `_offer` allows now. Positions of other villagers appear only for those at its place (and its
  family's homes, which are known).
- Events: only those it would witness. That means events with it as actor or target, events whose
  `data.to` or receipt names it, events at its place this tick by people present, and public events
  (`fire`, which the village sees). No mind output, prompt hash or raw text of any other villager, and
  for its own villager only the plan blocks, never the raw reply.
- Intents: `act` with the action and target, and `seen_tick`. Divine verbs are refused with `scope`.

A god-scope session is unchanged and keeps `fire`. "Both" is one person holding two connections, an
inhabitant and a god, allowed only when the world's run.json says `"divine": true`. The god connection
is the dev/review tool until a game design says what divine power costs.

## Bars (the build must pass these before review)
1. **No-player worlds unchanged:** for seeds 1 to 3 at 100 villagers over 3 days, the log is identical
   to the previous engine's apart from the genesis `engine` field, and the state hash at every
   checkpoint matches.
2. **Live = batch** with possess, act and release at scripted ticks, hash for hash, plus `replay()` and
   the S8 kill/reopen fuzz from tests/test_live.py, extended to the new verbs.
3. **Same rules as a mind (S1):** a hostile player sends every action with random, missing, wrong-typed
   and absent targets every tick for 3 days. Invariants and ledgers are checked every tick
   (test_invariants), and every refusal is a receipt with the same reason a mind's identical choice
   gets. Every act outside that tick's offer is refused with `"not offered"` and changes nothing else
   (this is checked against a twin where the act is absent: the same log apart from the act and
   receipt events). For offered acts: replaying each one as a recorded mind deliberation for the same
   villager at the same tick gives the same receipts, because both resolve in `_commit` in `_order`.
4. **No deliberation while possessed:** zero `deliberate` mind events for the possessed villager
   between possess and release. Its plan and reflect events continue (recommended option).
5. **Inhabitant leak test:** over a 3-day world with a player in the inhabitant scope, every message
   sent is scanned. It must contain no `soul`, `traits`, `params`, `raw_output` or `prompt_hash` of any
   villager, and no position of a villager who isn't at the player's place, unless it's family at
   home. Fuzz the hello (`agent` out of range, wrong type, already taken).
6. **Client:** the Godot client binds to its villager, shows the offered actions as buttons (the menu
   comes from the server, never computed in GDScript), and its headless test drives `act` and checks
   the receipts against the stream.

## Cost
$0, using stub and noisy minds. A real-mind playtest needs its own charter, on a pod that isn't serving
a canonical run (T21's batching caveat).

## What would change this
- If joon picks option 1 (stands still), step 4 changes, and the possessed villager's memory has
  gaps. Bar 4 then expects no plan or reflect either.
- If playtests show that orders through the deliberation menu feel too coarse (no free walking to a
  tile), a `walk {place}` verb maps onto `_set_activity` with the `explore` activity. It's still one
  rules path, but it's a new action the minds don't have, so it would need its own spec line and L19
  check.
