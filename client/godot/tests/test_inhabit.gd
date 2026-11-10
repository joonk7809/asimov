# Headless, against a running live server started paused (python -m asiimov.live --paused):
#   godot --headless --path client/godot --script res://tests/test_inhabit.gd -- ws://127.0.0.1:8765/ AGENT [ticks]
# Bar 6 (plan/player-body-spec.md): one client inhabits villager AGENT, a god client steps the world. Every few
# ticks the inhabitant presses a button from its menu, which comes from the server's offer, and the test
# checks each applied act's receipt arrives in the inhabitant's own stream, citing the act event the ack
# named. It also checks the view's keys against the spec's whitelist, and that what it witnessed reads as words. Exits 1 on any failure or timeout.
extends SceneTree

const LiveClient := preload("res://scripts/live_client.gd")
const Words := preload("res://scripts/inhabit_ui.gd")
const VIEW_KEYS := ["tick", "place", "route", "pos", "activity", "asleep", "hunger", "food", "coin", "plan",
		"recent", "present", "kin", "market_food", "occasion", "offer", "cooldown_until", "food_price", "intent", "asks"]

var god := LiveClient.new()
var me := LiveClient.new()
var want_ticks := 200
var seen := 0
var pressed := []       # intent ids sent
var buttons := 0
var started := 0
var done := false


func _init() -> void:
	# Which buttons are left off: only the ones the engine can only refuse (sim.INSTANT_STEPS elsewhere,
	# ask_kin of non-family). Family elsewhere keeps the gifts, seek and avoid: a gift walks there.
	var elsewhere := {}
	var here_kin := {7: true}
	var here_other := {8: false}
	for c in [["give_food", 5, elsewhere, true], ["give_coin", 5, elsewhere, true], ["avoid", 5, elsewhere, true],
			["seek", 5, elsewhere, true], ["ask_food", 5, elsewhere, false], ["buy_from", 5, elsewhere, false],
			["steal", 5, elsewhere, false], ["ask_kin", 5, elsewhere, false], ["ask_kin", 7, here_kin, true],
			["ask_kin", 8, here_other, false], ["ask_food", 8, here_other, true], ["work", -1, elsewhere, true]]:
		if Words.shown({"action": c[0], "target": c[1]}, c[2]) != c[3]:
			printerr("button %s for %d shown should be %s" % [c[0], c[1], c[3]])
			quit(1)
			return
	# An ask waiting on the villager's answer (player-ask-1) reads as one, with the time left, and the
	# answer buttons are shown wherever the asker is.
	var waiting := Words.asks({"tick": 100, "asks": [{"from": 4, "name": "Ari", "action": "ask_food", "until": 104}]}, {}, 20.0)
	var said := Words.describe({"kind": "ask", "actor": 4, "origin": "rule", "data": {"action": "buy_from", "to": 3}},
			{"id": 3, "name": "Me"}, {4: "Ari"})
	if not ("Ari asked you for food" in waiting and "4 ticks" in waiting and "80s" in waiting) \
			or said != "Ari wanted to buy food from you. Say yes or no." \
			or not Words.shown({"action": "agree", "target": 4}, {}) or not Words.shown({"action": "decline", "target": 4}, {}):
		printerr("an ask doesn't read right: %s | %s" % [waiting, said])
		quit(1)
		return
	var a := OS.get_cmdline_user_args()
	if a.size() > 2:
		want_ticks = int(a[2])
	me.scope = "inhabitant"
	me.agent = int(a[1])
	god.failed.connect(func(r): _fail("god client failed: " + r))
	me.failed.connect(func(r): _fail("inhabitant client failed: " + r))
	me.welcomed.connect(_on_welcome)
	me.ticked.connect(_on_tick)
	started = Time.get_ticks_msec()
	if god.connect_to(a[0]) != OK or me.connect_to(a[0]) != OK:
		_fail("cannot connect to " + a[0])


func _on_welcome(m: Dictionary) -> void:
	if int(m["world"]["me"]["id"]) != me.agent or m["verbs"].keys() != ["act"]:
		_fail("welcome is not for villager %d in inhabitant scope: %s" % [me.agent, m["world"]["me"]])
	_check_view(m["view"])
	god.control("step")


func _on_tick(m: Dictionary) -> void:
	var v: Dictionary = m["view"]
	_check_view(v)
	var menu := LiveClient.menu(v)
	var want := 0
	for action in v["offer"]:
		want += maxi(1, v["offer"][action].size())
	if menu.size() != want:
		_fail("menu has %d buttons for an offer of %d" % [menu.size(), want])
	buttons += menu.size()
	seen += 1
	if not menu.is_empty() and seen % 3 == 0:
		var b: Dictionary = menu[(seen / 3) % menu.size()]
		var named := []  # JSON numbers arrive as floats
		for t in v["offer"].get(b["action"], []):
			named.append(int(t))
		if not v["offer"].has(b["action"]) or (b["target"] >= 0 and not b["target"] in named):
			_fail("button %s is not in the offer" % b)
		pressed.append(me.act(b["action"], b["target"]))
	if seen < want_ticks:
		god.control("step")


func _check_view(v: Dictionary) -> void:
	for k in v:
		if not k in VIEW_KEYS:
			_fail("view field %s is not in the spec's whitelist" % k)
	for p in v["present"]:
		if p["pos"] != v["pos"]:
			_fail("a present villager's pos %s is not at its place %s" % [p["pos"], v["pos"]])


func _process(_delta: float) -> bool:
	if done:
		return true
	god.poll()
	me.poll()
	if Time.get_ticks_msec() - started > 60000:
		_fail("timed out after %d ticks" % seen)
	if seen >= want_ticks and not done:
		var receipts := {}
		for e in me.seen_events:
			if e["kind"] == "receipt" and int(e["actor"]) == me.agent and not e["parents"].is_empty():
				receipts[int(e["parents"][0])] = e["data"]["why"]
		var applied := 0
		var receipted := 0
		for id in pressed:
			var ack: Dictionary = me.acks.get(id, {})
			if ack.get("status") == "applied":
				applied += 1
				receipted += int(receipts.has(int(ack["event"])))
			elif ack.get("status") == "rejected":
				_fail("act %s was rejected: %s" % [id, ack])
		if done:
			return true
		var names := {}  # every witnessed event reads as words, and every one of its own receipts says something
		var worded := 0
		for e in me.seen_events:
			var line := Words.describe(e, me.me, names)
			worded += int(line != "")
			if e["kind"] == "receipt" and int(e["actor"]) == me.agent and line == "" and not e["data"]["ok"]:
				_fail("a refused act has no words: %s" % e)
		if done:
			return true
		if worded == 0:
			_fail("no witnessed event read as words")
			return true
		if applied < 20 or receipted != applied or buttons == 0:  # an ack follows the tick that carries its receipt
			_fail("%d acts sent, %d applied, %d with receipts, %d buttons shown" % [pressed.size(), applied, receipted, buttons])
			return true
		print("inhabited villager %d for %d ticks: %d buttons shown, %d acts applied, %d with receipts"
				% [me.agent, seen, buttons, applied, receipted])
		done = true
		quit(0)
	return false


func _fail(why: String) -> void:
	if done:
		return
	printerr(why)
	done = true
	quit(1)
