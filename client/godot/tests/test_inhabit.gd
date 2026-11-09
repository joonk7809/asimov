# Headless, against a running live server started paused (python -m asiimov.live --paused):
#   godot --headless --path client/godot --script res://tests/test_inhabit.gd -- ws://127.0.0.1:8765/ AGENT [ticks]
# Bar 6 (plan/player-body-spec.md): one client inhabits villager AGENT, a god client steps the world. Every few
# ticks the inhabitant presses a button from its menu, which comes from the server's offer, and the test
# checks each applied act's receipt arrives in the inhabitant's own stream, citing the act event the ack
# named. It also checks the view's keys against the spec's whitelist. Exits 1 on any failure or timeout.
extends SceneTree

const LiveClient := preload("res://scripts/live_client.gd")
const VIEW_KEYS := ["tick", "place", "route", "pos", "activity", "asleep", "hunger", "food", "coin", "plan",
		"recent", "present", "kin", "market_food", "occasion", "offer", "cooldown_until"]

var god := LiveClient.new()
var me := LiveClient.new()
var want_ticks := 200
var seen := 0
var pressed := []       # intent ids sent
var buttons := 0
var started := 0
var done := false


func _init() -> void:
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
