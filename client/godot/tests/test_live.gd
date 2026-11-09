# Headless, against a running live server started paused (python -m asiimov.live --paused):
#   godot --headless --path client/godot --script res://tests/test_live.gd -- ws://127.0.0.1:8765/ [ticks]
# Steps the world tick by tick, fires once, and checks that every villager's position rebuilt from the
# streamed events equals the engine's own pos in that tick's view. Exits 1 on any mismatch or timeout.
extends SceneTree

const LiveClient := preload("res://scripts/live_client.gd")

var client := LiveClient.new()
var want_ticks := 300
var seen := 0
var bad := 0
var walkers := 0
var fire_id := ""
var started := 0
var done := false


func _init() -> void:
	var a := OS.get_cmdline_user_args()
	if a.size() > 1:
		want_ticks = int(a[1])
	client.welcomed.connect(_on_welcome)
	client.ticked.connect(_on_tick)
	client.failed.connect(func(r): _fail("client failed: " + r))
	started = Time.get_ticks_msec()
	if client.connect_to(a[0]) != OK:
		_fail("cannot connect to " + a[0])


func _on_welcome(m: Dictionary) -> void:
	if not m["paused"]:
		_fail("start the server with --paused, so every tick is checked")
	client.control("step")


func _on_tick(m: Dictionary) -> void:
	var v: Dictionary = m["view"]
	var t := int(v["tick"])
	for ag in v["agents"]:
		var aid := int(ag["id"])
		var exp := Vector2i(int(ag["pos"][0]), int(ag["pos"][1]))
		var got := client.model.position(aid, t)
		if ag["route"] != null:
			walkers += 1
		if got != exp or client.model.state_at(aid, t)[1] != ag["place"]:
			bad += 1
			if bad <= 10:
				printerr("view tick %d villager %d: client %s, engine %s" % [t, aid, got, exp])
	seen += 1
	if seen == 5:
		fire_id = client.fire(0)
	if seen < want_ticks:
		client.control("step")


func _process(_delta: float) -> bool:
	if done:
		return true
	client.poll()
	if Time.get_ticks_msec() - started > 60000:
		_fail("timed out after %d ticks" % seen)
	if seen >= want_ticks:
		var ack: Dictionary = client.acks.get(fire_id, {})
		if ack.get("status") != "applied":
			_fail("fire %s was not acked as applied: %s" % [fire_id, ack])
		var fired := client.model.events.filter(func(e): return e["kind"] == "fire" and e["origin"] == "player")
		if not done and (fired.size() != 1 or int(fired[0]["id"]) != int(ack["event"])):
			_fail("expected one fire event matching the ack, got %d" % fired.size())
		if done:
			return true
		print("checked %d ticks, %d walker positions, %d mismatches; fire acked as event %s"
				% [seen, walkers, bad, ack["event"]])
		done = true
		quit(1 if bad else 0)
	return false


func _fail(why: String) -> void:
	if done:
		return
	done = true
	printerr(why)
	quit(1)
