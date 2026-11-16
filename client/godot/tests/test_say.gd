# Headless, against a running live server started paused (python -m asiimov.live --paused), engine say-1 or later:
#   godot --headless --path client/godot --script res://tests/test_say.gd -- ws://127.0.0.1:8765/ AGENT
# The player talks to a villager (say-1). Once the body is awake with someone awake beside it, it says a line in
# its own words. The act must be applied, its receipt must say "said" and cite it, the `said` event must carry the
# words, and within a few ticks the listener's `reply` must come back to this body and read as words.
extends SceneTree

const LiveClient := preload("res://scripts/live_client.gd")
const Words := preload("res://scripts/inhabit_ui.gd")

const LINE := "Do you know where the good well is?"

var god := LiveClient.new()
var me := LiveClient.new()
var act_id := ""
var act_event := -1
var to := -1
var sent_tick := -1
var ticks := 0
var started := 0
var done := false


func _init() -> void:
	var names := {4: "Ari"}
	var said := Words.describe({"kind": "said", "actor": 3, "origin": "rule", "data": {"with": 4, "text": "Hi [there]"}},
			{"id": 3, "name": "Me"}, names)
	var heard := Words.describe({"kind": "reply", "actor": 4, "origin": "rule", "data": {"with": 3, "says": "Hello."}},
			{"id": 3, "name": "Me"}, names)
	var mute := Words.describe({"kind": "reply", "actor": 4, "origin": "rule", "data": {"with": 3, "says": ""}},
			{"id": 3, "name": "Me"}, names)
	if not ("Ari" in said and "Hi [lb]there]" in said) or not ("Ari" in heard and "Hello." in heard) \
			or not ("doesn't answer" in mute) or Words.WHY.get("said", "") == "":
		printerr("say doesn't read right: %s | %s | %s" % [said, heard, mute])
		quit(1)
		return
	var a := OS.get_cmdline_user_args()
	me.scope = "inhabitant"
	me.agent = int(a[1])
	god.failed.connect(func(x): _fail("god client failed: " + x))
	me.failed.connect(func(x): _fail("inhabitant client failed: " + x))
	me.welcomed.connect(func(_m): god.control("step"))
	me.ticked.connect(_on_tick)
	me.acked.connect(_on_ack)
	god.connect_to(a[0])
	me.connect_to(a[0])
	started = Time.get_ticks_msec()


func _on_ack(m: Dictionary) -> void:
	if m["id"] == act_id:
		if m["status"] != "applied":
			_fail("say rejected: %s" % [m])
		act_event = int(m["event"])


func _on_tick(m: Dictionary) -> void:
	ticks += 1
	var v: Dictionary = m["view"]
	if act_id == "" and not v["asleep"] and v["place"] != null:
		var awake: Array = v["present"].filter(func(p): return not p["asleep"])
		if not awake.is_empty():
			to = int(awake[0]["id"])
			act_id = me.act("say", to, "", LINE)
			sent_tick = int(v["tick"])
	if act_event >= 0:
		var rec: Array = me.seen_events.filter(func(e): return e["kind"] == "receipt" and not e["parents"].is_empty() \
				and int(e["parents"][0]) == act_event)
		if not rec.is_empty() and (not rec[0]["data"]["ok"] or rec[0]["data"]["why"] != "said"):
			_fail("say got %s" % [rec[0]["data"]])
		var said: Array = me.seen_events.filter(func(e): return e["kind"] == "said" and int(e["actor"]) == me.agent)
		var reply: Array = me.seen_events.filter(func(e): return e["kind"] == "reply" and int(e["actor"]) == to \
				and int(e["data"]["with"]) == me.agent)
		if not said.is_empty() and str(said[0]["data"]["text"]) != LINE:
			_fail("said carries %s" % [said[0]["data"]])
		if not rec.is_empty() and not said.is_empty() and not reply.is_empty():
			var line := Words.describe(reply[0], me.me, {to: "them"})
			if line == "" or not ("them" in line) or line.count(str(reply[0]["data"]["says"])) != 1:
				_fail("the reply doesn't read: %s" % [reply[0]])
			print("said to %d at tick %d, answered by tick %d: %s" % [to, sent_tick, int(v["tick"]), line])
			done = true
			quit(0)
			return
		if int(v["tick"]) - sent_tick > 4:
			_fail("no reply within 4 ticks: receipt %s, said %s" % [rec, said])
	if ticks > 200:
		_fail("nobody awake to talk to in 200 ticks")
	god.control("step")


func _fail(why: String) -> void:
	if done:
		return
	done = true
	printerr(why)
	quit(1)


func _process(_delta: float) -> bool:
	if done:
		return true
	god.poll()
	me.poll()
	if Time.get_ticks_msec() - started > 60000:
		_fail("timed out after %d ticks" % ticks)
	return false
