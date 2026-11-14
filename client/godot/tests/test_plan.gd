# Headless, against a running live server started paused (python -m asiimov.live --paused), engine 24 or later:
#   godot --headless --path client/godot --script res://tests/test_plan.gd -- ws://127.0.0.1:8765/ AGENT
# The player's own plan (readable-world C2): the plan queue's rules, then a real plan run against the engine.
# Steps are sent one at a time as the client sends them: go to the market, buy food, walk home, rest. Each
# settles by Plan.settle on the receipts citing its act: the walks on their arrival, buy_food on its receipt,
# rest by holding until "Next". The body must end at home, and every step must have settled.
extends SceneTree

const LiveClient := preload("res://scripts/live_client.gd")
const Plan := preload("res://scripts/plan_queue.gd")

var god := LiveClient.new()
var me := LiveClient.new()
var plan := []
var state := ""
var act_id := ""
var act_event := -1
var settled := []
var started := 0
var ticks := 0
var done := false


func _init() -> void:
	var go := Plan.step("go_to", -1, "market", "Go to the market")
	var r := func(why: String, ok := true) -> Dictionary: return {"id": randi(), "data": {"ok": ok, "why": why}}
	var cases := [[go, [], "wait"], [go, [r.call("heading there")], "wait"],
			[go, [r.call("heading there"), r.call("arrived")], "done"], [go, [r.call("already here")], "done"],
			[go, [r.call("asleep", false)], "failed: asleep"],
			[Plan.step("buy_food"), [r.call("bought")], "done"], [Plan.step("buy_food"), [r.call("no coin", false)], "failed: no coin"],
			[Plan.step("work"), [r.call("started")], "hold"], [Plan.step("ask_food", 4), [r.call("refused", false)], "failed: refused"]]
	for c in cases:
		if Plan.settle(c[0], c[1]) != c[2]:
			printerr("settle %s on %s is %s, not %s" % [c[0]["action"], c[1], Plan.settle(c[0], c[1]), c[2]])
			quit(1)
			return
	var shown := Plan.lines([go, Plan.step("buy_food", -1, "", "Buy food")], "failed: no coin", {"no coin": "you have no coin"})
	if not ("Go to the market" in shown and "you have no coin" in shown and "2. Buy food" in shown):
		printerr("the plan doesn't read right: " + shown)
		quit(1)
		return
	var a := OS.get_cmdline_user_args()
	me.scope = "inhabitant"
	me.agent = int(a[1])
	god.failed.connect(func(x): _fail("god client failed: " + x))
	me.failed.connect(func(x): _fail("inhabitant client failed: " + x))
	me.welcomed.connect(_on_welcome)
	me.ticked.connect(_on_tick)
	me.acked.connect(_on_ack)
	god.connect_to(a[0])
	me.connect_to(a[0])
	started = Time.get_ticks_msec()


func _on_welcome(m: Dictionary) -> void:
	var home := str(m["world"]["me"]["home"])
	plan = [Plan.step("go_to", -1, "market", "Go to the market"), Plan.step("buy_food", -1, "", "Buy food"),
			Plan.step("go_to", -1, home, "Go home"), Plan.step("rest", -1, "", "Rest")]
	god.control("step")


func _on_ack(m: Dictionary) -> void:
	if m["id"] == act_id:
		if m["status"] != "applied":
			_fail("step %s rejected: %s" % [plan[0], m])
		act_event = int(m["event"])


func _on_tick(m: Dictionary) -> void:
	ticks += 1
	var v: Dictionary = m["view"]
	if not plan.is_empty() and act_event >= 0:
		var rs := me.seen_events.filter(func(e): return e["kind"] == "receipt" and not e["parents"].is_empty() \
				and int(e["parents"][0]) == act_event)
		state = Plan.settle(plan[0], rs)
		if state.begins_with("failed"):
			_fail("step %s failed: %s" % [plan[0]["label"], state])
		if state == "done" or state == "hold":
			settled.append([plan[0]["label"], state, int(v["tick"])])
			if state == "hold" and str(v["place"]) != str(me.me["home"]):
				_fail("rested away from home at %s" % v["place"])
			plan.pop_front()
			act_event = -1
			act_id = ""
	if plan.is_empty():
		_finish(v)
		return
	if act_id == "" and not v["asleep"]:
		act_id = me.act(plan[0]["action"], int(plan[0]["target"]), str(plan[0]["place"]))
	if ticks > 100:
		_fail("the plan didn't finish in 100 ticks: %s, step %s is %s" % [settled, plan[0]["label"], state])
	god.control("step")


func _finish(v: Dictionary) -> void:
	if settled.size() != 4 or settled[0][0] != "Go to the market" or settled[3][1] != "hold":
		_fail("steps settled wrong: %s" % [settled])
	print("plan of 4 steps settled in %d ticks: %s" % [ticks, settled])
	done = true
	quit(0)


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
