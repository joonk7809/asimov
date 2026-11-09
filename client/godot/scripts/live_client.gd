# Protocol 1 client (plan/client-protocol.md): hello, then welcome, backlog and ticks into a WorldModel.
# Sends intents and controls only; never state.
extends RefCounted

const PROTOCOL := 1
const WorldModel := preload("res://scripts/world_model.gd")

signal welcomed(msg: Dictionary)
signal ticked(msg: Dictionary)
signal acked(msg: Dictionary)
signal status_changed(msg: Dictionary)
signal failed(reason: String)

var model := WorldModel.new()
var ws := WebSocketPeer.new()
var url := ""
var scope := "god"
var welcome := {}
var view := {}          # the latest engine view: {tick, agents: [{id, place, activity, asleep, pos, route}]}
var view_tick := 0
var tick_wall := 0.0    # Time.get_ticks_msec() when the latest view arrived, for interpolation only
var tps := 1.0
var paused := false
var acks := {}          # intent id -> ack
var _hello_sent := false
var _closed := false
var _reload := false   # the server refused our since (the world reopened shorter after a crash)
var _seq := 0


func connect_to(u: String) -> int:
	url = u
	ws.inbound_buffer_size = 64 << 20  # backlog chunks and the welcome can be several MB
	ws.outbound_buffer_size = 1 << 20
	return ws.connect_to_url(url)


func poll() -> void:
	ws.poll()
	var state := ws.get_ready_state()
	if state == WebSocketPeer.STATE_OPEN and not _hello_sent:
		_hello_sent = true
		_send({"type": "hello", "protocol": PROTOCOL, "scope": scope, "since": model.events.size()})
	while ws.get_available_packet_count() > 0:  # also while closing: an error arrives just before the close
		_handle(ws.get_packet().get_string_from_utf8())
	if state != WebSocketPeer.STATE_CLOSED:
		return
	if ws.get_close_reason() == "bad_since" and not model.events.is_empty():
		_reload = true  # Godot can drop the error message that came just before the close; the close repeats it
	if _reload:  # start over from event 0, as the protocol asks after bad_since
		_reload = false
		_hello_sent = false
		model.events.clear()
		model.last_tick = -1
		connect_to(url)
	elif not _closed:
		_closed = true
		failed.emit("closed %d %s" % [ws.get_close_code(), ws.get_close_reason()])


func _handle(text: String) -> void:
	var m = JSON.parse_string(text)
	if typeof(m) != TYPE_DICTIONARY:
		failed.emit("not a JSON object")
		return
	if int(m.get("protocol", -1)) != PROTOCOL:
		failed.emit("unknown protocol %s" % m.get("protocol"))
		ws.close()
		return
	match m["type"]:
		"welcome":
			welcome = m
			view_tick = int(m["view_tick"])
			tps = float(m["tps"])
			paused = m["paused"]
			if model.events.is_empty():
				model.load_world(m["world"])
			welcomed.emit(m)
		"events":
			for e in m["events"]:
				model.add_event(e)
		"tick":
			for e in m["events"]:
				model.add_event(e)
			view = m["view"]
			view_tick = int(view["tick"])
			tick_wall = Time.get_ticks_msec()
			ticked.emit(m)
		"ack":
			acks[str(m["id"])] = m
			acked.emit(m)
		"status":
			paused = m["paused"]
			tps = float(m["tps"])
			status_changed.emit(m)
		"error":
			if m["reason"] == "bad_since" and not model.events.is_empty():
				_reload = true
				return
			failed.emit(str(m["reason"]))


func _send(m: Dictionary) -> void:
	ws.send_text(JSON.stringify(m))


func fire(family: int) -> String:
	## Sends the fire intent; the ack arrives once the engine has applied it and the log is on disk.
	_seq += 1
	var id := "g%x%x-%d" % [Time.get_ticks_usec(), randi(), _seq]  # unique per world, as the protocol asks
	_send({"type": "intent", "id": id, "verb": "fire", "args": {"family": family}, "seen_tick": view_tick})
	return id


func control(op: String, rate: float = 0.0) -> void:
	var m := {"type": "control", "op": op}
	if op == "speed":
		m["tps"] = rate
	_send(m)


func display_tick() -> float:
	## The view tick plus how far into the next tick the wall clock is, for drawing only.
	if paused or view.is_empty():
		return float(view_tick)
	return view_tick + clampf((Time.get_ticks_msec() - tick_wall) / 1000.0 * tps, 0.0, 1.0)
