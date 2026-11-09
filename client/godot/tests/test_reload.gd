# Headless, against a running live server: a client whose `since` is past the server's log (as after a
# crash reopened the world shorter) gets bad_since, starts over from event 0, and ends with the whole log.
#   godot --headless --path client/godot --script res://tests/test_reload.gd -- ws://127.0.0.1:8765/
extends SceneTree

const LiveClient := preload("res://scripts/live_client.gd")

var client := LiveClient.new()
var started := 0
var welcomes := 0


func _init() -> void:
	for i in 100000:  # pretend we had seen far more of the world than the server has
		client.model.events.append({})
	client.welcomed.connect(func(_m): welcomes += 1)
	started = Time.get_ticks_msec()
	client.connect_to(OS.get_cmdline_user_args()[0])


func _process(_delta: float) -> bool:
	client.poll()
	var n := int(client.welcome.get("n_events", -1))
	if welcomes == 1 and client.model.events.size() == n and n > 0:
		var g: Dictionary = client.model.events[0]
		if g.get("kind") != "genesis":
			printerr("reloaded log doesn't start with genesis")
			quit(1)
			return true
		print("reloaded from 0 after bad_since: %d events" % n)
		quit(0)
		return true
	if Time.get_ticks_msec() - started > 30000:
		printerr("no reload: welcomes %d, events %d, server n_events %d" % [welcomes, client.model.events.size(), n])
		quit(1)
		return true
	return false
