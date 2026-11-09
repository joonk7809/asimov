# Headless: the client's positions, rebuilt from a recorded log, equal the engine's at every tick boundary.
#   godot --headless --path client/godot --script res://tests/test_positions.gd -- FIXTURE_DIR
# FIXTURE_DIR comes from tests/make_fixture.py. Exits 1 on any mismatch.
extends SceneTree

const WorldModel := preload("res://scripts/world_model.gd")


func _init() -> void:
	var dir: String = OS.get_cmdline_user_args()[0]
	var model := WorldModel.new()
	for line in FileAccess.get_file_as_string(dir.path_join("events.jsonl")).split("\n", false):
		model.add_event(JSON.parse_string(line))
	var want: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(dir.path_join("expected.json")))
	var checked := 0
	var mid_walk := 0
	var bad := 0
	for i in want["ticks"].size():
		var t := int(want["ticks"][i])
		for aid in want["pos"][i].size():
			var p = want["pos"][i][aid]
			var exp := Vector2i(int(p[0]), int(p[1]))
			var got := model.position(aid, t)
			var s := model.state_at(aid, t)
			var exp_place = want["place"][i][aid]
			checked += 1
			if exp_place == null:  # the engine has this villager on a route
				mid_walk += 1
			if got != exp or s[1] != exp_place:
				bad += 1
				if bad <= 10:
					printerr("tick %d villager %d: client %s at %s, engine %s at %s" % [t, aid, got, s[1], exp, exp_place])
	print("checked %d positions at %d boundaries, %d mid-walk, %d mismatches" % [checked, want["ticks"].size(), mid_walk, bad])
	if model.events.is_empty() or mid_walk == 0:
		printerr("vacuous: no events or no villager caught mid-walk")
		bad += 1
	quit(1 if bad else 0)
