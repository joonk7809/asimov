# Headless: a playback pack, as the client reads it, equals the engine's replay at every 10th view tick on
# positions, holdings and who is at each place (G7's bar).
#   python client/godot/tools/make_playback.py LOG PACK.json.gz --expected EXPECTED.json
#   godot --headless --path client/godot --script res://tests/test_playback.gd -- PACK.json.gz EXPECTED.json
# EXPECTED is the engine's own state (Sim._pos, inv, place), written as the log replays. Exits 1 on any mismatch.
extends SceneTree

const WorldModel := preload("res://scripts/world_model.gd")


func _init() -> void:
	var a := OS.get_cmdline_user_args()
	var model := WorldModel.new()
	var err := model.load_file(a[0])
	if err != "":
		printerr(err)
		quit(1)
		return
	var want: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(a[1]))
	var bad := 0
	var mid_walk := 0
	var changed := 0   # holdings that differ from the villager's first: a pack that never updates would pass without this
	var crowded := 0   # places checked with two or more people
	var n: int = want["ticks"].size()
	for i in n:
		var t := int(want["ticks"][i])
		for aid in want["pos"][i].size():
			var p = want["pos"][i][aid]
			var s := model.state_at(aid, t)
			var h := model.holding_at(aid, t)
			if want["place"][i][aid] == null:
				mid_walk += 1
			if model.position(aid, t) != Vector2i(int(p[0]), int(p[1])) or s[1] != want["place"][i][aid]:
				bad += 1
				if bad <= 10:
					printerr("tick %d villager %d: client at %s %s, engine at %s %s" % [t, aid, model.position(aid, t), s[1], p, want["place"][i][aid]])
			if h.is_empty() or h[1] != int(want["food"][i][aid]) or h[2] != int(want["coin"][i][aid]):
				bad += 1
				if bad <= 10:
					printerr("tick %d villager %d: client holds %s, engine food %d coin %d" % [t, aid, h, want["food"][i][aid], want["coin"][i][aid]])
			elif h[1] != int(want["food"][0][aid]) or h[2] != int(want["coin"][0][aid]):
				changed += 1
		for place in model.places:
			var got := model.present(place, t)
			var exp := []
			for id in want["present"][i].get(place, []):
				exp.append(int(id))
			if exp.size() > 1:
				crowded += 1
			if got != exp:
				bad += 1
				if bad <= 10:
					printerr("tick %d %s: client has %s there, engine %s" % [t, place, got, exp])
	print("engine %s: %d view ticks (0 to %d), %d villagers, %d mid-walk, %d changed holdings, %d crowded places, %d mismatches" % [
			model.engine, n, int(want["ticks"][-1]), model.agents.size(), mid_walk, changed, crowded, bad])
	if n < 10 or mid_walk == 0 or changed == 0 or crowded == 0 or int(want["ticks"][-1]) != model.last_tick + 1:
		printerr("vacuous: too few ticks, nobody mid-walk, no holdings changed, nobody together, or the pack ends elsewhere")
		bad += 1
	quit(1 if bad else 0)
