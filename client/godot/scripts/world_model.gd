# Where each villager is, rebuilt from the engine's events alone. Render-only: nothing here is sent back.
#
# The engine changes a villager's place or route in exactly two events (sim.py _set_activity and _arrive):
#   activity  {activity, place, arrive[, from]}  a walk when arrive > tick (from: where it starts; absent when
#             the villager keeps its current walk), else already there
#   arrive    {place}                            the walk ends
# wake and sleep only change the activity. Position math is worldmap.position, in integer tiles, so at a
# tick boundary this matches the engine exactly; tests/test_positions.gd checks it against Sim._pos.
extends RefCounted

var places := {}        # name -> {anchor: [x, y], rect: [x, y, w, h]}
var walk := 40          # tiles per tick, from genesis
var agents := []        # population agents, as in genesis
var families := []
var ticks_per_day := 144
var moves := {}         # agent id -> [[tick, place, route, activity], ...]: state after that tick's events
var events := []        # every event received, in id order
var last_tick := -1     # the newest tick with events applied


func load_world(world: Dictionary) -> void:
	## genesis.data, or welcome.world. Resets everything.
	places = world["map"]["places"]
	walk = int(world["map"]["walk"])
	families = world["population"]["families"]
	agents = world["population"]["agents"]
	ticks_per_day = int(world.get("ticks_per_day", 144))
	moves.clear()
	for a in agents:
		moves[int(a["id"])] = [[-1, families[int(a["family"])]["home"], null, "sleep"]]


func add_event(e: Dictionary) -> void:
	var t := int(e["tick"])
	var d: Dictionary = e["data"]
	if e["kind"] == "genesis":
		events.clear()
		load_world(d)
	events.append(e)
	last_tick = maxi(last_tick, t)
	match e["kind"]:
		"activity":
			var aid := int(e["actor"])
			var arrive := int(d.get("arrive", t))
			if arrive > t:
				var route = _last(aid)[2]  # no "from": still on the same walk
				if d.has("from"):
					route = {"from": [int(d["from"][0]), int(d["from"][1])], "to": d["place"],
							"depart": t, "arrive": arrive}
				_put(aid, t, null, route, d["activity"])
			else:
				_put(aid, t, d["place"], null, d["activity"])
		"arrive":
			var aid := int(e["actor"])
			_put(aid, t, d["place"], null, _last(aid)[3])
		"wake":
			var aid := int(e["actor"])
			var m = _last(aid)
			_put(aid, t, m[1], m[2], "home")
		"sleep":
			var aid := int(e["actor"])
			var m = _last(aid)
			_put(aid, t, m[1], m[2], "sleep")


func _last(aid: int) -> Array:
	return moves[aid][-1]


func _put(aid: int, t: int, place, route, activity) -> void:
	var m: Array = moves[aid]
	if m[-1][0] == t:
		m.pop_back()  # only the last change in a tick is visible between ticks
	m.append([t, place, route, activity])


func state_at(aid: int, view_tick: int) -> Array:
	## [tick, place, route, activity] at view tick N: after tick N-1 ran, before tick N.
	var m: Array = moves[aid]
	var lo := 0
	var hi := m.size() - 1
	while lo < hi:  # the last entry with tick <= view_tick - 1
		var mid := (lo + hi + 1) / 2
		if m[mid][0] <= view_tick - 1:
			lo = mid
		else:
			hi = mid - 1
	return m[lo]


func anchor(place: String) -> Vector2i:
	var a = places[place]["anchor"]
	return Vector2i(int(a[0]), int(a[1]))


func position(aid: int, view_tick: int) -> Vector2i:
	## The engine's position at a tick boundary (worldmap.position): WALK tiles a tick along the L path, x first.
	var s := state_at(aid, view_tick)
	if s[2] == null:
		return anchor(s[1])
	var r: Dictionary = s[2]
	var to := anchor(r["to"])
	var x := int(r["from"][0])
	var y := int(r["from"][1])
	var left := walk * maxi(0, view_tick - int(r["depart"]))
	var dx := mini(left, absi(to.x - x))
	x += dx if to.x >= x else -dx
	var dy := mini(left - dx, absi(to.y - y))
	y += dy if to.y >= y else -dy
	return Vector2i(x, y)


func display_position(aid: int, t: float) -> Vector2:
	## Between boundaries, for drawing only: linear between the two boundary positions.
	var t0 := int(floor(t))
	var a := Vector2(position(aid, t0))
	var b := Vector2(position(aid, t0 + 1))
	return a.lerp(b, t - t0)
