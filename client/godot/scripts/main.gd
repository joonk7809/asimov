# The client: draws the map from genesis, villagers from events, a timeline with a scrubber, and sends fire.
#   godot --path client/godot -- --url ws://127.0.0.1:8765/     live, against python -m asiimov.live
#   ... --url URL --inhabit 3                                     live as villager 3: its view and its action menu
#   godot --path client/godot -- --log runs/demo/events.jsonl   a recorded log
#   ... --log LOG --at TICK --screenshot out.png                  save one frame and quit (needs a display: xvfb-run;
#                                                                 live, it waits about 3 s first)
# Interpolation between ticks is for drawing only. Nothing drawn here is sent to the engine.
extends Node2D

const WorldModel := preload("res://scripts/world_model.gd")
const LiveClient := preload("res://scripts/live_client.gd")
const COLORS := {"sleep": Color(0.35, 0.35, 0.5), "home": Color(0.6, 0.6, 0.7), "work": Color(0.85, 0.6, 0.2),
		"shop": Color(0.3, 0.7, 0.4), "socialize": Color(0.8, 0.4, 0.7), "rest": Color(0.5, 0.7, 0.9),
		"explore": Color(0.9, 0.9, 0.5)}  # sim.py ACTIVITIES plus sleep; anything newer is drawn white

var model: WorldModel
var client: LiveClient = null
var replay_tick := 0.0      # recorded mode: where the scrubber is
var playing := false
var follow := true          # live mode: the view follows the newest tick
var scrub_tick := 0
var ui := {}
var shot := {}              # --screenshot: {path, frames left}


func _ready() -> void:
	var args := _args()
	_build_ui()
	if args.has("log"):
		model = WorldModel.new()
		for line in FileAccess.get_file_as_string(args["log"]).split("\n", false):
			var e = JSON.parse_string(line)
			if typeof(e) == TYPE_DICTIONARY:  # a torn last line is skipped
				model.add_event(e)
		ui["status"].text = "recorded log, %d events" % model.events.size()
		replay_tick = float(args.get("at", "0"))
	else:
		client = LiveClient.new()
		model = client.model
		if args.has("inhabit"):
			client.scope = "inhabitant"
			client.agent = int(args["inhabit"])
			client.ticked.connect(func(_m): _rebuild_menu())
			client.welcomed.connect(func(m): ui["status"].text = "inhabiting " + m["world"]["me"]["name"]; _rebuild_menu())
			for k in ["family", "fire", "play"]:  # divine verbs and controls aren't an inhabitant's
				ui[k].visible = false
		client.failed.connect(func(r): ui["status"].text = "disconnected: " + r)
		client.acked.connect(func(m): ui["status"].text = "intent %s: %s" % [m["id"], m["status"]])
		client.connect_to(args.get("url", "ws://127.0.0.1:8765/"))
		ui["status"].text = "connecting"
	if args.has("screenshot"):  # live: wait for the welcome and a few ticks first
		shot = {"path": args["screenshot"], "frames": 5 if args.has("log") else 180}


func _args() -> Dictionary:
	var out := {}
	var a := OS.get_cmdline_user_args()
	for i in a.size() - 1:
		if a[i].begins_with("--"):
			out[a[i].substr(2)] = a[i + 1]
	return out


func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var bar := HBoxContainer.new()
	bar.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_WIDE)
	bar.offset_top = -40
	layer.add_child(bar)
	var play := Button.new()
	play.text = "play/pause"
	play.pressed.connect(_toggle_play)
	bar.add_child(play)
	var slider := HSlider.new()
	slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	slider.step = 1
	slider.value_changed.connect(_scrub)
	bar.add_child(slider)
	var clock := Label.new()
	bar.add_child(clock)
	var fam := SpinBox.new()
	fam.min_value = 0
	bar.add_child(fam)
	var fire := Button.new()
	fire.text = "fire"
	fire.pressed.connect(func(): if client: ui["status"].text = "sent " + client.fire(int(fam.value)))
	bar.add_child(fire)
	var status := Label.new()
	status.position = Vector2(8, 8)
	layer.add_child(status)
	var scroll := ScrollContainer.new()  # inhabitant mode: one button per offered action (and person)
	scroll.set_anchors_and_offsets_preset(Control.PRESET_RIGHT_WIDE)
	scroll.offset_left = -210
	scroll.offset_top = 8
	scroll.offset_bottom = -48
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	layer.add_child(scroll)
	var menu := VBoxContainer.new()
	menu.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(menu)
	ui = {"slider": slider, "clock": clock, "family": fam, "status": status, "menu": menu, "fire": fire, "play": play}


var _menu_key := ""


func _rebuild_menu() -> void:
	## The menu is the server's offer for this view, rebuilt only when it changes.
	var items := LiveClient.menu(client.view)
	var key := JSON.stringify(items)
	if key == _menu_key:
		return
	_menu_key = key
	var box: VBoxContainer = ui["menu"]
	for c in box.get_children():
		c.queue_free()
	for it in items:
		var b := Button.new()
		b.text = it["label"]
		b.pressed.connect(func(): ui["status"].text = "sent %s" % client.act(it["action"], it["target"]))
		box.add_child(b)


func _toggle_play() -> void:
	if client:
		client.control("run" if client.paused else "pause")
	else:
		playing = not playing


func _scrub(v: float) -> void:
	if client:
		follow = int(v) >= client.view_tick
		scrub_tick = int(v)
	else:
		replay_tick = v


func _process(delta: float) -> void:
	if client:
		client.poll()
	elif playing:
		replay_tick = minf(replay_tick + delta * 6.0, model.last_tick + 1)
	var newest := client.view_tick if client else model.last_tick + 1
	var s: HSlider = ui["slider"]
	s.max_value = newest
	s.set_value_no_signal(_tick_shown() if not (client and follow) else newest)
	ui["family"].max_value = maxi(0, model.families.size() - 1)
	var t := int(_tick_shown())
	ui["clock"].text = "day %d %02d:%02d (tick %d)" % [t / model.ticks_per_day, (t % model.ticks_per_day) / 6,
			(t % 6) * 10, t]
	queue_redraw()
	if not shot.is_empty():
		shot["frames"] -= 1
		if shot["frames"] == 0:
			get_viewport().get_texture().get_image().save_png(shot["path"])
			get_tree().quit()


func _tick_shown() -> float:
	if client:
		return client.display_tick() if follow else float(scrub_tick)
	return replay_tick


func _draw() -> void:
	if client and client.scope == "inhabitant":
		_draw_inhabitant()
		return
	if model == null or model.places.is_empty():
		return
	var size := get_viewport_rect().size
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	for p in model.places.values():
		var r = p["rect"]
		lo = lo.min(Vector2(r[0], r[1]))
		hi = hi.max(Vector2(r[0] + r[2], r[1] + r[3]))
	var k := minf((size.x - 40) / (hi.x - lo.x), (size.y - 100) / (hi.y - lo.y))
	var origin := (size - Vector2(0, 60) - (hi - lo) * k) / 2.0  # centred above the bar
	var to_screen := func(v: Vector2) -> Vector2: return (v - lo) * k + origin
	var font := ThemeDB.fallback_font
	for name in model.places:
		var r = model.places[name]["rect"]
		var rect := Rect2(to_screen.call(Vector2(r[0], r[1])), Vector2(r[2], r[3]) * k)
		draw_rect(rect, Color(0.25, 0.3, 0.25) if name.begins_with("home") else Color(0.3, 0.35, 0.45))
		draw_string(font, rect.position + Vector2(2, 12), name, HORIZONTAL_ALIGNMENT_LEFT, -1, 10)
	var t := _tick_shown()
	var at := {}  # several villagers at one anchor: spread them out so they don't stack
	for a in model.agents:
		var aid := int(a["id"])
		var p := model.display_position(aid, t)
		var n: int = at.get(p, 0)
		at[p] = n + 1
		var spot: Vector2 = to_screen.call(p) + Vector2((n % 5) * 5 - 10, (n / 5) * 5 - 5)
		draw_circle(spot, 3.0, COLORS.get(model.state_at(aid, int(t))[3], Color.WHITE))


func _draw_inhabitant() -> void:
	## Only what the server sent this villager: the map, itself and the people at its place.
	if client.places.is_empty() or client.view.is_empty():
		return
	var size := get_viewport_rect().size
	var lo := Vector2(INF, INF)
	var hi := Vector2(-INF, -INF)
	for p in client.places.values():
		var r = p["rect"]
		lo = lo.min(Vector2(r[0], r[1]))
		hi = hi.max(Vector2(r[0] + r[2], r[1] + r[3]))
	var k := minf((size.x - 240) / (hi.x - lo.x), (size.y - 100) / (hi.y - lo.y))
	var origin := Vector2(20, 20)
	var font := ThemeDB.fallback_font
	for name in client.places:
		var r = client.places[name]["rect"]
		var rect := Rect2((Vector2(r[0], r[1]) - lo) * k + origin, Vector2(r[2], r[3]) * k)
		draw_rect(rect, Color(0.25, 0.3, 0.25) if name.begins_with("home") else Color(0.3, 0.35, 0.45))
		draw_string(font, rect.position + Vector2(2, 12), name, HORIZONTAL_ALIGNMENT_LEFT, -1, 10)
	var v: Dictionary = client.view
	var at := (Vector2(v["pos"][0], v["pos"][1]) - lo) * k + origin
	for i in v["present"].size():
		draw_circle(at + Vector2((i % 5) * 5 - 10, (i / 5) * 5 + 6), 3.0, Color(0.7, 0.7, 0.7))
	draw_circle(at, 5.0, COLORS.get(v["activity"], Color.WHITE))
	var occ = v.get("occasion")
	var line := "%s  hunger %d  food %d  coin %d" % [client.me.get("name", ""), v["hunger"], v["food"], v["coin"]]
	if occ != null:
		line += "  (" + str(occ["text"]) + ")"
	draw_string(font, Vector2(20, size.y - 70), line, HORIZONTAL_ALIGNMENT_LEFT, -1, 14)
