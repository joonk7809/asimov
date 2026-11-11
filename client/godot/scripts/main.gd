# The client: draws the map from genesis, villagers from events, a timeline with a scrubber, and sends fire.
#   godot --path client/godot -- --url ws://127.0.0.1:8765/     live, against python -m asiimov.live
#   ... --url URL --inhabit 3 [--token T]                         live as villager 3: what it sees, in words, and
#                                                                 its action menu
#   godot --path client/godot -- --log runs/demo/events.jsonl   a recorded log
#   ... --log LOG --at TICK --screenshot out.png                  save one frame and quit (needs a display: xvfb-run;
#                                                                 live, it waits about 3 s first)
# Interpolation between ticks is for drawing only. Nothing drawn here is sent to the engine.
extends Node2D

const WorldModel := preload("res://scripts/world_model.gd")
const LiveClient := preload("res://scripts/live_client.gd")
const Words := preload("res://scripts/inhabit_ui.gd")
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
var names := {}             # inhabitant: id -> name, from what the server sent (welcome, present, kin)
var doing := {}             # inhabitant: what each villager here was last seen doing
var pending := {}           # inhabitant: the act sent and not yet resolved {id, label, tick}


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
		client.token = args.get("token", "")
		if args.has("inhabit"):
			client.scope = "inhabitant"
			client.agent = int(args["inhabit"])
			client.ticked.connect(_on_inhabit_tick)
			client.welcomed.connect(_on_inhabit_welcome)
			_build_inhabit_ui()
			for k in ["family", "fire", "play", "menu_scroll", "slider", "clock"]:  # divine verbs and controls aren't an inhabitant's
				ui[k].visible = false
		client.failed.connect(func(r): ui["status"].text = "disconnected: " + r)
		client.acked.connect(_on_ack)
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
	ui = {"slider": slider, "clock": clock, "family": fam, "status": status, "menu": menu, "fire": fire, "play": play,
			"menu_scroll": scroll, "layer": layer}


func _panel(parent: Control, title: String, expand := false) -> RichTextLabel:
	var head := Label.new()
	head.text = title
	head.add_theme_color_override("font_color", Color(0.65, 0.75, 0.9))
	parent.add_child(head)
	var body := RichTextLabel.new()
	body.bbcode_enabled = true
	body.fit_content = not expand
	body.scroll_active = expand
	if expand:
		body.size_flags_vertical = Control.SIZE_EXPAND_FILL
		body.scroll_following = true
	parent.add_child(body)
	return body


func _build_inhabit_ui() -> void:
	## Inhabitant mode: you, your plan, who's here and your buttons on the right; what happened below the map.
	var side := VBoxContainer.new()
	side.set_anchors_and_offsets_preset(Control.PRESET_RIGHT_WIDE)
	side.offset_left = -400
	side.offset_right = -8
	side.offset_top = 8
	side.offset_bottom = -12
	ui["layer"].add_child(side)
	ui["you"] = _panel(side, "You")
	ui["plan"] = _panel(side, "Today's plan")
	ui["here"] = _panel(side, "Here with you")
	var head := Label.new()
	head.text = "What you can do"
	head.add_theme_color_override("font_color", Color(0.65, 0.75, 0.9))
	side.add_child(head)
	var asked := RichTextLabel.new()
	asked.bbcode_enabled = true
	asked.fit_content = true
	side.add_child(asked)
	ui["asked"] = asked
	var wait := Label.new()
	wait.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	side.add_child(wait)
	ui["wait"] = wait
	var scroll := ScrollContainer.new()
	scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	side.add_child(scroll)
	var menu := VBoxContainer.new()
	menu.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	scroll.add_child(menu)
	ui["menu"] = menu
	var feed_box := VBoxContainer.new()
	feed_box.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	feed_box.offset_left = 340  # the map takes the left column
	feed_box.offset_right = -416
	feed_box.offset_top = 8
	feed_box.offset_bottom = -12
	ui["layer"].add_child(feed_box)
	ui["feed"] = _panel(feed_box, "What's happening", true)


var _menu_key := ""
var _caught_up := false


func _learn_names(v: Dictionary) -> void:
	for p in v.get("present", []) + v.get("kin", []):
		names[int(p["id"])] = p["name"]


func _on_inhabit_welcome(m: Dictionary) -> void:
	ui["status"].text = ""
	names[int(m["world"]["me"]["id"])] = m["world"]["me"]["name"]
	for f in m["world"]["family"] + m["world"].get("people", []):
		names[int(f["id"])] = f["name"]
	_learn_names(m["view"])
	_refresh_inhabit()
	get_tree().create_timer(1.0).timeout.connect(_catch_up)  # the backlog follows the welcome in chunks


func _catch_up() -> void:
	## Today's earlier events from the welcome's backlog, once, so a paused or slow world isn't a blank feed.
	if _caught_up or client.view.is_empty():
		return
	_caught_up = true
	var v: Dictionary = client.view
	var day0 := int(v["tick"]) - int(v["tick"]) % client.ticks_per_day()
	_feed(client.seen_events.filter(func(e): return int(e["tick"]) >= day0))


func _feed(evs: Array) -> void:
	for e in evs:
		var line := Words.describe(e, client.me, names)
		if line != "":
			ui["feed"].append_text("[color=#8a93a6]%s[/color]  %s\n" % [Words.clock(int(e["tick"]), client.ticks_per_day()).split(", ")[1], line])


func _on_inhabit_tick(m: Dictionary) -> void:
	var v: Dictionary = m["view"]
	_learn_names(v)
	var here := {}
	for p in v["present"]:
		here[int(p["id"])] = true
	for id in doing.keys():
		if not here.has(id):
			doing.erase(id)
	var evs: Array = m["events"]
	if not _caught_up:  # the first tick before the catch-up timer: today's backlog, this tick's included
		_caught_up = true
		var day0 := int(v["tick"]) - int(v["tick"]) % client.ticks_per_day()
		evs = client.seen_events.filter(func(e): return int(e["tick"]) >= day0)
	_feed(evs)
	for e in evs:
		if e.get("actor") != null and here.has(int(e["actor"])):
			match e["kind"]:
				"activity":
					doing[int(e["actor"])] = "came to " + str(Words.VERB.get(str(e["data"].get("activity")), e["data"].get("activity")))
				"talk":
					doing[int(e["actor"])] = "talking with " + Words._who(e["data"].get("with"), int(client.me["id"]), names)
				"eat":
					doing[int(e["actor"])] = "eating"
				"sleep":
					doing[int(e["actor"])] = "asleep"
	if not pending.is_empty() and int(v["tick"]) > int(pending["tick"]):
		pending = {}  # its receipt is in this tick's lines
	_refresh_inhabit()


func _on_ack(m: Dictionary) -> void:
	if client.scope != "inhabitant":
		ui["status"].text = "intent %s: %s" % [m["id"], m["status"]]
	elif m["status"] == "rejected" and pending.get("id") == m["id"]:
		ui["feed"].append_text("[color=#e07070]Not sent: %s (%s).[/color]\n" % [pending["label"], m.get("reason", "")])
		pending = {}
		_refresh_inhabit()


func _refresh_inhabit() -> void:
	var v: Dictionary = client.view
	if v.is_empty():
		return
	var per_day := client.ticks_per_day()
	ui["you"].text = Words.status(v, client.me, per_day)
	ui["plan"].text = Words.plan(v, per_day, names)
	ui["here"].text = Words.here(v, doing)
	ui["asked"].text = Words.asks(v, names, 1.0 / maxf(client.tps, 0.001))  # what waits on your answer, if anything
	_rebuild_menu()


func _rebuild_menu() -> void:
	## The menu is the server's offer for this view, rebuilt only when it or the pending act changes.
	var v: Dictionary = client.view
	var items := LiveClient.menu(v)
	var cooling := int(v["tick"]) < int(v["cooldown_until"])
	var key := JSON.stringify([items, pending.get("id"), cooling, v.get("asks", [])])
	if key == _menu_key:
		return
	_menu_key = key
	var box: VBoxContainer = ui["menu"]
	for c in box.get_children():
		c.queue_free()
	if items.is_empty():
		var l := Label.new()
		l.text = "Nothing to do while asleep." if v["asleep"] else "Nothing to choose right now."
		box.add_child(l)
	var present := {}
	for p in v["present"]:
		present[int(p["id"])] = p["kin"]
	var rows := {}  # one row of short buttons per person; the actions that name nobody come first
	var askers := {}
	for q in v.get("asks", []):
		askers[int(q["from"])] = true
	items = items.filter(func(x): return askers.has(int(x["target"]))) + \
			items.filter(func(x): return not askers.has(int(x["target"])))  # who's waiting on you comes first
	for it in items:
		var t := int(it["target"])
		if not Words.shown(it, present):
			continue
		var b := Button.new()
		b.text = Words.action_label(it["action"], names.get(t, "#%d" % t)) if t < 0 else Words.SHORT.get(it["action"], it["action"])
		b.disabled = not pending.is_empty()
		if cooling and it["action"] in Words.RATED:
			b.disabled = true
			b.tooltip_text = "Too soon after your last ask or purchase."
		var label := Words.action_label(it["action"], names.get(t, "#%d" % t))
		b.pressed.connect(func(): pending = {"id": client.act(it["action"], it["target"]), "label": label,
				"tick": int(client.view["tick"])}; _refresh_inhabit())
		if t < 0:
			b.alignment = HORIZONTAL_ALIGNMENT_LEFT
			box.add_child(b)
			continue
		if it["action"] in ["agree", "decline"]:  # the answer to an ask: its own row, first, in the ask's colour
			if not rows.has("answer%d" % t):
				var who := Label.new()
				who.text = "Answer %s:" % names.get(t, "#%d" % t)
				who.add_theme_color_override("font_color", Color(0.94, 0.75, 0.38))
				rows["answer%d" % t] = HFlowContainer.new()
				rows["answer%d" % t].add_child(who)
				box.add_child(rows["answer%d" % t])
				box.move_child(rows["answer%d" % t], 0)
			b.add_theme_color_override("font_color", Color(0.94, 0.75, 0.38))
			rows["answer%d" % t].add_child(b)
			continue
		if not rows.has(t):
			var who := Label.new()
			who.text = names.get(t, "#%d" % t) + ":"
			rows[t] = HFlowContainer.new()
			rows[t].add_child(who)
			box.add_child(rows[t])
		rows[t].add_child(b)


func _wait_text() -> String:
	var left := 0.0 if client.paused else maxf(0.0, 1.0 / client.tps - (Time.get_ticks_msec() - client.tick_wall) / 1000.0)
	var next := "the world is paused" if client.paused else "next tick in %ds" % ceili(left)
	if not pending.is_empty():
		return "Sent: %s. It happens on the next tick (%s)." % [pending["label"], next]
	return "Each tick is 10 minutes of village time; %s. One action per tick." % next


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
	if client and client.scope == "inhabitant" and ui.has("wait"):
		ui["wait"].text = _wait_text()
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
	var k := minf(300.0 / (hi.x - lo.x), (size.y - 40) / (hi.y - lo.y))  # the feed and panels take the rest
	var origin := Vector2(20, 20)
	var font := ThemeDB.fallback_font
	for name in client.places:
		var r = client.places[name]["rect"]
		var rect := Rect2((Vector2(r[0], r[1]) - lo) * k + origin, Vector2(r[2], r[3]) * k)
		draw_rect(rect, Color(0.25, 0.3, 0.25) if name.begins_with("home") else Color(0.3, 0.35, 0.45))
		var mine: bool = name == str(client.me.get("home", ""))
		if mine or name == str(client.view.get("place")):
			draw_rect(rect, Color(0.95, 0.85, 0.4), false, 2.0)
		draw_string(font, rect.position + Vector2(2, 12), "your home" if mine else name.replace("_", " "),
				HORIZONTAL_ALIGNMENT_LEFT, -1, 10)
	var v: Dictionary = client.view
	var at := (Vector2(v["pos"][0], v["pos"][1]) - lo) * k + origin
	for i in v["present"].size():
		draw_circle(at + Vector2((i % 5) * 5 - 10, (i / 5) * 5 + 6), 3.0, Color(0.7, 0.7, 0.7))
	draw_circle(at, 5.0, COLORS.get(v["activity"], Color.WHITE))
