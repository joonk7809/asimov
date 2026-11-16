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
const Plan := preload("res://scripts/plan_queue.gd")
const Art := preload("res://scripts/art.gd")
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
var roles := {}             # inhabitant: id -> role, from the welcome's people and the present list
var doing := {}             # inhabitant: what each villager here was last seen doing
var pending := {}           # inhabitant: the act sent and not yet resolved {id, label, tick, event?, step?}
var plan_q := []            # inhabitant: the player's own plan (readable-world C2), head first
var plan_state := ""        # the head step's state: "" (not sent), "wait", "hold" or "failed: WHY"
var plan_act := -1          # the act event of the head step, once acked
var plan_receipts := []     # receipts (and a seek's meet) so far that cite plan_act
var plan_sent := 0          # the view tick the head step was sent at
var map_rects := {}         # inhabitant: place name -> its rect on screen, for clicks
const MAP_W := 420          # inhabitant: the map's column; the feed and panels take the rest
var zoom := 1.8             # inhabitant: map zoom (V4), wheel over the map; past 1 the map follows you
var bubbles := []           # inhabitant: words shown over their speaker (V3) {who, text, mine, born}
var drawn_at := {}          # inhabitant: id -> where that villager was drawn this frame (for bubbles)
const BUBBLE_S := 6.0       # a bubble shows this long, then fades; the line stays in "Said near you"
var say_once := ""          # --say TEXT (dev, screenshots): say it once to the first villager here, awake


func _ready() -> void:
	texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST  # pixel art stays crisp when scaled (V-pass)
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
			client.early.connect(_on_early)
			client.welcomed.connect(_on_inhabit_welcome)
			client.heard.connect(_on_heard)
			_build_inhabit_ui()
			for k in ["family", "fire", "play", "menu_scroll", "slider", "clock"]:  # divine verbs and controls aren't an inhabitant's
				ui[k].visible = false
		client.failed.connect(func(r): ui["status"].text = "disconnected: " + r)
		client.acked.connect(_on_ack)
		client.connect_to(args.get("url", "ws://127.0.0.1:8765/"))
		ui["status"].text = "connecting"
	say_once = str(args.get("say", ""))
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
	ui["myplan"] = _panel(side, "Your plan")
	var ctl := HFlowContainer.new()
	side.add_child(ctl)
	var ahead := CheckButton.new()
	ahead.text = "Plan ahead"
	ahead.tooltip_text = "On: actions and places you press line up in your plan instead of happening now."
	ahead.toggled.connect(func(on): _menu_key = ""; _refresh_inhabit())
	ctl.add_child(ahead)
	ui["ahead"] = ahead
	for k in ["Next", "Retry", "Clear"]:
		var b := Button.new()
		b.text = k
		b.pressed.connect(_plan_button.bind(k))
		ctl.add_child(b)
		ui["plan_" + k.to_lower()] = b
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
	# say-1: talk to someone here in your own words; their mind answers on the next tick.
	var talk_head := Label.new()
	talk_head.text = "Talk"
	talk_head.add_theme_color_override("font_color", Color(0.65, 0.75, 0.9))
	side.add_child(talk_head)
	var talk := HBoxContainer.new()
	side.add_child(talk)
	var who := OptionButton.new()
	who.custom_minimum_size = Vector2(120, 0)
	who.clip_text = true
	talk.add_child(who)
	ui["talk_who"] = who
	var line := LineEdit.new()
	line.max_length = 140
	line.placeholder_text = "Say something..."
	line.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	line.text_submitted.connect(func(_t): _say())
	talk.add_child(line)
	ui["talk_line"] = line
	var say := Button.new()
	say.text = "Say"
	say.pressed.connect(_say)
	talk.add_child(say)
	ui["talk_say"] = say
	var feed_box := VBoxContainer.new()
	feed_box.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	feed_box.offset_left = MAP_W  # the map takes the left column
	feed_box.offset_right = -416
	feed_box.offset_top = 8
	feed_box.offset_bottom = -12
	ui["layer"].add_child(feed_box)
	ui["feed"] = _panel(feed_box, "What's happening", true)
	ui["feed"].size_flags_stretch_ratio = 1.6
	ui["said"] = _panel(feed_box, "Said near you", true)  # readable-world C3: talk you can hear, in words
	ui["said"].append_text("[color=#8a93a6]Conversations near you, and what people say to you, show up here.[/color]\n")


var _menu_key := ""
var _caught_up := false


func _learn_names(v: Dictionary) -> void:
	for p in v.get("present", []) + v.get("kin", []):
		names[int(p["id"])] = p["name"]
		if p.has("role"):
			roles[int(p["id"])] = p["role"]


func _on_inhabit_welcome(m: Dictionary) -> void:
	ui["status"].text = ""
	names[int(m["world"]["me"]["id"])] = m["world"]["me"]["name"]
	for f in m["world"]["family"] + m["world"].get("people", []):
		names[int(f["id"])] = f["name"]
		roles[int(f["id"])] = f.get("role", "")
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
			var at := "[color=#8a93a6]%s[/color]  %s\n" % [Words.clock(int(e["tick"]), client.ticks_per_day()).split(", ")[1], line]
			ui["feed"].append_text(at)
			if e["kind"] in ["said", "reply"]:  # say-1: spoken words also go where talk is read
				ui["said"].append_text(at)
				var words := str(e["data"].get("text" if e["kind"] == "said" else "says", ""))
				if words != "" and _caught_up:
					_bubble(int(e["actor"]), words, e["data"].get("with") != null and int(e["data"]["with"]) == int(client.me["id"]), 0.0)


func _on_heard(m: Dictionary) -> void:
	var line := Words.heard(m, client.ticks_per_day())
	if line != "":
		ui["said"].append_text(line + "\n")
	var lines: Array = m.get("lines", [])
	for i in lines.size():  # one after another, as a conversation is spoken
		_bubble(int(lines[i]["who"]), str(lines[i]["text"]), bool(m.get("to_player", false)), i * 2.0)


func _bubble(who: int, text: String, to_me: bool, delay: float) -> void:
	## V3: words over their speaker, for BUBBLE_S seconds (drawing only; the log keeps every line).
	bubbles = bubbles.filter(func(b): return b["who"] != who or b["born"] > Time.get_ticks_msec() / 1000.0)
	bubbles.append({"who": who, "text": text, "mine": who == int(client.me["id"]), "to_me": to_me,
			"born": Time.get_ticks_msec() / 1000.0 + delay})


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
	_take_receipts(evs)
	if say_once != "" and not v["asleep"]:
		for p in v["present"]:
			if not p.get("asleep", false):
				client.act("say", int(p["id"]), "", say_once)
				say_once = ""
				break
	if not pending.is_empty() and not pending.get("step", false) and int(v["tick"]) > int(pending["tick"]):
		pending = {}  # its receipt is in this tick's lines
	_plan_run()
	_refresh_inhabit()


func _on_early(m: Dictionary) -> void:
	## E4: the player's act applied mid-tick; its receipt is here now, not with the tick.
	if not _caught_up:
		return  # the catch-up on the first tick reads seen_events, these included
	_feed(m["events"])
	_take_receipts(m["events"])
	_plan_run()
	_refresh_inhabit()


func _on_ack(m: Dictionary) -> void:
	if client.scope != "inhabitant":
		ui["status"].text = "intent %s: %s" % [m["id"], m["status"]]
	elif m["status"] == "applied" and pending.get("id") == m["id"]:
		pending["event"] = int(m["event"])
		if pending.get("step", false):
			plan_act = int(m["event"])
			_take_receipts(client.seen_events.filter(func(e): return int(e["tick"]) >= int(pending["tick"])))
	elif m["status"] == "rejected" and pending.get("id") == m["id"]:
		ui["feed"].append_text("[color=#e07070]Not sent: %s (%s).[/color]\n" % [pending["label"], m.get("reason", "")])
		if pending.get("step", false):
			plan_state = "failed: " + str(m.get("reason", "rejected"))
		pending = {}
		_refresh_inhabit()


func _take_receipts(evs: Array) -> void:
	## Receipts for the plan's step in hand; settle it, and send the next step when it's done.
	if plan_act < 0 or plan_q.is_empty():
		return
	for e in evs:
		if e["kind"] in ["receipt", "meet"] and not e["parents"].is_empty() and int(e["parents"][0]) == plan_act \
				and not plan_receipts.any(func(r): return r["id"] == e["id"]):
			plan_receipts.append(e)
	var st := Plan.settle(plan_q[0], plan_receipts, int(client.view.get("tick", 0)) - plan_sent)
	plan_state = st
	if st != "wait" and pending.get("step", false):
		pending = {}
	if st == "done":
		_plan_pop()


func _plan_pop() -> void:
	plan_q.pop_front()
	plan_state = ""
	plan_act = -1
	plan_receipts = []


func _plan_button(k: String) -> void:
	match k:
		"Next":
			if plan_state == "hold":
				_plan_pop()
		"Retry":
			if plan_state.begins_with("failed"):
				plan_state = ""
				plan_act = -1
				plan_receipts = []
		"Clear":
			plan_q = []
			plan_state = ""
			plan_act = -1
			plan_receipts = []
	_plan_run()
	_refresh_inhabit()


func _plan_run() -> void:
	## Sends the head step when nothing else is in flight and it hasn't been sent.
	if plan_q.is_empty() or plan_state != "" or not pending.is_empty() or client.view.is_empty() or client.view["asleep"]:
		return
	var s: Dictionary = plan_q[0]
	pending = {"id": client.act(s["action"], int(s["target"]), str(s["place"])), "label": s["label"],
			"tick": int(client.view["tick"]), "step": true}
	plan_state = "wait"
	plan_sent = int(client.view["tick"])


var _talk_key := ""


func _refresh_talk() -> void:
	## Who you can talk to: the people here and awake (the server checks again when it applies).
	var v: Dictionary = client.view
	var ids := []
	if not v.get("asleep", true):
		for p in v.get("present", []):
			if not p.get("asleep", false):
				ids.append(int(p["id"]))
	var key := JSON.stringify(ids)
	var who: OptionButton = ui["talk_who"]
	if key != _talk_key:
		_talk_key = key
		var keep := who.get_selected_id() if who.item_count > 0 else -1
		who.clear()
		for i in ids:
			who.add_item(names.get(i, "#%d" % i), i)
		var at := who.get_item_index(keep) if keep >= 0 else -1
		if at >= 0:
			who.select(at)
	var none: bool = ids.is_empty()
	ui["talk_line"].editable = not none
	ui["talk_line"].placeholder_text = "Nobody here to talk to." if none else "Say something..."
	ui["talk_say"].disabled = none or not pending.is_empty()


func _say() -> void:
	var who: OptionButton = ui["talk_who"]
	var text := str(ui["talk_line"].text).strip_edges()
	if text == "" or who.item_count == 0 or not pending.is_empty():
		return
	var t := who.get_selected_id()
	pending = {"id": client.act("say", t, "", text), "label": "Say to %s" % names.get(t, "#%d" % t),
			"tick": int(client.view["tick"])}
	ui["talk_line"].text = ""
	_refresh_inhabit()


func _choose(action: String, target: int, place: String, label: String) -> void:
	## A button or a map place: into the plan with "Plan ahead" on, else done now.
	if ui["ahead"].button_pressed:
		if plan_q.size() < Plan.MAX_STEPS:
			plan_q.append(Plan.step(action, target, place, label))
		_plan_run()
	elif pending.is_empty():
		pending = {"id": client.act(action, target, place), "label": label, "tick": int(client.view["tick"])}
	_refresh_inhabit()


func _unhandled_input(ev: InputEvent) -> void:
	if client == null or client.scope != "inhabitant" or not (ev is InputEventMouseButton) or not ev.pressed:
		return
	if ev.position.x >= MAP_W:
		return
	if ev.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN]:  # V4: zoom the map
		zoom = clampf(zoom * (1.15 if ev.button_index == MOUSE_BUTTON_WHEEL_UP else 1.0 / 1.15), 1.0, 4.0)
		get_viewport().set_input_as_handled()
		return
	if ev.button_index != MOUSE_BUTTON_LEFT:
		return
	for name in map_rects:
		if map_rects[name].has_point(ev.position):
			_choose("go_to", -1, name, "Go to " + Words.place(name, client.me))
			get_viewport().set_input_as_handled()
			return


func _refresh_inhabit() -> void:
	var v: Dictionary = client.view
	if v.is_empty():
		return
	var per_day := client.ticks_per_day()
	ui["you"].text = Words.status(v, client.me, per_day)
	ui["plan"].text = Words.plan(v, per_day, names)
	ui["here"].text = Words.here(v, doing)
	ui["asked"].text = Words.asks(v, names, 1.0 / maxf(client.tps, 0.001))  # what waits on your answer, if anything
	ui["myplan"].text = Plan.lines(plan_q, plan_state, Words.WHY)
	ui["plan_next"].disabled = plan_state != "hold"
	ui["plan_retry"].disabled = not plan_state.begins_with("failed")
	ui["plan_clear"].disabled = plan_q.is_empty()
	_rebuild_menu()
	_refresh_talk()


func _rebuild_menu() -> void:
	## The menu is the server's offer for this view, rebuilt only when it or the pending act changes.
	var v: Dictionary = client.view
	var items := LiveClient.menu(v)
	var cooling := int(v["tick"]) < int(v["cooldown_until"])
	var ahead: bool = ui["ahead"].button_pressed
	var key := JSON.stringify([items, pending.get("id"), cooling, v.get("asks", []), ahead])
	if key == _menu_key:
		return
	_menu_key = key
	var box: VBoxContainer = ui["menu"]
	for c in box.get_children():
		c.queue_free()
	if ahead:  # planning: the acts that name nobody can be lined up from anywhere; the engine checks each when it comes up
		var row := HFlowContainer.new()
		var lab := Label.new()
		lab.text = "Plan from anywhere:"
		row.add_child(lab)
		for act in Plan.ANYWHERE:
			var pb := Button.new()
			pb.text = Words.action_label(act, "")
			pb.pressed.connect(_choose.bind(act, -1, "", Words.action_label(act, "")))
			row.add_child(pb)
		box.add_child(row)
		var hint := Label.new()
		hint.text = "Click a place on the map to walk there."
		box.add_child(hint)
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
		b.disabled = not pending.is_empty() and not ahead
		if cooling and it["action"] in Words.RATED and not ahead:
			b.disabled = true
			b.tooltip_text = "Too soon after your last ask or purchase."
		var label := Words.action_label(it["action"], names.get(t, "#%d" % t))
		b.pressed.connect(_choose.bind(it["action"], int(it["target"]), "", label))
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
			var tags := [roles.get(t, "")] if roles.get(t, "") != "" else []
			if present.get(t, false):
				tags.append("family")
			who.text = names.get(t, "#%d" % t) + (" (%s)" % ", ".join(tags) if not tags.is_empty() else "") + ":"
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
	_draw_ground(model.places, to_screen, lo, hi, k)
	for name in model.places:
		var r = model.places[name]["rect"]
		var rect := Rect2(to_screen.call(Vector2(r[0], r[1])), Vector2(r[2], r[3]) * k)
		if not Art.place(self, name, rect, _tile_px(k)):
			draw_rect(rect, Color(0.25, 0.3, 0.25) if name.begins_with("home") else Color(0.3, 0.35, 0.45))
		if not name.begins_with("home"):
			Art.label(self, rect.position + Vector2(2, 12), name, 10)
	var t := _tick_shown()
	var at_anchor := {}  # a place's anchor -> its rect on screen: villagers there spread over the place
	for name in model.places:
		var r = model.places[name]["rect"]
		var an = model.places[name]["anchor"]
		at_anchor[Vector2(an[0], an[1])] = Rect2(to_screen.call(Vector2(r[0], r[1])), Vector2(r[2], r[3]) * k)
	var vpx := maxf(k * 1.6, 9.0)
	for a in model.agents:
		var aid := int(a["id"])
		var p := model.display_position(aid, t)
		var spot: Vector2 = Art.spot_in(at_anchor[p], aid, vpx) if at_anchor.has(p) else to_screen.call(p)
		var st = model.state_at(aid, int(t))
		Art.villager(self, spot, str(a.get("role", "")), vpx, false, st[3] == "sleep")


func _tile_px(k: float) -> float:
	return maxf(k * 2.5, 8.0)  # one 16 px ground tile covers 2.5 m


func _draw_ground(places: Dictionary, to_screen: Callable, lo: Vector2, hi: Vector2, k: float) -> void:
	## Grass under the whole map, and dirt paths from the plaza to every place (drawing only).
	var pad := Vector2(4, 4)
	Art.ground(self, Rect2(to_screen.call(lo - pad), (hi - lo + pad * 2) * k), _tile_px(k))
	if not places.has("plaza"):
		return
	var hub: Vector2 = to_screen.call(Vector2(places["plaza"]["anchor"][0], places["plaza"]["anchor"][1]))
	for name in places:
		if name != "plaza" and not name.begins_with("home"):  # homes sit on the grass: their paths were a web
			var a = places[name]["anchor"]
			Art.path(self, hub, to_screen.call(Vector2(a[0], a[1])), maxf(k * 1.2, 3.0))


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
	var k0 := minf((MAP_W - 40.0) / (hi.x - lo.x), (size.y - 60) / (hi.y - lo.y))  # the whole map fits at zoom 1
	var k := k0 * zoom
	var origin := Vector2(20, 20)
	if zoom > 1.0:  # V4: the map follows you, centred in its column
		var me_m := Vector2(client.view["pos"][0], client.view["pos"][1])
		origin = Vector2(MAP_W / 2.0, (size.y - 30) / 2.0) - (me_m - lo) * k
	var font := ThemeDB.fallback_font
	var kin_homes := {}
	for m in client.view.get("kin", []):
		kin_homes[str(m["home"])] = true
	var going := ""  # where the plan's walk, or the walk under way, is headed
	if not plan_q.is_empty() and plan_q[0]["action"] == "go_to":
		going = str(plan_q[0]["place"])
	elif client.view.get("route") != null:
		going = str(client.view["route"]["to"])
	map_rects = {}
	var to_screen := func(p: Vector2) -> Vector2: return (p - lo) * k + origin
	_draw_ground(client.places, to_screen, lo, hi, k)
	for name in client.places:
		var r = client.places[name]["rect"]
		var rect := Rect2((Vector2(r[0], r[1]) - lo) * k + origin, Vector2(r[2], r[3]) * k)
		map_rects[name] = rect.grow(2)
		if not Art.place(self, name, rect, _tile_px(k)):
			draw_rect(rect, Color(0.25, 0.3, 0.25) if name.begins_with("home") else Color(0.3, 0.35, 0.45))
		if name == going:
			draw_rect(rect, Color(0.5, 0.85, 0.6), false, 2.0)
		var mine: bool = name == str(client.me.get("home", ""))
		if mine or name == str(client.view.get("place")):
			draw_rect(rect, Color(0.95, 0.85, 0.4), false, 2.0)
		# Only the homes that are yours to know get a label (yours, your family's): at 100 villagers the
		# others' labels overlap into noise. Public places keep their names.
		var label: String = "you" if mine else ("family" if kin_homes.has(name) else ("" if name.begins_with("home") else name.replace("_", " ")))
		if label != "":
			Art.label(self, rect.position + Vector2(3, 14), label, clampi(int(9 + zoom * 2), 11, 16), Art.LABEL,
					maxf(rect.size.x - 2, 8.0))
	var v: Dictionary = client.view
	var at := (Vector2(v["pos"][0], v["pos"][1]) - lo) * k + origin
	var px := maxf(k * 2.2, 12.0)
	var here: Rect2 = map_rects.get(str(v.get("place")), Rect2(at - Vector2(px, px), Vector2(px, px) * 2)).grow(-2)
	drawn_at = {}
	for p in v["present"]:
		var spot := Art.spot_in(here, int(p["id"]), px * 0.85)
		drawn_at[int(p["id"])] = spot - Vector2(0, px * 0.85)
		Art.villager(self, spot, str(p.get("role", "")), px * 0.85, false, bool(p.get("asleep", false)))
	Art.villager(self, at, str(client.me.get("role", "")), px, true, bool(v["asleep"]))
	drawn_at[int(client.me["id"])] = at - Vector2(0, px * 1.1)
	_draw_bubbles()
	# The map's column ends at MAP_W: cover what a zoomed map draws beyond it, under the panels, and its foot.
	draw_rect(Rect2(MAP_W, 0, size.x - MAP_W, size.y), RenderingServer.get_default_clear_color())
	draw_rect(Rect2(0, size.y - 30, MAP_W, 30), RenderingServer.get_default_clear_color())
	Art.label(self, Vector2(12, size.y - 10), "Click a place to walk there. Scroll the map to zoom.", 13, Color(0.75, 0.82, 0.95))


func _draw_bubbles() -> void:
	## V3: each line over its speaker for BUBBLE_S seconds, fading out; a line to you is warm yellow.
	var now := Time.get_ticks_msec() / 1000.0
	bubbles = bubbles.filter(func(b): return now - b["born"] < BUBBLE_S)
	var font := ThemeDB.fallback_font
	var fs := 13
	# At most 4 at once, so a busy plaza stays readable: lines to you first, then yours, then the newest.
	var shown := bubbles.filter(func(b): return now >= b["born"] and drawn_at.has(b["who"]))
	shown.sort_custom(func(x, y): return [int(x["to_me"]), int(x["mine"]), x["born"]] > [int(y["to_me"]), int(y["mine"]), y["born"]])
	var placed := []
	for b in shown.slice(0, 4):
		var age: float = now - b["born"]
		var alpha := clampf((BUBBLE_S - age) / 1.0, 0.0, 1.0)
		var text: String = b["text"]
		if text.length() > 90:
			text = text.substr(0, 88) + "\u2026"
		var w := minf(font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, fs).x, 200.0)
		var lines := font.get_multiline_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, w, fs)
		var box := Rect2(drawn_at[b["who"]] - Vector2(w / 2.0 + 6, lines.y + 14), Vector2(w + 12, lines.y + 8))
		box.position.x = clampf(box.position.x, 4.0, MAP_W - box.size.x - 4.0)
		var y0 := box.position.y
		for other in placed:  # stack upward past any bubble already drawn, never over it
			if box.grow(2).intersects(other):
				box.position.y = other.position.y - box.size.y - 4
		placed.append(box)
		var stacked := box.position.y != y0  # a stacked bubble's tail would cross the ones below it
		var fill := Color(1.0, 0.93, 0.62, 0.95 * alpha) if b["to_me"] else (Color(0.86, 0.93, 1.0, 0.95 * alpha) if b["mine"] else Color(1, 1, 1, 0.95 * alpha))
		var tip: Vector2 = drawn_at[b["who"]] - Vector2(0, 4)
		if not stacked:
			draw_colored_polygon(PackedVector2Array([Vector2(tip.x - 5, box.end.y - 1), Vector2(tip.x + 5, box.end.y - 1), tip]), fill)
		draw_style_box(_bubble_box(fill, alpha), box)
		draw_multiline_string(font, box.position + Vector2(6, fs + 2), text, HORIZONTAL_ALIGNMENT_LEFT, w, fs, -1,
				Color(0.12, 0.1, 0.14, alpha))


func _bubble_box(fill: Color, alpha: float) -> StyleBoxFlat:
	var sb := StyleBoxFlat.new()
	sb.bg_color = fill
	sb.set_corner_radius_all(6)
	sb.border_color = Color(0.15, 0.12, 0.18, 0.8 * alpha)
	sb.set_border_width_all(1)
	return sb
