# Headless, no server:  godot --headless --path client/godot --script res://tests/test_talkbox.gd
# The Talk box's rules after aipt3 (Steering 2026-11-20 09:07Z): the chip hint is quiet in a typed world; a pick
# stays the pick and a line being written keeps its listener; a line that can't be said, or wasn't heard, is kept
# and the log says so; ctrl+a selects the line whole, so typing replaces it; and an answer from someone the engine
# ruled out of giving is shown with the engine's reason.
extends SceneTree

const Words := preload("res://scripts/inhabit_ui.gd")
const TalkBox := preload("res://scripts/talk_box.gd")

const NAMES := {3: "Ari Vota", 4: "Isba Isneor", 7: "Riri Deka"}
const ROLES := {3: "farmer", 4: "weaver", 7: "elder"}

var bad := []


func check(ok: bool, what: String) -> void:
	if not ok:
		bad.append(what)


func _init() -> void:
	# 1. The chip hint: shown for an ask with no chip, and never in a typed world.
	var ask := "Could you spare a little food?"
	check(Words.ask_hint(ask, 0) == Words.HINT_ASK_TEXT and Words.ask_hint(ask, 0, false) == Words.HINT_ASK_TEXT,
			"no chip hint in a world where only a chip asks")
	check(Words.ask_hint(ask, 0, true) == "" and Words.ask_hint("Can I buy a meal? I have coin.", 0, true) == "",
			"a chip hint in a typed world")
	check(Words.reads_like_ask(ask) and Words.reads_like_ask("Even a crust would do, please.")
			and Words.reads_like_ask("Will you sell me a meal for a fair price?")
			and not Words.reads_like_ask("Good morning.") and not Words.reads_like_ask("How was the harvest?"),
			"reads_like_ask")

	# 2. Whom the line goes to.
	var t := TalkBox.new()
	check(t.settle([], false) == -1 and t.note([], false, NAMES, ROLES) == "", "nobody here, nothing typed")
	check(t.settle([4, 7], false) == 4 and t.note([4, 7], false, NAMES, ROLES) == "To Isba Isneor, weaver.",
			"the first one here stands in while nobody is picked and the box is empty")
	check(t.settle([3, 4, 7], false) == 4, "someone arriving ahead on the list moved the line's listener")
	check(t.settle([3, 7], true) == -1 and t.chosen == 4, "a line being written went to someone else when its listener left")
	check(t.note([3, 7], true, NAMES, ROLES) == "Isba Isneor isn't here now. Your line is kept: pick who hears it.",
			"the note for a line whose listener left: " + t.note([3, 7], true, NAMES, ROLES))
	check(t.unsent([3, 7], NAMES, ROLES, false) == "Not said. Isba Isneor isn't here now. Your line is kept: pick who hears it.",
			"Say with the listener gone isn't reported: " + t.unsent([3, 7], NAMES, ROLES, false))
	check(t.settle([3, 4, 7], true) == 4, "the listener came back and the line didn't go back to them")
	check(t.settle([3, 7], false) == 3 and not t.picked, "an empty box with its default gone takes the next default")
	t.pick(7)
	check(t.settle([3, 4, 7], false) == 7 and t.settle([3, 4, 7], true) == 7, "a pick didn't take")
	check(t.settle([3, 4], false) == -1 and t.chosen == 7, "a pick who left was replaced, with the box empty")
	check(t.note([3, 4], false, NAMES, ROLES) == "Riri Deka isn't here now. Pick who to talk to.",
			"the note for a pick who left: " + t.note([3, 4], false, NAMES, ROLES))
	check(t.settle([], true) == -1 and t.note([], true, NAMES, ROLES) == "Riri Deka isn't here now. Your line is kept.",
			"the note with nobody here: " + t.note([], true, NAMES, ROLES))
	check(t.unsent([3, 4], NAMES, ROLES, false).begins_with("Not said. Riri Deka isn't here now."), "Say to a pick who left")
	t.pick(3)
	check(t.settle([3, 4], true) == 3 and t.unsent([3, 4], NAMES, ROLES, false) == "", "a new pick sends")
	check(t.unsent([3, 4], NAMES, ROLES, true).begins_with("Not said yet:"), "Say while another act is on its way says nothing")
	var u := TalkBox.new()  # typed with nobody here, then people arrive: nobody is picked for the line
	check(u.settle([], true) == -1 and u.note([], true, NAMES, ROLES) == "Nobody is here to hear you. Your line is kept.",
			"typed with nobody here: " + u.note([], true, NAMES, ROLES))
	check(u.settle([4, 7], true) == -1 and u.note([4, 7], true, NAMES, ROLES) == "Nobody is picked. Your line is kept: pick who hears it.",
			"a line typed to nobody was given a listener: " + u.note([4, 7], true, NAMES, ROLES))

	# 3. A line that was sent and not heard.
	var back := TalkBox.lost("", "Spare a meal?", "Ari Vota")
	check(back["returned"] and back["box"] == "Spare a meal?" and "back in the Talk box" in back["log"], "a lost line isn't put back")
	var kept := TalkBox.lost("A newer line", "Spare a [meal]?", "Ari Vota")
	check(not kept["returned"] and kept["box"] == "A newer line" and "Spare a [lb]meal]?" in kept["log"]
			and "Ari Vota" in kept["log"], "a lost line is dropped when the box holds a newer one: " + str(kept))

	# 4. The engine's reason under an answer to you, and only to you.
	var me := {"id": 5, "name": "Me"}
	var reply := {"kind": "reply", "actor": 3, "origin": "rule", "data": {"with": 5, "says": "I can't, sorry.", "next": "carry_on"}}
	var plain := Words.describe(reply, me, NAMES)
	for k in Words.WHY_NOT:
		var e: Dictionary = reply.duplicate(true)
		e["data"]["why"] = k
		e["data"]["price"] = 6
		var line := Words.describe(e, me, NAMES)
		var want: String = "You don't have 6 coin." if k == "coin" else (str(Words.WHY_NOT[k]) % "Ari Vota") + "."
		check(line.begins_with(plain) and want in line, "the reason %s isn't shown: %s" % [k, line])
	var theirs: Dictionary = reply.duplicate(true)
	theirs["data"]["with"] = 4
	theirs["data"]["why"] = "food"
	check(not ("spare" in Words.describe(theirs, me, NAMES)), "a reason shown under an answer to someone else")
	var odd: Dictionary = reply.duplicate(true)
	odd["data"]["why"] = "undecided"
	check(Words.describe(odd, me, NAMES) == plain, "a reason that isn't TALK_WHY's is shown")

func _process(_delta: float) -> bool:
	# 5. ctrl+a selects the line whole, so the next key replaces it (Godot's own LineEdit, as the Talk box uses it).
	# Run on the first frame: a control takes keys only once the tree is running.
	var box := LineEdit.new()
	box.max_length = 140
	root.add_child(box)
	box.text = "an old line that was kept"
	box.grab_focus()
	box.caret_column = 7
	_key(KEY_A, 0, true)
	check(box.get_selected_text() == box.text, "ctrl+a doesn't select the whole line")
	_key(KEY_N, "N".unicode_at(0), false)
	check(box.text == "N", "typing over a selected line doesn't replace it: " + box.text)

	for b in bad:
		printerr("talk box: " + b)
	print("talk box: %s" % ("ok" if bad.is_empty() else "%d failed" % bad.size()))
	quit(0 if bad.is_empty() else 1)
	return true


func _key(code: Key, unicode: int, ctrl: bool) -> void:
	for down in [true, false]:
		var ev := InputEventKey.new()
		ev.keycode = code
		ev.physical_keycode = code
		ev.unicode = unicode if down else 0
		ev.ctrl_pressed = ctrl
		ev.pressed = down
		root.push_input(ev)
