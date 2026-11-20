# The Talk box's rules, apart from its widgets (main.gd draws them): whom the line in the box goes to, and what
# becomes of a line that can't be said. It reads what it is given and decides nothing about the world.
# aipt3 (2026-11-20, Steering 09:07Z): 5 typed lines went to a villager the player hadn't meant (the list was
# rebuilt under the click, and a pick who had left was replaced by the first name on it), and about 8 lines were
# never sent, with nothing saying so.
extends RefCounted

var chosen := -1     # whom the line in the box is for
var picked := false  # the player picked them. The client replaces only its own default, never a pick.


func pick(id: int) -> void:
	chosen = id
	picked = true


func settle(ids: Array, has_text: bool) -> int:
	## Whom Say would send to now, or -1 for nobody. `ids`: who is here and can hear, in the list's order. The
	## first of them stands in only while the player has picked nobody and the box is empty: a line being written
	## keeps whom it was written to, and so does a pick, even after they've left.
	if not ids.has(chosen) and not picked and not has_text:
		chosen = int(ids[0]) if not ids.is_empty() else -1
	return chosen if ids.has(chosen) else -1


func note(ids: Array, has_text: bool, names: Dictionary, roles: Dictionary) -> String:
	## The line over the Talk row: whom Say sends to, or why it can't send and that the typed line is kept.
	if ids.has(chosen):
		return "To %s." % who(chosen, names, roles)
	var kept := " Your line is kept: pick who hears it." if not ids.is_empty() else " Your line is kept."
	if chosen >= 0:
		var gone := "%s isn't here now." % str(names.get(chosen, "They"))
		if has_text:
			return gone + kept
		return gone + (" Pick who to talk to." if not ids.is_empty() else "")
	if has_text:
		return ("Nobody is picked." if not ids.is_empty() else "Nobody is here to hear you.") + kept
	return ""


func unsent(ids: Array, names: Dictionary, roles: Dictionary, busy: bool) -> String:
	## Why Say sent nothing, for the log: a line that isn't sent is said to be unsent, and stays in the box.
	if ids.has(chosen):
		return "Not said yet: your last action is still on its way. Your line is kept: press Say again." if busy else ""
	return "Not said. " + note(ids, true, names, roles)


static func who(id: int, names: Dictionary, roles: Dictionary) -> String:
	var role := str(roles.get(id, ""))
	return str(names.get(id, "#%d" % id)) + (", " + role if role != "" else "")


static func lost(box_text: String, line: String, name: String) -> Dictionary:
	## A line that was sent and not heard (they had left, or were asleep, or the server refused it). `box`: what
	## the Talk box should hold now; `returned`: whether that is the lost line, to be selected whole so typing
	## replaces it; `log`: what the log says. With a newer line in the box the lost one is written out in the log.
	if box_text.strip_edges() == "":
		return {"box": line, "returned": true,
				"log": "Your line to %s is back in the Talk box: Say sends it again, typing replaces it." % name}
	return {"box": box_text, "returned": false,
			"log": "Your line to %s wasn't heard. The Talk box holds a newer line, so the lost one is kept here: “%s”"
			% [name, line.replace("[", "[lb]")]}
