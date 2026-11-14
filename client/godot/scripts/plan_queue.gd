# The player's own plan (readable-world C2, G5 checklist line 2): a short queue of steps the player builds,
# sent one at a time, each when the one before it has settled. It lives in the client as the player's input
# and is never sent or stored as world state: the server only ever sees one act at a time, as before.
#
# A step is {action, target (-1 for none), place ("" for none), label}. How a step settles, from the receipts
# that cite its act event (sim: the first receipt's parent is the act):
#   go_to      waits for "arrived" or "already here" (its first receipt only says it set off)
#   activity   (work, rest, home, socialize, explore) holds the plan until the player presses Next: the next
#              step would replace it at once, so the player says when it's had enough
#   the rest   is done with its first receipt
# A failed receipt stops the plan at that step, with the reason, until the player retries or clears.
extends RefCounted

const ACTIVITY := ["work", "rest", "home", "socialize", "explore"]
# Acts the player may plan from anywhere: they name no person, and the engine decides when each comes up.
const ANYWHERE := ["buy_food", "go_market", "work", "socialize", "rest", "explore", "home"]
const MAX_STEPS := 8


static func step(action: String, target := -1, place := "", label := "") -> Dictionary:
	return {"action": action, "target": target, "place": place, "label": label}


static func settle(s: Dictionary, receipts: Array) -> String:
	## "wait", "done", "hold" or "failed: WHY", from the receipts (event dicts) that cite the step's act.
	for r in receipts:
		var d: Dictionary = r["data"]
		if not d.get("ok", false):
			return "failed: " + str(d.get("why", ""))
	if receipts.is_empty():
		return "wait"
	if s["action"] == "go_to":
		for r in receipts:
			if str(r["data"].get("why", "")) in ["arrived", "already here"]:
				return "done"
		return "wait"
	return "hold" if s["action"] in ACTIVITY else "done"


static func lines(plan: Array, state: String, why_words: Dictionary) -> String:
	## The plan as the player reads it: the step in hand marked, what's left after it, and why it stopped.
	if plan.is_empty():
		return "No plan. Turn on \"Plan ahead\" and press actions or places on the map to line them up."
	var out := []
	for i in plan.size():
		var lab: String = plan[i]["label"]
		if i == 0:
			match state:
				"wait":
					out.append("[b]> %s[/b]  (under way)" % lab)
				"hold":
					out.append("[b]> %s[/b]  (press Next when you're done)" % lab)
				"":
					out.append("[b]> %s[/b]" % lab)
				_:
					var why := state.trim_prefix("failed: ")
					out.append("[color=#e07070][b]x %s[/b]: %s. Retry or clear.[/color]" % [lab, why_words.get(why, why)])
		else:
			out.append("   %d. %s" % [i + 1, lab])
	return "\n".join(out)
