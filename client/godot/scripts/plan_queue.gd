# The player's own plan (readable-world C2, G5 checklist line 2): a short queue of steps the player builds,
# sent one at a time, each when the one before it has settled. It lives in the client as the player's input
# and is never sent or stored as world state: the server only ever sees one act at a time, as before.
#
# A step is {action, target (-1 for none), place ("" for none), label}. How a step settles, from the events
# that cite its act event (sim: a receipt's or a meet's first parent is the act, or the step holding it):
#   go_to      waits for "arrived" or "already here" (its first receipt only says it set off)
#   gifts, go_market
#              wait past "started" / "heading to market" (the walk began) for the act's own receipt: gave,
#              bought, no stock... (review of #63: settling on "started" let the next step cancel the gift)
#   seek       is done when the meet happens (a meet event, no receipt)
#   activity, avoid
#              hold the plan until the player presses Next: the next step would replace it at once
#   the rest   is done with its first receipt
# A failed receipt stops the plan at that step, with the reason, until the player retries or clears. A step
# still waiting after STEP_TICKS (the engine's step time, plus slack) has run out of time: it failed too.
extends RefCounted

const ACTIVITY := ["work", "rest", "home", "socialize", "explore", "avoid"]
const WALK_FIRST := ["give_food", "give_coin", "go_market"]  # their first receipt only says the walk began
const STARTED := ["started", "heading to market"]
const STEP_TICKS := 20  # sim.DEFAULT_STEP_TICKS (18) and two ticks of slack
# Acts the player may plan from anywhere: they name no person, and the engine decides when each comes up.
const ANYWHERE := ["buy_food", "go_market", "work", "socialize", "rest", "explore", "home"]
const MAX_STEPS := 8


static func step(action: String, target := -1, place := "", label := "") -> Dictionary:
	return {"action": action, "target": target, "place": place, "label": label}


static func settle(s: Dictionary, evs: Array, waited := 0) -> String:
	## "wait", "done", "hold" or "failed: WHY", from the events (dicts) that cite the step's act, `waited`
	## ticks after it was sent.
	var receipts := evs.filter(func(e): return e.get("kind", "receipt") == "receipt")
	for r in receipts:
		var d: Dictionary = r["data"]
		if not d.get("ok", false):
			return "failed: " + str(d.get("why", ""))
	var a: String = s["action"]
	var st := "wait"
	if a == "go_to":
		if receipts.any(func(r): return str(r["data"].get("why", "")) in ["arrived", "already here"]):
			st = "done"
	elif a == "seek":
		if evs.any(func(e): return e.get("kind", "") == "meet"):
			st = "done"
	elif a in ACTIVITY:
		if not receipts.is_empty():
			st = "hold"
	elif a in WALK_FIRST:
		if receipts.any(func(r): return not str(r["data"].get("why", "")) in STARTED):
			st = "done"
	elif not receipts.is_empty():
		st = "done"
	if st == "wait" and waited > STEP_TICKS:
		return "failed: time ran out"
	return st


static func bought(s: Dictionary, evs: Array) -> bool:
	## A market trip that bought on arrival (aipt1 fix 2): a buy_food queued right after it has nothing left to do.
	return s["action"] == "go_market" and evs.any(func(e): return e.get("kind", "receipt") == "receipt" \
			and e["data"].get("ok", false) and str(e["data"].get("why", "")) == "bought")


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
					out.append("[color=#e07070][b]x %s[/b]: %s. Skipped next tick; Retry keeps it.[/color]" % [lab, why_words.get(why, why)])
		else:
			out.append("   %d. %s" % [i + 1, lab])
	return "\n".join(out)
