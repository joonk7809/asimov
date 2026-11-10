# Plain words for an inhabitant: what each event it witnessed means, its status, its plan, who is here and
# what each button does. Only reads what the server sent (the view, the witnessed events and the names in
# them); it never decides anything, and an act is still the server's offer, sent as it is.
extends RefCounted

const VERB := {"work": "work", "shop": "shop", "socialize": "see people", "home": "go home", "rest": "rest",
		"explore": "explore", "sleep": "sleep"}
const ACTION := {"buy_food": "Buy food at the market", "go_market": "Go to the market", "ask_kin": "Ask %s (family) for food",
		"ask_food": "Ask %s for food", "buy_from": "Buy food from %s", "steal": "Steal food from %s",
		"give_food": "Give food to %s", "give_coin": "Give a coin to %s", "seek": "Go and find %s",
		"avoid": "Keep away from %s", "work": "Go to work", "socialize": "Go and see people", "home": "Go home",
		"rest": "Rest", "explore": "Explore", "wait": "Wait"}
const NOW := {"work": "Working", "shop": "Shopping", "socialize": "Seeing people", "home": "Staying",
		"rest": "Resting", "explore": "Exploring"}
const SHORT := {"ask_kin": "ask family for food", "ask_food": "ask for food", "buy_from": "buy food", "steal": "steal food",
		"give_food": "give food", "give_coin": "give a coin", "seek": "go to them", "avoid": "keep away"}
const DOING := {"buy_food": "buy food", "go_market": "go to the market", "ask_kin": "ask family for food",
		"ask_food": "ask for food", "buy_from": "buy food", "steal": "steal", "give_food": "give food",
		"give_coin": "give a coin", "seek": "find someone", "avoid": "keep away", "work": "go to work",
		"socialize": "see people", "home": "go home", "rest": "rest", "explore": "explore", "wait": "wait"}
const WHY := {"heading to market": "you head to the market", "started": "you start", "waited": "you wait",
		"bought": "you buy %d food", "had enough": "you have enough food, so you buy none",
		"not at market": "you aren't at the market", "not kin": "they aren't family",
		"one act per tick": "only one action counts each tick; the first one did",
		"not offered": "it wasn't possible any more by the time it came up",
		"cooldown": "too soon after your last ask; wait a little", "unknown target": "they aren't here any more",
		"target not here": "they aren't here any more", "target asleep": "they're asleep",
		"target has no food": "they have no food", "kin has nothing to spare": "they have nothing to spare",
		"kin gave food": "they gave you food", "kin refused": "they refused", "no coin": "you have no coin",
		"nothing to spare": "they have nothing to spare", "refused": "they refused", "sold": "they sold it to you",
		"gave": "done", "stolen": "you got away with it", "caught": "you were caught",
		"released": "you let go of the villager first", "stays home sick": "you stay home sick",
		"works sick": "you work while sick"}
const RATED := ["ask_kin", "ask_food", "buy_from", "steal", "buy_food"]  # held to the cooldown (sim.RATED_ACTS)


static func clock(tick: int, per_day: int) -> String:
	var m := (tick % per_day) * 1440 / per_day
	return "Day %d, %02d:%02d" % [tick / per_day + 1, m / 60, m % 60]


static func place(p, me: Dictionary) -> String:
	if p == null:
		return "on the road"
	if str(p) == str(me.get("home", "")):
		return "your home"
	return "the " + str(p).replace("_", " ")


static func hunger_word(h: int) -> String:
	if h < 300:
		return "full"
	if h < 550:
		return "a little hungry"
	if h < 800:
		return "hungry"
	return "starving"


static func _who(id, me_id: int, names: Dictionary, cap := false) -> String:
	if id == null:
		return "Someone" if cap else "someone"
	if int(id) == me_id:
		return "You" if cap else "you"
	return names.get(int(id), "Someone" if cap else "someone")


static func action_label(action: String, target_name: String) -> String:
	var s: String = ACTION.get(action, action.replace("_", " "))
	return s % target_name if "%s" in s else s


static func describe(e: Dictionary, me: Dictionary, names: Dictionary) -> String:
	## One line for an event the villager witnessed, or "" for bookkeeping it needn't read.
	var me_id := int(me.get("id", -1))
	var d: Dictionary = e.get("data", {})
	var mine: bool = e.get("actor") != null and int(e["actor"]) == me_id
	var a := _who(e.get("actor"), me_id, names, true)
	match e["kind"]:
		"possess":
			return "You take over %s." % me.get("name", "your villager")
		"release":
			return "You let go. %s's own habits take over." % me.get("name", "Your villager")
		"wake":
			return "%s wake%s up." % [a, "" if mine else "s"]
		"sleep":
			return "%s go%s to sleep." % [a, "" if mine else "es"]
		"activity":
			if str(d.get("activity", "")) == "home":
				return "%s head%s home." % [a, "" if mine else "s"]
			var v: String = VERB.get(str(d.get("activity", "")), str(d.get("activity", "")))
			return "%s head%s off to %s at %s." % [a, "" if mine else "s", v, place(d.get("place"), me)]
		"arrive":
			return "%s arrive%s at %s." % [a, "" if mine else "s", place(d.get("place"), me)]
		"eat":
			if d.get("household") != null and mine:
				return "You eat from %s's food." % _who(d["household"], me_id, names)
			return "%s eat%s." % [a, "" if mine else "s"]
		"produce":
			if d.has("coin"):
				return "You earn %d coin for your work." % int(d["coin"])
			return "You harvest %d food." % int(d.get("food", 0))
		"sell":
			return "%s sell%s %d food at the market." % [a, "" if mine else "s", int(d.get("food", 0))]
		"tithe":
			return "You pay %d coin to the village chest." % int(d.get("coin", 0))
		"stock":
			return "You stock up for the winter."
		"sick":
			return "You fall sick." if mine else ""
		"recover":
			return "You get over the sickness." if mine else ""
		"talk":
			var line := "%s talk%s with %s." % [a, "" if mine else "s", _who(d.get("with"), me_id, names)]
			for g in d.get("gossip", []):
				var k := str(g[1])
				if k.begins_with("thief:"):
					line += " %s hears that %s is a thief." % [_who(g[0], me_id, names, true),
							_who(int(k.split(":")[1]), me_id, names)]
			return line
		"meet":
			return "%s come%s to see %s." % [a, "" if mine else "s", _who(d.get("with"), me_id, names)]
		"snub":
			return "%s turn%s away from %s." % [a, "" if mine else "s", _who(d.get("with"), me_id, names)]
		"fire":
			return "Fire! A family's food stores burn."
		"plan_receipt":
			return "You plan your day." if d.get("ok", false) else ""
		"receipt":
			return _receipt(e, d, mine, a, me_id, names)
	return ""  # act (the button press itself), day_rhythm, intent_* bookkeeping


static func _receipt(e: Dictionary, d: Dictionary, mine: bool, a: String, me_id: int, names: Dictionary) -> String:
	var why := str(d.get("why", ""))
	var said: String = WHY.get(why, why)
	if "%d" in said:
		said = said % int(d.get("food", 0))
	var doing: String = DOING.get(str(d.get("action", "")), str(d.get("action", "")).replace("_", " "))
	if not mine:  # someone's act aimed at this villager
		return "%s tried to %s from you: %s." % [a, doing, said] if not d.get("ok", false) else \
				"%s: %s (%s)." % [a, doing, said]
	if e.get("origin") == "rule" and why in ["heading to market", "started", "waited"] and d.get("ok", false):
		return ""  # the activity line that follows says it
	return ("Done: %s, %s." if d.get("ok", false) else "Didn't work: %s, because %s.") % [doing, said]


static func status(v: Dictionary, me: Dictionary, per_day: int) -> String:
	var lines := ["[b]%s[/b], %s" % [me.get("name", "?"), me.get("role", "")], clock(int(v["tick"]), per_day)]
	if v["asleep"]:
		lines.append("Asleep at %s." % place(v["place"], me))
	elif v.get("route") != null:
		lines.append("Walking to %s." % place(v["route"]["to"], me))
	else:
		lines.append("%s at %s." % [NOW.get(str(v["activity"]), str(v["activity"])), place(v["place"], me)])
	lines.append("Hunger %d/1000, %s." % [int(v["hunger"]), hunger_word(int(v["hunger"]))])
	lines.append("You carry %d food and %d coin." % [int(v["food"]), int(v["coin"])])
	if v.has("market_food"):
		lines.append("The market has %d food." % int(v["market_food"]))
	if v.get("occasion") != null:
		lines.append("[color=#f0c060]On your mind: %s.[/color]" % str(v["occasion"]["text"]))
	return "\n".join(lines)


static func plan(v: Dictionary, per_day: int) -> String:
	if v["plan"].is_empty():
		return "No plan for today yet."
	var hour := float(int(v["tick"]) % per_day) * 24.0 / per_day
	var out := []
	for b in v["plan"]:
		var line := "%02d–%02d  %s" % [int(b["start_h"]), int(b["end_h"]), VERB.get(str(b["activity"]), b["activity"])]
		out.append("[b]> %s[/b]" % line if hour >= float(b["start_h"]) and hour < float(b["end_h"]) else "   " + line)
	return "\n".join(out)


static func here(v: Dictionary, doing: Dictionary) -> String:
	if v["present"].is_empty():
		return "Nobody else is here." if v["place"] != null else "You're on the road."
	var out := []
	for p in v["present"]:
		var tags := []
		if p["kin"]:
			tags.append("family")
		if p["has_food"]:
			tags.append("has food")
		var line := "%s, %s" % [p["name"], p["role"]]
		if not tags.is_empty():
			line += " (" + ", ".join(tags) + ")"
		if doing.has(int(p["id"])):
			line += ": " + doing[int(p["id"])]
		out.append(line)
	return "\n".join(out)

