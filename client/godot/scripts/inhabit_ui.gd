# Plain words for an inhabitant: what each event it witnessed means, its status, its plan, who is here and
# what each button does. Only reads what the server sent (the view, the witnessed events and the names in
# them); it never decides anything, and an act is still the server's offer, sent as it is.
extends RefCounted

const VERB := {"work": "work", "shop": "shop", "socialize": "see people", "home": "go home", "rest": "rest",
		"explore": "explore", "visit": "visit someone", "sleep": "sleep"}
const ACTION := {"buy_food": "Buy food at the market", "go_market": "Go to the market", "ask_kin": "Ask %s (family) for food",
		"ask_food": "Ask %s for food", "buy_from": "Buy food from %s", "steal": "Steal food from %s",
		"give_food": "Give food to %s", "give_coin": "Give a coin to %s", "seek": "Go and find %s",
		"avoid": "Keep away from %s", "work": "Go to work", "socialize": "Go and see people", "home": "Go home",
		"rest": "Rest", "explore": "Explore", "wait": "Wait", "agree": "Say yes to %s", "decline": "Say no to %s"}
const NOW := {"work": "Working", "shop": "Shopping", "socialize": "Seeing people", "home": "Staying",
		"rest": "Resting", "explore": "Exploring", "visit": "Visiting"}
const SHORT := {"ask_kin": "ask family for food", "ask_food": "ask for food", "buy_from": "buy food", "steal": "steal food",
		"give_food": "give food", "give_coin": "give a coin", "seek": "go to them", "avoid": "keep away",
		"agree": "say yes", "decline": "say no"}
const DOING := {"buy_food": "buy food", "go_market": "go to the market", "ask_kin": "ask family for food",
		"ask_food": "ask for food", "buy_from": "buy food", "steal": "steal", "give_food": "give food",
		"give_coin": "give a coin", "seek": "find someone", "avoid": "keep away", "work": "go to work",
		"socialize": "see people", "home": "go home", "rest": "rest", "explore": "explore", "wait": "wait",
		"agree": "say yes", "decline": "say no"}
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
		"works sick": "you work while sick", "said yes": "they get what they asked for",
		"said no": "they're told no", "no ask": "they hadn't asked you anything",
		"already asked": "they're still waiting for an answer", "asleep": "you were asleep",
		"bedtime": "it's bedtime, so you head home", "falling asleep": "you're falling asleep",
		"no food to give": "you have no food to give", "no coin to give": "you have no coin to give", "no stock": "the market has no food left",
		"unparseable or unknown action": "that isn't something you can do",
		"heading there": "you set off", "already here": "you're already there", "arrived": "you arrive"}
const TO_YOU := {"gave": "you gave them 1 food", "sold": "you sold them 1 food", "refused": "you said no",
		"kin gave food": "you gave them food", "kin refused": "you said no", "stolen": "they stole food from you",
		"caught": "you caught them stealing", "nothing to spare": "you had nothing to spare",
		"kin has nothing to spare": "you had nothing to spare", "target asleep": "you were asleep",
		"target not here": "you weren't there any more", "no coin": "they couldn't pay", "target has no food": "you had no food"}
const ANSWERS := ["gave", "sold", "refused", "kin gave food", "kin refused"]  # an ask's yes or no
# What a villager did with your own act aimed at them, by name (%s): the outcome first, then the why.
const BY_THEM := {"refused": "%s said no.", "kin refused": "%s said no.", "nothing to spare": "%s had nothing to spare.",
		"kin has nothing to spare": "%s had nothing to spare.", "kin gave food": "%s gave you food.",
		"target asleep": "%s is asleep.", "target not here": "%s isn't here any more.",
		"unknown target": "%s isn't here any more.",
		"caught": "%s caught you stealing.", "stolen": "You stole food from %s and got away with it."}
const ASKING := {"ask_kin": "You asked %s for food", "ask_food": "You asked %s for food", "buy_from": "You tried to buy food from %s",
		"steal": "You tried to steal food from %s", "give_food": "You gave food to %s", "give_coin": "You gave a coin to %s"}
const ASKED := {"ask_kin": "asked you for food", "ask_food": "asked you for food", "buy_from": "wanted to buy food from you",
		"steal": "tried to steal from you", "give_food": "gave you food", "give_coin": "gave you a coin",
		"seek": "came to see you"}
const HERE_ONLY := ["ask_kin", "ask_food", "buy_from", "steal"]  # sim.INSTANT_STEPS: refused with the target elsewhere
const RATED := ["ask_kin", "ask_food", "buy_from", "steal", "buy_food"]  # held to the cooldown (sim.RATED_ACTS)


static func shown(item: Dictionary, present: Dictionary) -> bool:
	## Whether a menu item gets a button. Asks and theft need the target here ("target not here"), and
	## asking as kin needs family here ("not kin"): those can only be refused, so they're left off. Seek,
	## avoid and the gifts work from anywhere (a gift walks to the target first). `present`: id -> kin.
	var t := int(item["target"])
	if t >= 0 and item["action"] in HERE_ONLY and not present.has(t):
		return false
	return item["action"] != "ask_kin" or present.get(t, false)


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
	## One line for an event the villager witnessed, or "" for bookkeeping it needn't read. Another's act aimed
	## at this villager adds what that villager said of it, its own mind's words (the server's `says`).
	var line := _line(e, me, names)
	return line + said(e.get("data", {})) if line != "" else ""


static func said(d: Dictionary) -> String:
	## The quoted line a villager says with its act aimed at you, or "".
	var w = d.get("says")
	return "\n    [i]\"%s\"[/i]" % str(w).replace("[", "[lb]") if w is String and w != "" else ""


static func _line(e: Dictionary, me: Dictionary, names: Dictionary) -> String:
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
		"ask":
			var what: String = ASKED.get(str(d.get("action", "")), "asked you something")
			if d.get("to") != null and int(d["to"]) == me_id:
				return "%s %s. Say yes or no." % [a, what]
			return "%s %s." % [a, what.replace("you", _who(d.get("to"), me_id, names))]
		"plan_receipt":
			return "You plan your day." if d.get("ok", false) else ""
		"receipt":
			return _receipt(e, d, mine, a, me, names)
	return ""  # act (the button press itself), day_rhythm, intent_* bookkeeping


static func _receipt(e: Dictionary, d: Dictionary, mine: bool, a: String, me: Dictionary, names: Dictionary) -> String:
	var me_id := int(me.get("id", -1))
	var why := str(d.get("why", ""))
	var said: String = WHY.get(why, why)
	if "%d" in said:
		said = said % int(d.get("food", 0))
	var doing: String = DOING.get(str(d.get("action", "")), str(d.get("action", "")).replace("_", " "))
	if not mine:  # someone's act aimed at this villager; its mind decided the ask, and your villager's nature the answer
		var line := "%s %s" % [a, ASKED.get(str(d.get("action", "")), "tried to " + doing + " with you")]
		var answer: String = TO_YOU.get(why, "")
		if why in ANSWERS and not d.has("answered"):  # nobody pressed yes or no: the villager's habits answered
			answer = "you didn't answer, so your habits did: " + answer
		return line + (": " + answer + "." if answer != "" else ".")
	if e.get("origin") == "rule" and why in ["heading to market", "started", "waited"] and d.get("ok", false):
		return ""  # the activity line that follows says it
	if str(d.get("action", "")) == "go_to":  # go-to-1: where you're walking, and when you'll be there
		var where := place(d.get("place"), me)
		if why == "arrived":
			return ""  # the arrive event says it
		if why == "already here":
			return "You're already at %s, so you stay." % where
		var mins := (int(d.get("arrive", e["tick"])) - int(e["tick"])) * 10
		return "You set off for %s; you'll be there in about %d minutes." % [where, mins]
	var whom = d.get("to", d.get("victim"))  # a theft's receipt names its victim
	if whom != null and int(whom) != me_id and why in ["gave", "sold"]:
		var who := _who(whom, me_id, names)
		match str(d.get("action", "")):
			"give_food":
				return "You gave %d food to %s." % [int(d.get("food", 1)), who]
			"give_coin":
				return "You gave %d coin to %s." % [int(d.get("coin", 1)), who]
			"buy_from":
				return "%s sold you food." % who
			"ask_food", "ask_kin":
				return "You asked %s for food. %s gave you some." % [who, who]
	if whom != null and int(whom) != me_id and why in BY_THEM:
		var them := _who(whom, me_id, names)
		var lead: String = ASKING.get(str(d.get("action", "")), "")
		return ((lead % them + ". ") if lead != "" and why not in ["stolen", "caught"] else "") + BY_THEM[why] % them
	if d.get("to") != null and int(d["to"]) != me_id:
		doing += " (" + _who(d["to"], me_id, names) + ")"
	return ("Done: %s, %s." if d.get("ok", false) else "Didn't work: %s, because %s.") % [doing, said]


static func status(v: Dictionary, me: Dictionary, per_day: int) -> String:
	var lines := ["[b]%s[/b], %s" % [me.get("name", "?"), me.get("role", "")], clock(int(v["tick"]), per_day)]
	if v["asleep"]:
		lines.append("Asleep at %s." % place(v["place"], me))
	elif v.get("route") != null:
		lines.append("Walking to %s, there at %s." % [place(v["route"]["to"], me),
				clock(int(v["route"]["arrive"]), per_day).split(", ")[1]])
	else:
		lines.append("%s at %s." % [NOW.get(str(v["activity"]), str(v["activity"])), place(v["place"], me)])
	lines.append("Hunger %d/1000, %s." % [int(v["hunger"]), hunger_word(int(v["hunger"]))])
	lines.append("You carry %d food and %d coin." % [int(v["food"]), int(v["coin"])])
	var price := int(v.get("food_price", 0))
	if price > 0:
		lines.append("Food costs %d coin at the market; your coin buys %d." % [price, int(v["coin"]) / price])
	var need := needs(v)
	if need != "":
		lines.append("[color=#f0c060]%s[/color]" % need)
	if v.has("market_food"):
		lines.append("The market has %d food." % int(v["market_food"]))
	if v.get("occasion") != null:
		lines.append("[color=#f0c060]On your mind: %s.[/color]" % str(v["occasion"]["text"]))
	return "\n".join(lines)


static func needs(v: Dictionary) -> String:
	## What this body needs now, from its own numbers: food first, as its mind would see it.
	var price := int(v.get("food_price", 0))
	var h := int(v["hunger"])
	if int(v["food"]) == 0 and h >= 550:
		if price > 0 and int(v["coin"]) >= price:
			return "You need food: buy some at the market, or ask someone."
		return "You need food and can't pay for it: ask family or someone here, or work for coin."
	if int(v["food"]) == 0:
		return "You have no food left for later."
	return ""


static func intent(v: Dictionary, names: Dictionary) -> String:
	## Its own mind's current intent: what this body meant to do, with the step it is on marked.
	var it = v.get("intent")
	if it == null:
		return ""
	var steps := []
	for i in it["steps"].size():
		var st: Dictionary = it["steps"][i]
		var who: String = "" if st.get("target") == null else names.get(int(st["target"]), "someone")
		var txt := str(VERB.get(str(st["do"]), st["do"]))
		if str(st["do"]) in ACTION:
			txt = action_label(str(st["do"]), who)
			txt = txt[0].to_lower() + txt.substr(1)
		steps.append("[b]%s[/b]" % txt if i == int(it["i"]) else txt)
	return "You meant to %s: %s." % [str(it["goal"]).trim_suffix("."), " → ".join(steps)]


static func plan(v: Dictionary, per_day: int, names := {}) -> String:
	var meant := intent(v, names)
	if v["plan"].is_empty():
		return (meant + "\n" if meant != "" else "") + "No plan for today yet."
	var hour := float(int(v["tick"]) % per_day) * 24.0 / per_day
	var out := []
	for b in v["plan"]:
		var line := "%02d–%02d  %s" % [int(b["start_h"]), int(b["end_h"]), VERB.get(str(b["activity"]), b["activity"])]
		out.append("[b]> %s[/b]" % line if hour >= float(b["start_h"]) and hour < float(b["end_h"]) else "   " + line)
	if meant != "":
		out.push_front(meant)
	return "\n".join(out)


static func asks(v: Dictionary, names: Dictionary, secs_per_tick: float) -> String:
	## Asks waiting on this villager's answer (player-ask-1), with how long is left before its nature answers.
	var out := []
	for q in v.get("asks", []):
		var left := int(q["until"]) - int(v["tick"])
		out.append("[color=#f0c060][b]%s %s.[/b]%s\nSay yes or no within %d tick%s (about %ds), or your habits answer for you.[/color]"
				% [q["name"], ASKED.get(str(q["action"]), "asked you something"), said(q), left, "" if left == 1 else "s",
				int(left * secs_per_tick)])
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



static func heard(m: Dictionary, per_day: int) -> String:
	## Words spoken near you (readable-world E2): who said what, quoted; what was said to you in the ask colour.
	## A villager's words are text, never markup.
	var t := clock(int(m["tick"]), per_day).split(", ")[1]
	var out := []
	for x in m.get("lines", []):
		var line := "[b]%s[/b]: \u201c%s\u201d" % [str(x["name"]).replace("[", "[lb]"), str(x["text"]).replace("[", "[lb]")]
		out.append("[color=#f0c060]%s[/color]" % line if m.get("to_player", false) else line)
	if out.is_empty():
		return ""
	return "[color=#8a93a6]%s[/color]  %s" % [t, "\n         ".join(out)]
