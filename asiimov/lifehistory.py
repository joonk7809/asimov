"""The life-history pilot (plan/v0-after-g3.md §5b, graph row LH): research only, on the neighbor-1 branch, never
merged. Each working adult gets a remembered past in genesis, by a keyed draw: a costly past with a sick neighbor,
a good one, or none. The past is 3 episodes in the engine's note style, marked "last year", at importance 5 so
they rank above everyday notes in the prompt's "Earlier" section and survive forgetting through the pilot.
A population's agent carries `past` (the arm) and `past_episodes`; Sim seeds episodes from the latter."""
import hashlib
import json

ARMS = ("costly", "good", "none")
PAST_IMPORTANCE = 5


def _draw(salt: int, key: str, n: int) -> int:
    return int(hashlib.sha256(f"lh:{salt}:{key}".encode()).hexdigest(), 16) % n


def past_texts(arm: str, neighbor: str) -> list[str]:
    if arm == "costly":
        return [f"last year: went to see {neighbor} at their home while they were sick",
                f"last year: fell sick at {neighbor}'s home near {neighbor}",
                "last year: was sick for 3 days and earned nothing, and my household went short of food"]
    if arm == "good":
        return [f"last year: went to see {neighbor} at their home while they were sick, and took them 1 food",
                "last year: stayed well after the visit",
                f"last year: at the harvest {neighbor} brought me 4 food for the help when they were sick"]
    return []


def seed_pasts(pop: dict, salt: int) -> dict:
    pop = json.loads(json.dumps(pop))
    adults = [a for a in pop["agents"] if a["role"] not in ("child", "elder")]
    for a in adults:
        arm = ARMS[_draw(salt, f"arm:{a['id']}", 3)]
        others = [o for o in adults if o["family"] != a["family"]]
        n = others[_draw(salt, f"who:{a['id']}", len(others))]
        a["past"] = arm
        a["past_episodes"] = [{"tick": -1000 - i,  # the prompt lists by -tick: oldest first
                              "importance": PAST_IMPORTANCE, "text": t}
                              for i, t in enumerate(past_texts(arm, n["name"]))]
    pop["pasts_salt"] = salt
    return pop
