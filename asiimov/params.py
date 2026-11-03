"""Population genesis: per-agent parameters that each drive a computation.

All values are integers (per-mille unless noted) so the simulation never depends on
platform float behavior. Genesis is recorded in the log, so replay loads it rather than
recomputing it.
"""
from .rng import Rng

TRAITS = ("honesty", "emotionality", "extraversion", "agreeableness", "conscientiousness", "openness")
VALUES = ("tradition", "conformity", "power", "achievement", "benevolence",
          "universalism", "security", "hedonism", "stimulation", "self_direction")
NEEDS = ("hunger", "rest", "safety", "belonging", "status", "autonomy",
         "novelty", "wealth", "kin_care", "meaning")
# Needs the P0 core actually simulates; the rest are carried for the soul and later systems.
SIMULATED_NEEDS = ("hunger", "rest", "belonging", "wealth")

# How traits load on values (sign and weight); a loose, documented prior, not a fitted model.
VALUE_LOADINGS = {
    "benevolence": {"agreeableness": 2, "honesty": 1},
    "universalism": {"openness": 2, "agreeableness": 1},
    "tradition": {"openness": -2, "conscientiousness": 1},
    "conformity": {"openness": -1, "conscientiousness": 2},
    "power": {"honesty": -2, "extraversion": 1},
    "achievement": {"conscientiousness": 1, "honesty": -1, "extraversion": 1},
    "security": {"emotionality": 2, "openness": -1},
    "hedonism": {"conscientiousness": -2, "extraversion": 1},
    "stimulation": {"openness": 1, "extraversion": 2, "emotionality": -1},
    "self_direction": {"openness": 2, "agreeableness": -1},
}

ROLE_SHARES = (("farmer", 40), ("crafter", 25), ("trader", 15), ("healer", 5), ("guard", 15))
SYLLABLES = ("ka", "mi", "ro", "ta", "len", "sa", "vo", "ri", "an", "de", "lu", "ne", "or", "is", "ba", "el")


def clamp(v: int, lo: int = 0, hi: int = 1000) -> int:
    return max(lo, min(hi, v))


def _name(rng: Rng, stream: str) -> str:
    n = rng.draw(stream, 2, 3)[1]
    s = "".join(SYLLABLES[rng.draw(stream, 0, len(SYLLABLES) - 1)[1]] for _ in range(n))
    return s.capitalize()


def _trait(rng: Rng, family_z: int, stream: str) -> int:
    # z-scores in milli-units; sibling correlation ~0.25 (0.5*family + 0.866*individual).
    own = rng.normal_milli(stream, 1000)
    z = (500 * family_z + 866 * own) // 1000
    return clamp(500 + z * 170 // 1000)


def _need_weights(rng: Rng, stream: str, role: str) -> dict:
    # Dirichlet(1)-like via exponential-ish draws, with a small role prior.
    prior = {"farmer": "hunger", "crafter": "status", "trader": "wealth",
             "healer": "kin_care", "guard": "safety", "child": "belonging", "elder": "meaning"}
    raw = {}
    for need in NEEDS:
        u = rng.draw(stream, 1, 1000)[1]
        raw[need] = 1000 - u + 1 + (400 if prior.get(role) == need else 0)
    total = sum(raw.values())
    return {k: v * 1000 // total for k, v in raw.items()}


def generate_population(seed: int, n_agents: int) -> dict:
    """Return {'agents': [...], 'families': [...]} for n_agents."""
    rng = Rng(seed * 7919 + 17)  # genesis has its own RNG so the sim stream starts clean
    agents, families = [], []
    roles_pool = [r for r, share in ROLE_SHARES for _ in range(share)]
    aid = 0
    fid = 0
    while aid < n_agents:
        size = min(rng.draw("fam_size", 1, 5)[1], n_agents - aid)
        surname = _name(rng, "surname")
        fam_z = {t: rng.normal_milli(f"famz:{t}", 1000) for t in TRAITS}
        fam_value_shift = {v: rng.draw("fam_values", -150, 150)[1] for v in VALUES}
        wealth_class = rng.draw("fam_wealth", 0, 2)[1]
        members = []
        for k in range(size):
            s = f"agent:{aid}"
            if k == 0:
                age = rng.draw(s, 22, 60)[1]
            elif k == 1:
                age = rng.draw(s, 20, 62)[1]
            else:
                age = rng.draw(s, 6, 80)[1]
            traits = {t: _trait(rng, fam_z[t], f"{s}:{t}") for t in TRAITS}
            if age < 14:
                role = "child"
            elif age >= 68:
                role = "elder"
            else:
                role = roles_pool[rng.draw(s, 0, len(roles_pool) - 1)[1]]
            scores = {}
            for v in VALUES:
                base = rng.normal_milli(f"{s}:val", 300) + fam_value_shift[v]
                load = sum(w * (traits[t] - 500) for t, w in VALUE_LOADINGS[v].items())
                scores[v] = base + load
            ranked = sorted(VALUES, key=lambda v: (-scores[v], v))
            agent = {
                "id": aid,
                "name": f"{_name(rng, s)} {surname}",
                "family": fid,
                "age": age,
                "role": role,
                "rhythm": {
                    # chronotype: minutes from village mean wake (06:30), sd 90, clipped to +-4h
                    "chronotype_min": max(-240, min(240, rng.normal_milli(f"{s}:chrono", 90))),
                    # children need more sleep, elders less (rough human ranges)
                    "sleep_need_min": rng.draw(s, *((540, 660) if age < 14 else (360, 480) if age >= 65 else (330, 570)))[1],
                    "jitter_max_min": rng.draw(s, 0, 40)[1],
                },
                "body": {
                    "metabolism": rng.draw(s, 700, 1400)[1],
                    "stamina": clamp(rng.draw(s, 200, 1000)[1] - (max(0, age - 50) * 10)),
                    "strength": clamp(rng.draw(s, 200, 1000)[1] - (300 if age < 14 else 0)),
                },
                "traits": traits,
                "values": {"top": ranked[:3], "bottom": ranked[-2:]},
                "need_weights": _need_weights(rng, s, role),
                "cognition": {
                    "deliberation_threshold": rng.draw(s, 0, 1000)[1],
                    "planning_horizon_days": rng.draw(s, 1, 7)[1],
                    "forgetting": rng.draw(s, 0, 1000)[1],
                    "credulity": rng.draw(s, 0, 1000)[1],
                },
                "start": {
                    "coin": (5 + 10 * wealth_class) + rng.draw(s, 0, 10)[1],
                    "food": rng.draw(s, 1, 4)[1] + wealth_class,
                },
            }
            agents.append(agent)
            members.append(aid)
            aid += 1
        families.append({"id": fid, "surname": surname, "members": members, "home": f"home_{fid}",
                         "wealth_class": wealth_class})
        fid += 1
    return {"agents": agents, "families": families}
