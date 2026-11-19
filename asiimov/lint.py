"""Static checks for the AGENTS.md rules (stdlib ast, no imports of the checked code).

Each rule here is one a reviewer has had to repeat; a lint makes it a check that runs with the tests
(tests/test_lints.py). A rule is only checked where it can be checked precisely: what this module can't
see (a state edit with no event, a float read from JSON at run time) stays with replay and the invariants.

    python -m asiimov.lint          # findings on the package, exit 1 if any

Rules:
  int     simulation arithmetic is integer-only (sim core)
  rng     randomness only through Rng; no wall clock, hash(), id(), set or directory order (sim core)
  keyed   T22: a counter draw (Rng.draw/normal_milli) in the tick loop only on a registered own-actor stream,
          and a pick never indexes a list with a draw (sim core)
  order   mind replies are fetched and applied only in Sim._commit, inside `for ... in self._order(...)`,
          and only Sim._order reads the order setting (sim.py)
"""
import ast
import os
import sys
from dataclasses import dataclass

PKG = os.path.dirname(os.path.abspath(__file__))

# The sim core: what runs or rebuilds a world (asiimov.sim's imports within the package, plus replay and
# persist). tests/test_lints.py checks this against the actual import graph, so a new module the engine
# imports has to be put here or in MIND_SIDE.
SIM_CORE = ("sim.py", "rng.py", "params.py", "worldmap.py", "memory.py", "events.py", "replay.py", "persist.py")
# Imported by the engine but on the far side of the mind interface: what a mind returns is origin "mind",
# recorded verbatim and replayed from the log, so it may sample (NoisyMind uses os.urandom; a reply carries
# float sampling settings). Not checked by int/rng/keyed.
MIND_SIDE = ("minds.py",)
# Roots of the engine's import graph.
CORE_ROOTS = ("sim.py", "replay.py", "persist.py")

# keyed: streams a counter draw may use in the tick loop, by f-string prefix. A counter that only the
# stream's own agent can move is safe from T22 (someone else being around can't shift it). Any other draw
# whose odds depend on who is around goes through Rng.at(stream, tick, ...). Genesis (params.py) draws a
# fixed sequence from (seed, n_agents) before any tick, so it is not in scope for this rule.
DRAW_STREAMS = {
    "jitter:": "an agent's own wake-time jitter, drawn only when that agent schedules its wake",
    "steal:": "the thief's own steal and notice rolls, drawn only when that agent steals",
}
KEYED_GENESIS = ("params.py",)
RNG_CALLS = ("draw", "at", "raw", "normal_milli")

# keyed: an index into a list with a draw, allowed only where the list is no set of candidates (who is
# around can't change it). (file, function, list name) -> why.
INDEX_PICK_ALLOW = {
    ("params.py", "generate_population", "roles_pool"):
        "the role table built from the ROLE_SHARES constant at genesis; a pick of a role, not of a person",
    ("sim.py", "_sickness", "adults"):
        "every adult in the village: roles are fixed at genesis and agents are only added there, so the list "
        "is the same in a twin whatever anyone does; not a pick among those present. Reported, not changed "
        "(changing it changes the log: an engine bump)",
}

# int: math functions that stay in the integers.
INT_MATH = ("isqrt", "gcd", "lcm", "comb", "perm", "factorial")
FLOAT_MODULES = ("math", "cmath", "statistics", "decimal", "fractions", "numpy")
ENTROPY_MODULES = ("random", "secrets", "uuid", "numpy")
CLOCK = {"time": ("time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns",
                  "process_time", "process_time_ns"),
         "datetime": ("now", "utcnow", "today"), "date": ("today",),
         "os": ("urandom", "getrandom")}
DIR_ORDER = {"os": ("listdir", "scandir"), "glob": ("glob", "iglob")}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self):
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


def _parents(tree):
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._lint_parent = node


def _ancestors(node):
    while (node := getattr(node, "_lint_parent", None)) is not None:
        yield node


def _func(node) -> str | None:
    for a in _ancestors(node):
        if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return a.name
    return None


def _dotted(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _module_root(name: str | None) -> str:
    return (name or "").split(".")[0]


# ---- int -------------------------------------------------------------------------------------------

def check_int(tree, path) -> list[Finding]:
    out = []

    def f(node, msg):
        out.append(Finding(path, node.lineno, "int", msg))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, complex)):
            f(node, f"float literal {node.value!r}")
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Div):
            f(node, "true division `/` (use `//`)")
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Pow):
            e = node.right if isinstance(node, ast.BinOp) else node.value
            if not (isinstance(e, ast.Constant) and type(e.value) is int and e.value >= 0):
                f(node, "`**` with an exponent that is not a non-negative int literal (can give a float)")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("float", "complex"):
            f(node, f"`{node.func.id}(` call")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if _module_root(a.name) in FLOAT_MODULES and a.name != "math":  # math: its uses are checked
                    f(node, f"import of float module `{a.name}`")
        elif isinstance(node, ast.ImportFrom) and _module_root(node.module) in FLOAT_MODULES:
            names = [a.name for a in node.names]
            if node.module != "math" or any(n not in INT_MATH for n in names):
                f(node, f"import from float module `{node.module}`: {', '.join(names)}")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("math", "cmath"):
            if node.value.id == "cmath" or node.attr not in INT_MATH:
                f(node, f"`{node.value.id}.{node.attr}` is float arithmetic")
    return out


# ---- rng -------------------------------------------------------------------------------------------

def _set_expr(node) -> bool:
    return isinstance(node, (ast.Set, ast.SetComp)) or (
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("set", "frozenset"))


def check_rng(tree, path) -> list[Finding]:
    out = []

    def f(node, msg):
        out.append(Finding(path, node.lineno, "rng", msg))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if _module_root(a.name) in ENTROPY_MODULES:
                    f(node, f"import of `{a.name}` (draw through Rng)")
        elif isinstance(node, ast.ImportFrom):
            if _module_root(node.module) in ENTROPY_MODULES:
                f(node, f"import from `{node.module}` (draw through Rng)")
            for a in node.names:
                if a.name in CLOCK.get(_module_root(node.module), ()) or a.name in DIR_ORDER.get(node.module, ()):
                    f(node, f"import of `{node.module}.{a.name}`")
        elif isinstance(node, ast.Attribute):
            d = _dotted(node)
            if d is None:
                continue
            parts = d.split(".")
            if len(parts) >= 2 and parts[-1] in CLOCK.get(parts[-2], ()):
                f(node, f"`{d}`: wall clock or OS entropy in the sim core")
            elif len(parts) >= 2 and parts[-1] in DIR_ORDER.get(parts[-2], ()):
                if not any(isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id == "sorted"
                           for a in _ancestors(node)):
                    f(node, f"`{d}` order is the filesystem's: wrap it in sorted()")
            elif len(parts) >= 2 and parts[-2] == "random" and parts[0] in ("random", "np", "numpy"):
                f(node, f"`{d}` (draw through Rng)")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("hash", "id"):
            f(node, f"`{node.func.id}(` varies between processes (PYTHONHASHSEED, addresses)")
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and _set_expr(node.iter):
            f(node.iter, "iterating a set: its order is not the log's (sort it)")
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id in ("list", "tuple", "enumerate", "iter", "next") and node.args and _set_expr(node.args[0])):
            f(node, f"`{node.func.id}(` of a set: its order is not the log's (sort it)")
    return out


# ---- keyed (T22) -------------------------------------------------------------------------------------

def _rng_call(node, names=RNG_CALLS) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in names


def _walk_unkeyed(node):
    """ast.walk, minus the key= argument of min/max/sorted (where a draw scores candidates)."""
    todo = [node]
    while todo:
        n = todo.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("min", "max", "sorted")
                    and isinstance(c, ast.keyword) and c.arg == "key"):
                continue
            todo.append(c)


def _draws(node) -> bool:
    return any(_rng_call(c) for c in _walk_unkeyed(node))


def _stream_prefix(arg, func_node) -> str | None:
    """The literal start of a stream argument: a str constant, an f-string's leading text, or a local name
    assigned exactly once in the function from one of those."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.JoinedStr):
        head = arg.values[0] if arg.values else None
        return head.value if isinstance(head, ast.Constant) and isinstance(head.value, str) else ""
    if isinstance(arg, ast.Name) and func_node is not None:
        sources = [n.value for n in ast.walk(func_node) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == arg.id for t in n.targets)]
        if len(sources) == 1:
            return _stream_prefix(sources[0], None)
    return None


def check_keyed(tree, path) -> list[Finding]:
    out = []
    base = os.path.basename(path)

    def f(node, msg):
        out.append(Finding(path, node.lineno, "keyed", msg))

    for node in ast.walk(tree):
        if base not in KEYED_GENESIS and base != "rng.py" and _rng_call(node, ("draw", "normal_milli")):
            fn = next((a for a in _ancestors(node) if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
            prefix = _stream_prefix(node.args[0], fn) if node.args else None
            if prefix is None:
                f(node, "counter draw on a stream this lint can't read: name it with a literal or f-string")
            elif not any(prefix.startswith(p) for p in DRAW_STREAMS):
                f(node, f"counter draw on stream {prefix!r}...: use Rng.at(stream, tick, ...) (T22), "
                        "or register an own-actor stream in asiimov/lint.py DRAW_STREAMS")
    # Index picks: x[<expression using a draw>], directly or through a local name assigned from one (in
    # the subscript's function or one enclosing it). A draw used only as the key of min/max/sorted over the
    # candidates is the lowest-keyed-score pick T22 asks for, not an index.
    scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
    tainted: dict[int, set] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and _draws(n.value):
            scope = next((a for a in _ancestors(n) if isinstance(a, scopes)), tree)
            for t in n.targets:
                if isinstance(t, (ast.Tuple, ast.List)):
                    t_names = [e.id for e in t.elts if isinstance(e, ast.Name)]
                else:
                    t_names = [t.id] if isinstance(t, ast.Name) else []
                tainted.setdefault(id(scope), set()).update(t_names)
    for n in ast.walk(tree):
        if not isinstance(n, ast.Subscript) or isinstance(n.ctx, ast.Store):
            continue
        names = set().union(*(tainted.get(id(a), set()) for a in [*_ancestors(n)] if isinstance(a, scopes + (ast.Module,))))
        if not (_draws(n.slice) or any(isinstance(c, ast.Name) and c.id in names for c in _walk_unkeyed(n.slice))):
            continue
        target = _dotted(n.value) or "<expr>"
        if target.split(".")[-1].isupper():
            continue  # a module constant table (SYLLABLES): no candidates to add or drop
        if (base, _func(n), target) in INDEX_PICK_ALLOW:
            continue
        f(n, f"`{target}[...]` indexed by a draw: pick the candidate with the lowest keyed score instead (T22)")
    return sorted(set(out), key=lambda x: (x.line, x.message))


# ---- order (sim.py) ----------------------------------------------------------------------------------

ORDER_READERS = ("_setup", "snapshot", "_order", "__init__")


def check_order(tree, path) -> list[Finding]:
    out = []

    def f(node, msg):
        out.append(Finding(path, node.lineno, "order", msg))

    for node in ast.walk(tree):
        fn = _func(node)
        if isinstance(node, ast.Name) and node.id == "dispatch" and isinstance(node.ctx, ast.Load) and fn != "_commit":
            f(node, f"`dispatch` used in `{fn}`: mind requests are sent only from Sim._commit")
        elif isinstance(node, ast.Attribute) and node.attr == "decide" and isinstance(node.ctx, ast.Load):
            f(node, "`.decide` called directly: go through dispatch in Sim._commit")
        elif (isinstance(node, ast.Attribute) and node.attr == "mind" and isinstance(node.value, ast.Name)
              and node.value.id == "self" and fn not in ("_setup", "__init__", "_commit")):
            f(node, f"`self.mind` used in `{fn}`: the mind is reached only from Sim._commit")
        elif (isinstance(node, ast.Attribute) and node.attr == "order" and isinstance(node.value, ast.Name)
              and node.value.id == "self" and fn not in ORDER_READERS):
            f(node, f"`self.order` read in `{fn}`: who goes first is decided only in Sim._order")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_parse":
            loop = next((a for a in _ancestors(node) if isinstance(a, (ast.For, ast.AsyncFor))), None)
            in_order = (loop is not None and isinstance(loop.iter, ast.Call)
                        and _dotted(loop.iter.func) == "self._order")
            if fn != "_commit" or not in_order:
                f(node, "a mind reply is applied outside `for ... in self._order(...)` in Sim._commit")
    return out


CHECKS = {"int": check_int, "rng": check_rng, "keyed": check_keyed}


def lint_source(src: str, path: str, rules=("int", "rng", "keyed", "order")) -> list[Finding]:
    tree = ast.parse(src, filename=path)
    _parents(tree)
    out = []
    for r in rules:
        out += (check_order if r == "order" else CHECKS[r])(tree, path)
    return sorted(set(out), key=lambda x: (x.line, x.rule, x.message))


def core_imports(pkg: str = PKG) -> set[str]:
    """Package modules reachable from CORE_ROOTS by relative imports (file names)."""
    seen, todo = set(), list(CORE_ROOTS)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        with open(os.path.join(pkg, name)) as fh:
            tree = ast.parse(fh.read())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module:
                todo.append(n.module.split(".")[0] + ".py")
            elif isinstance(n, ast.ImportFrom) and n.level == 1:
                todo += [a.name + ".py" for a in n.names]
            elif isinstance(n, ast.ImportFrom) and (n.module or "").startswith("asiimov."):
                todo.append(n.module.split(".")[1] + ".py")
    return seen


def lint_package(pkg: str = PKG) -> list[Finding]:
    out = []
    for name in SIM_CORE:
        p = os.path.join(pkg, name)
        with open(p) as fh:
            src = fh.read()
        rules = ("int", "rng", "keyed") + (("order",) if name == "sim.py" else ())
        out += lint_source(src, os.path.relpath(p, os.path.dirname(pkg)), rules)
    return out


if __name__ == "__main__":
    found = lint_package()
    for x in found:
        print(x)
    sys.exit(1 if found else 0)
