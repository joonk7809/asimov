"""Static checks for the AGENTS.md rules (stdlib ast, no imports of the checked code).

Each rule here is one a reviewer has had to repeat; a lint makes it a check that runs with the tests
(tests/test_lints.py). A rule is only checked where it can be checked precisely: what this module can't
see (a state edit with no event, a float read from JSON at run time, a draw returned by a helper, taint that
crosses a function call) stays with replay and the invariants. The check is syntactic: names are resolved
through the module's imports (`import time as t` is `time`), and nothing is inferred about types.

    python -m asiimov.lint          # findings on the package, exit 1 if any

Rules:
  int     simulation arithmetic is integer-only (sim core)
  rng     randomness only through Rng; no wall clock, process ids, hash(), id(), dynamic imports, set or
          directory order (sim core)
  keyed   T22: a counter draw (Rng.draw/normal_milli) in the tick loop only on a registered own-actor stream,
          and a pick never indexes a list (or pops from it) with a draw (sim core)
  order   mind replies are fetched and applied only in Sim._commit, inside a loop over `self._order(...)` (or a
          list built from it in _commit), and only Sim._order reads the order setting (sim.py)

The sim core is SIM_CORE plus the class bodies of every Sim subclass anywhere in the package (ArmSim is run
and replayed by fork.py; ProbeSim and AuthorshipSim replay logs), all four rules.

A false positive is allowed on its own line with a comment, its reason mandatory:
    x = path / "events.jsonl"  # lint: allow int -- pathlib join, not a division
An unknown rule, a missing reason, or a suppression that suppresses nothing is itself a finding.
"""
import ast
import io
import os
import re
import sys
import tokenize
from dataclasses import dataclass

PKG = os.path.dirname(os.path.abspath(__file__))

# The sim core: what runs or rebuilds a world (asiimov.sim's imports within the package, plus replay and
# persist). tests/test_lints.py checks this against the actual import graph, so a new module the engine
# imports has to be put here or in MIND_SIDE.
SIM_CORE = ("sim.py", "rng.py", "params.py", "worldmap.py", "memory.py", "events.py", "replay.py", "persist.py")
# Imported by the engine but on the far side of the mind interface: what a mind returns is origin "mind",
# recorded verbatim and replayed from the log, so it may sample (NoisyMind uses os.urandom; a reply carries
# float sampling settings). Not checked by int/rng/keyed. tests/test_lints.py pins this to exactly minds.py.
MIND_SIDE = ("minds.py",)
# Roots of the engine's import graph.
CORE_ROOTS = ("sim.py", "replay.py", "persist.py")
# Not linted for Sim subclasses: this module (its tables name the banned calls).
NOT_SCANNED = ("lint.py",)

# keyed: streams a counter draw may use in the tick loop, by f-string prefix. A counter on such a stream is
# moved only by draws made for the stream's own agent. How many draws it takes may depend on who is around,
# but every draw it hands out lands on that same agent, so another agent's presence never shifts a draw that
# decides someone else (T22). Any draw that decides between agents, or whose stream others can move, goes
# through Rng.at(stream, tick, ...). Genesis (params.py) draws a fixed sequence from (seed, n_agents) before
# any tick, so it is not in scope for this rule.
DRAW_STREAMS = {
    "jitter:": "an agent's own wake-time jitter, drawn only when that agent schedules its wake",
    "steal:": "the thief's own steal and notice rolls, drawn only when that agent steals",
}
KEYED_GENESIS = ("params.py",)
RNG_CALLS = ("draw", "at", "raw", "normal_milli")
COUNTER_CALLS = ("draw", "normal_milli")

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

# order: functions besides _commit that may reach the mind, -> why. Each is called only from where its
# entry says (checked: ORDER_CALLERS), so the reply still reaches world state only in _commit's _order loop.
ORDER_MIND_REACH = {
    "_fly": "pace-2 (engine 38): called only by _commit; it files this tick's plan/reflect requests in "
            "state['inflight'] and returns the replies due now, unapplied; _commit applies them in its "
            "_order-ranked loop, and a lost call (restart) is asked again here without being applied",
    "_send": "pace-2 (engine 38): called only by _fly; asks one filed request (inline when workers <= 1, else on "
             "a thread) and returns a Future; nothing is applied",
}
ORDER_CALLERS = {"_fly": ("_commit",), "_send": ("_fly",)}
ORDER_READERS = ("_setup", "snapshot", "_order", "__init__")

# int: math functions that stay in the integers.
INT_MATH = ("isqrt", "gcd", "lcm", "comb", "perm", "factorial")
FLOAT_MODULES = ("math", "cmath", "statistics", "decimal", "fractions", "numpy")
FLOAT_CALLS = ("operator.truediv", "operator.itruediv", "operator.__truediv__", "operator.__itruediv__")
POW_CALLS = ("pow", "builtins.pow", "operator.pow", "operator.ipow", "operator.__pow__")
ENTROPY_MODULES = ("random", "secrets", "uuid", "numpy")
CLOCK = {"time": ("time", "time_ns", "monotonic", "monotonic_ns", "perf_counter", "perf_counter_ns",
                  "process_time", "process_time_ns"),
         "datetime": ("now", "utcnow", "today"), "date": ("today",),
         "os": ("urandom", "getrandom", "getpid", "getppid")}
DIR_ORDER = {"os": ("listdir", "scandir"), "glob": ("glob", "iglob")}
DYNAMIC_IMPORT = ("__import__", "importlib.import_module", "importlib.__import__")
SET_OPS = (ast.BitOr, ast.BitAnd, ast.Sub, ast.BitXor)
SET_METHODS = ("union", "intersection", "difference", "symmetric_difference", "copy")
ORDERED_CONSUMERS = ("list", "tuple", "enumerate", "iter", "next", "zip")

RULES = ("int", "rng", "keyed", "order")
SUPPRESS = re.compile(r"#\s*lint:\s*allow\s+(\S+)(?:\s+--\s*(.*))?$")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def __str__(self):
        return f"{self.path}:{self.line}: [{self.rule}] {self.message}"


# ---- tree helpers ------------------------------------------------------------------------------------

FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPES = FUNCS + (ast.Lambda,)


def _ancestors(node):
    while (node := getattr(node, "_lint_parent", None)) is not None:
        yield node


def _func(node) -> str | None:
    for a in _ancestors(node):
        if isinstance(a, FUNCS):
            return a.name
    return None


def _scope(node, kinds=FUNCS):
    """The nearest enclosing function (or the module)."""
    for a in _ancestors(node):
        if isinstance(a, kinds) or isinstance(a, ast.Module):
            return a
    return node


def _scope_chain(node):
    return [a for a in _ancestors(node) if isinstance(a, FUNCS + (ast.Module,))]


def _walk_scope(scope):
    """The nodes of one function's (or the module's) own body, not of functions or classes nested in it."""
    todo = list(scope.body) if isinstance(scope, (ast.Module, *FUNCS)) else [scope]
    while todo:
        n = todo.pop()
        yield n
        if isinstance(n, FUNCS + (ast.ClassDef, ast.Lambda)):
            continue
        todo.extend(ast.iter_child_nodes(n))


def _dotted(node) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _module_root(name: str | None) -> str:
    return (name or "").split(".")[0]


def _aliases(tree) -> dict[str, str]:
    """Local name -> the dotted name it was imported as, for every import in the module."""
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    root = a.name.split(".")[0]
                    out[root] = root
        elif isinstance(n, ast.ImportFrom):
            mod = "." * n.level + (n.module or "")
            for a in n.names:
                if a.name != "*":
                    out[a.asname or a.name] = f"{mod}.{a.name}" if mod else a.name
    return out


class Ctx:
    """One module being linted: its tree, the nodes in scope, and per-scope facts computed once."""

    def __init__(self, tree, path, roots):
        self.tree, self.path, self.base = tree, path, os.path.basename(path)
        self.roots = roots
        self.alias = _aliases(tree)
        self._sets, self._taint, self._rngfn, self._assigned, self._kind = {}, {}, {}, {}, {}
        self._fn = {}  # id(node) -> the name of the function it is in
        every, todo = [], [(tree, None)]
        while todo:  # one pass: parents, enclosing function names, and the node list
            n, name = todo.pop()
            every.append(n)
            self._fn[id(n)] = name
            inner = n.name if isinstance(n, FUNCS) else name
            for c in ast.iter_child_nodes(n):
                c._lint_parent = n
                todo.append((c, inner))
        self._nodes = every if roots == [tree] else [n for r in roots for n in ast.walk(r)]

    def func(self, node) -> str | None:
        return self._fn.get(id(node))

    def nodes(self):
        return self._nodes

    def canon(self, node) -> str | None:
        d = _dotted(node)
        if d is None:
            return None
        head, _, rest = d.partition(".")
        head = self.alias.get(head, head)
        return f"{head}.{rest}" if rest else head

    def assignments(self, scope):
        """name -> list of (kind, value): kind 'set' (=, :=, : T =), 'aug' (with its op) or 'other'."""
        if id(scope) in self._assigned:
            return self._assigned[id(scope)]
        out = {}

        def add(target, kind, value):
            if isinstance(target, ast.Name):
                out.setdefault(target.id, []).append((kind, value))
            elif isinstance(target, (ast.Tuple, ast.List)):
                for e in target.elts:
                    add(e.value if isinstance(e, ast.Starred) else e, "unpack", value)

        for n in _walk_scope(scope):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    add(t, "set", n.value)
            elif isinstance(n, ast.AnnAssign):
                add(n.target, "set" if n.value is not None else "other", n.value)
            elif isinstance(n, ast.AugAssign):
                add(n.target, "aug", n)
            elif isinstance(n, ast.NamedExpr):
                add(n.target, "set", n.value)
            elif isinstance(n, (ast.For, ast.AsyncFor, ast.comprehension)):
                add(n.target, "other", None)
            elif isinstance(n, ast.withitem) and n.optional_vars is not None:
                add(n.optional_vars, "other", None)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    out.setdefault((a.asname or a.name).split(".")[0], []).append(("other", None))
        if isinstance(scope, FUNCS):
            args = scope.args
            for a in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
                if a is not None:
                    out.setdefault(a.arg, []).append(("other", None))
        self._assigned[id(scope)] = out
        return out

    # -- sets: a local name whose every assignment in its function is a set
    def set_names(self, scope) -> set:
        if id(scope) not in self._sets:
            names, asg = set(), self.assignments(scope)
            changed = True
            while changed:
                changed = False
                for name, srcs in asg.items():
                    if name in names or not any(k == "set" for k, _ in srcs):
                        continue
                    if all((k == "set" and self._setlike(v, names, scope))
                           or (k == "aug" and isinstance(v.op, SET_OPS)) for k, v in srcs):
                        names.add(name)
                        changed = True
            self._sets[id(scope)] = names
        return self._sets[id(scope)]

    def _setlike(self, e, names, scope) -> bool:
        if isinstance(e, (ast.Set, ast.SetComp)):
            return True
        if isinstance(e, ast.Call) and self.canon(e.func) in ("set", "frozenset"):
            return True
        if isinstance(e, ast.Name):
            return e.id in names
        if isinstance(e, ast.BinOp) and isinstance(e.op, SET_OPS):
            return any(self._setlike(s, names, scope) or _keys_view(s) for s in (e.left, e.right))
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr in SET_METHODS:
            return self._setlike(e.func.value, names, scope)
        return False

    def is_set(self, e) -> bool:
        """A set expression, or a local name bound only to sets in the innermost function assigning it."""
        if isinstance(e, ast.Name):
            for sc in _scope_chain(e):
                if e.id in self.assignments(sc):
                    return e.id in self.set_names(sc)
            return False
        sc = _scope(e)
        return self._setlike(e, self.set_names(sc), sc)

    # -- rng callables: x = self.rng.draw / getattr(self.rng, "draw")
    def rng_fns(self, scope) -> dict:
        if id(scope) not in self._rngfn:
            out = {}
            for name, srcs in self.assignments(scope).items():
                kinds = [_rng_ref(v) if k == "set" else None for k, v in srcs]
                if kinds and all(kinds) and len(set(kinds)) == 1:
                    out[name] = kinds[0]
            self._rngfn[id(scope)] = out
        return self._rngfn[id(scope)]

    def draw_kind(self, node) -> str | None:
        """Which Rng method a call is (draw, at, raw, normal_milli), or None."""
        if not isinstance(node, ast.Call):
            return None
        if id(node) not in self._kind:
            self._kind[id(node)] = self._draw_kind(node)
        return self._kind[id(node)]

    def _draw_kind(self, node) -> str | None:
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in RNG_CALLS:
            return f.attr
        if (k := _rng_ref(f)) is not None and not isinstance(f, ast.Attribute):
            return k  # getattr(rng, "draw")(...)
        if isinstance(f, ast.Name):
            for sc in _scope_chain(node):
                if f.id in self.assignments(sc):
                    return self.rng_fns(sc).get(f.id)
        return None

    def draws(self, node) -> bool:
        return any(self.draw_kind(c) for c in _walk_unkeyed(node))

    # -- taint: local names whose value comes from a draw, to a fixpoint
    def tainted(self, scope) -> set:
        if id(scope) not in self._taint:
            names, asg = set(), self.assignments(scope)
            changed = True
            while changed:
                changed = False
                for name, srcs in asg.items():
                    if name in names:
                        continue
                    for k, v in srcs:
                        e = v.value if k == "aug" else v
                        if k in ("set", "unpack", "aug") and e is not None and (
                                self.draws(e) or any(isinstance(c, ast.Name) and c.id in names
                                                     for c in _walk_unkeyed(e))):
                            names.add(name)
                            changed = True
                            break
            self._taint[id(scope)] = names
        return self._taint[id(scope)]

    def is_tainted(self, e) -> bool:
        names = set().union(*(self.tainted(sc) for sc in _scope_chain(e)))
        return self.draws(e) or any(isinstance(c, ast.Name) and c.id in names for c in _walk_unkeyed(e))


def _keys_view(e) -> bool:
    return (isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr == "keys"
            and not e.args and not e.keywords)


def _rng_ref(e) -> str | None:
    """`rng.draw` (uncalled) or `getattr(rng, "draw")`: which Rng method it names."""
    if isinstance(e, ast.Attribute) and e.attr in RNG_CALLS:
        return e.attr
    if (isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "getattr" and len(e.args) >= 2
            and isinstance(e.args[1], ast.Constant) and e.args[1].value in RNG_CALLS):
        return e.args[1].value
    return None


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


def _outer(node) -> bool:
    """Not the `.value` of an attribute (so `a.b.c` is looked at once, as a whole)."""
    p = getattr(node, "_lint_parent", None)
    return not (isinstance(p, ast.Attribute) and p.value is node)


def _named(cx, node):
    """(canonical dotted name, parts) for an outermost Name/Attribute in Load context, else None."""
    if not isinstance(node, (ast.Name, ast.Attribute)) or not isinstance(node.ctx, ast.Load) or not _outer(node):
        return None
    c = cx.canon(node)
    return (c, c.split(".")) if c else None


def _nonneg_int(e) -> bool:
    return isinstance(e, ast.Constant) and type(e.value) is int and e.value >= 0


# ---- int -------------------------------------------------------------------------------------------

def check_int(cx) -> list[Finding]:
    out = []

    def f(node, msg):
        out.append(Finding(cx.path, node.lineno, "int", msg))

    for node in cx.nodes():
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, complex)):
            f(node, f"float literal {node.value!r}")
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Div):
            f(node, "true division `/` (use `//`)")
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.Pow):
            if not _nonneg_int(node.right if isinstance(node, ast.BinOp) else node.value):
                f(node, "`**` with an exponent that is not a non-negative int literal (can give a float)")
        elif isinstance(node, ast.Call) and cx.canon(node.func) in ("float", "complex"):
            f(node, f"`{cx.canon(node.func)}(` call")
        elif isinstance(node, ast.Call) and cx.canon(node.func) in POW_CALLS and len(node.args) == 2 \
                and not node.keywords and not _nonneg_int(node.args[1]):
            f(node, f"`{cx.canon(node.func)}(x, e)` with an exponent that is not a non-negative int literal")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if _module_root(a.name) in FLOAT_MODULES and a.name not in ("math", "cmath"):  # math: uses checked
                    f(node, f"import of float module `{a.name}`")
        elif isinstance(node, ast.ImportFrom) and _module_root(node.module) in FLOAT_MODULES:
            names = [a.name for a in node.names]
            if node.module != "math" or any(n not in INT_MATH for n in names):
                f(node, f"import from float module `{node.module}`: {', '.join(names)}")
        elif (n := _named(cx, node)) is not None:
            c, parts = n
            if c in FLOAT_CALLS:
                f(node, f"`{c}` is true division")
            elif parts[0] in ("math", "cmath") and len(parts) >= 2 and (parts[0] == "cmath" or parts[1] not in INT_MATH):
                f(node, f"`{c}` is float arithmetic")
    return out


# ---- rng -------------------------------------------------------------------------------------------

def check_rng(cx) -> list[Finding]:
    out = []

    def f(node, msg):
        out.append(Finding(cx.path, node.lineno, "rng", msg))

    for node in cx.nodes():
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
        elif (n := _named(cx, node)) is not None:
            c, parts = n
            if c in DYNAMIC_IMPORT:
                f(node, f"`{c}`: a dynamic import hides what the sim core runs")
            elif len(parts) >= 2 and parts[-1] in CLOCK.get(parts[-2], ()):
                f(node, f"`{c}`: wall clock, process id or OS entropy in the sim core")
            elif len(parts) >= 2 and parts[-1] in DIR_ORDER.get(parts[-2], ()):
                if not any(isinstance(a, ast.Call) and isinstance(a.func, ast.Name) and a.func.id == "sorted"
                           for a in _ancestors(node)):
                    f(node, f"`{c}` order is the filesystem's: wrap it in sorted()")
            elif (len(parts) >= 2 and parts[0] in ("random", "secrets", "uuid")) or parts[:2] == ["numpy", "random"]:
                f(node, f"`{c}` (draw through Rng)")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and cx.canon(node.func) in ("hash", "id"):
            f(node, f"`{node.func.id}(` varies between processes (PYTHONHASHSEED, addresses)")
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and cx.is_set(node.iter):
            f(node.iter, "iterating a set: its order is not the log's (sort it)")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ORDERED_CONSUMERS \
                and any(cx.is_set(a) for a in node.args):
            f(node, f"`{node.func.id}(` of a set: its order is not the log's (sort it)")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join" \
                and node.args and cx.is_set(node.args[0]):
            f(node, "`.join(` of a set: its order is not the log's (sort it)")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pop" \
                and not node.args and cx.is_set(node.func.value):
            f(node, "`.pop()` of a set takes an arbitrary element (sort it, or take min())")
    return out


# ---- keyed (T22) -------------------------------------------------------------------------------------

def _stream_prefix(cx, arg, func_node) -> str | None:
    """The literal start of a stream argument: a str constant, an f-string's leading text, or a local name
    assigned exactly once in the function from one of those."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value
    if isinstance(arg, ast.JoinedStr):
        head = arg.values[0] if arg.values else None
        return head.value if isinstance(head, ast.Constant) and isinstance(head.value, str) else ""
    if isinstance(arg, ast.Name) and func_node is not None:
        srcs = cx.assignments(func_node).get(arg.id, [])
        if len(srcs) == 1 and srcs[0][0] == "set":
            return _stream_prefix(cx, srcs[0][1], None)
    return None


def check_keyed(cx) -> list[Finding]:
    out = []

    def f(node, msg):
        out.append(Finding(cx.path, node.lineno, "keyed", msg))

    def pick(n, target_node, what):
        target = _dotted(target_node) or "<expr>"
        if target.split(".")[-1].isupper():
            return  # a module constant table (SYLLABLES): no candidates to add or drop
        if (cx.base, cx.func(n), target) in INDEX_PICK_ALLOW:
            return
        f(n, f"`{target}{what[0]}` {what[1]} by a draw: pick the candidate with the lowest keyed score instead (T22)")

    for node in cx.nodes():
        kind = cx.draw_kind(node)
        if cx.base not in KEYED_GENESIS and cx.base != "rng.py" and kind in COUNTER_CALLS:
            fn = next((a for a in _ancestors(node) if isinstance(a, FUNCS)), None)
            prefix = _stream_prefix(cx, node.args[0], fn) if node.args else None
            if prefix is None:
                f(node, "counter draw on a stream this lint can't read: name it with a literal or f-string")
            elif not any(prefix.startswith(p) for p in DRAW_STREAMS):
                f(node, f"counter draw on stream {prefix!r}...: use Rng.at(stream, tick, ...) (T22), "
                        "or register an own-actor stream in asiimov/lint.py DRAW_STREAMS")
        # Index picks: x[<draw>] or x.pop(<draw>), directly or through local names derived from one (to a
        # fixpoint, in the function or one enclosing it). A draw used only as the key of min/max/sorted over
        # the candidates is the lowest-keyed-score pick T22 asks for, not an index.
        if isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Store) and cx.is_tainted(node.slice):
            pick(node, node.value, ("[...]", "indexed"))
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "pop"
              and node.args and cx.is_tainted(node.args[0])):
            pick(node, node.func.value, (".pop(...)", "popped"))
    return sorted(set(out), key=lambda x: (x.line, x.message))


# ---- order (sim.py and Sim subclasses) -----------------------------------------------------------------

def _from_order(cx, e, fn) -> bool:
    """`self._order(...)`, or a local of `fn` every assignment of which is built from one (to a fixpoint)."""
    asg = cx.assignments(fn) if fn is not None else {}
    names, changed = set(), True
    while changed:
        changed = False
        for name, srcs in asg.items():
            if name not in names and srcs and all(
                    k == "set" and any((isinstance(c, ast.Call) and _dotted(c.func) == "self._order")
                                       or (isinstance(c, ast.Name) and c.id in names) for c in ast.walk(v))
                    for k, v in srcs):
                names.add(name)
                changed = True
    if isinstance(e, ast.Call) and _dotted(e.func) == "enumerate" and e.args:
        e = e.args[0]
    return (isinstance(e, ast.Call) and _dotted(e.func) == "self._order") or (isinstance(e, ast.Name) and e.id in names)


def check_order(cx) -> list[Finding]:
    out = []

    def f(node, msg):
        out.append(Finding(cx.path, node.lineno, "order", msg))

    for node in cx.nodes():
        if not isinstance(node, (ast.Name, ast.Attribute, ast.Call)):
            continue
        fn = cx.func(node)
        reach = fn in ("_commit", *ORDER_MIND_REACH)
        if isinstance(node, ast.Name) and node.id == "dispatch" and isinstance(node.ctx, ast.Load) and fn != "_commit":
            f(node, f"`dispatch` used in `{fn}`: mind requests are sent only from Sim._commit")
        elif isinstance(node, ast.Attribute) and node.attr == "decide" and isinstance(node.ctx, ast.Load) \
                and fn not in ORDER_MIND_REACH:
            f(node, "`.decide` called directly: go through dispatch in Sim._commit")
        elif (isinstance(node, ast.Attribute) and node.attr == "mind" and isinstance(node.value, ast.Name)
              and node.value.id == "self" and fn not in ("_setup", "__init__") and not reach):
            f(node, f"`self.mind` used in `{fn}`: the mind is reached only from Sim._commit")
        elif (isinstance(node, ast.Attribute) and node.attr == "order" and isinstance(node.value, ast.Name)
              and node.value.id == "self" and fn not in ORDER_READERS):
            f(node, f"`self.order` read in `{fn}`: who goes first is decided only in Sim._order")
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self" \
                and node.attr in ORDER_CALLERS and fn not in ORDER_CALLERS[node.attr] \
                and not (isinstance(node.ctx, ast.Store) or fn == node.attr):
            f(node, f"`self.{node.attr}` used in `{fn}`: it reaches the mind, and only "
                    f"{', '.join(ORDER_CALLERS[node.attr])} may call it")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_parse":
            loop = next((a for a in _ancestors(node) if isinstance(a, (ast.For, ast.AsyncFor))), None)
            func = next((a for a in _ancestors(node) if isinstance(a, FUNCS)), None)
            if fn != "_commit" or loop is None or not _from_order(cx, loop.iter, func):
                f(node, "a mind reply is applied outside a loop over `self._order(...)` in Sim._commit")
    return out


CHECKS = {"int": check_int, "rng": check_rng, "keyed": check_keyed, "order": check_order}


# ---- suppressions -------------------------------------------------------------------------------------

def _suppressions(src, path):
    """{(line, rule): reason} from `# lint: allow <rule> -- <reason>` comments, and findings for bad ones."""
    sup, bad = {}, []
    if "lint:" not in src:
        return sup, bad
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError):
        return sup, bad
    for t in toks:
        if t.type != tokenize.COMMENT or not re.match(r"#\s*lint:", t.string):
            continue
        line = t.start[0]
        m = SUPPRESS.match(t.string.strip())
        if m is None:
            bad.append(Finding(path, line, "suppress", f"unreadable suppression {t.string!r}: "
                                                       "write `# lint: allow <rule> -- <reason>`"))
        elif m.group(1) not in RULES:
            bad.append(Finding(path, line, "suppress", f"suppression of unknown rule {m.group(1)!r} "
                                                       f"(rules: {', '.join(RULES)})"))
        elif not (m.group(2) or "").strip():
            bad.append(Finding(path, line, "suppress", f"suppression of `{m.group(1)}` without a reason "
                                                       "(`-- <why this is not a violation>`)"))
        else:
            sup[(line, m.group(1))] = m.group(2).strip()
    return sup, bad


def lint_source(src: str, path: str, rules=RULES, classes=None) -> list[Finding]:
    """Lint one module; with `classes` (names), only those class bodies."""
    tree = ast.parse(src, filename=path)
    roots = [tree] if classes is None else [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name in classes]
    cx = Ctx(tree, path, roots)
    found = set()
    for r in rules:
        found.update(CHECKS[r](cx))
    sup, out = _suppressions(src, path)
    used = set()
    for x in found:
        if (x.line, x.rule) in sup:
            used.add((x.line, x.rule))
        else:
            out.append(x)
    for line, rule in sorted(set(sup) - used):
        out.append(Finding(path, line, "suppress", f"suppression of `{rule}` suppresses nothing: remove it"))
    return sorted(out, key=lambda x: (x.line, x.rule, x.message))


def _parse(path):
    with open(path) as fh:
        return ast.parse(fh.read())


def core_imports(pkg: str = PKG) -> set[str]:
    """Package modules reachable from CORE_ROOTS by relative imports (file names)."""
    seen, todo = set(), list(CORE_ROOTS)
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for n in ast.walk(_parse(os.path.join(pkg, name))):
            if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module:
                todo.append(n.module.split(".")[0] + ".py")
            elif isinstance(n, ast.ImportFrom) and n.level == 1:
                todo += [a.name + ".py" for a in n.names]
            elif isinstance(n, ast.ImportFrom) and (n.module or "").startswith("asiimov."):
                todo.append(n.module.split(".")[1] + ".py")
    return seen


def sim_subclasses(pkg: str = PKG) -> dict[str, list[str]]:
    """file name -> names of the classes in it that subclass Sim (directly or through another), outside SIM_CORE."""
    classes = []  # (file, class name, base last names)
    for name in sorted(os.listdir(pkg)):
        if name.endswith(".py") and name not in NOT_SCANNED:
            for n in ast.walk(_parse(os.path.join(pkg, name))):
                if isinstance(n, ast.ClassDef):
                    bases = [(_dotted(b) or "").split(".")[-1] for b in n.bases]
                    classes.append((name, n.name, bases))
    subs, changed = {"Sim"}, True
    while changed:
        changed = False
        for _, cls, bases in classes:
            if cls not in subs and subs & set(bases):
                subs.add(cls)
                changed = True
    out = {}
    for file, cls, bases in classes:
        if cls != "Sim" and cls in subs and file not in SIM_CORE:
            out.setdefault(file, []).append(cls)
    return out


def lint_package(pkg: str = PKG) -> list[Finding]:
    out = []
    rel = os.path.dirname(pkg)
    for name in SIM_CORE:
        p = os.path.join(pkg, name)
        with open(p) as fh:
            src = fh.read()
        rules = ("int", "rng", "keyed") + (("order",) if name == "sim.py" else ())
        out += lint_source(src, os.path.relpath(p, rel), rules)
    for name, classes in sorted(sim_subclasses(pkg).items()):
        p = os.path.join(pkg, name)
        with open(p) as fh:
            src = fh.read()
        out += lint_source(src, os.path.relpath(p, rel), RULES, classes=set(classes))
    return out


if __name__ == "__main__":
    found = lint_package()
    for x in found:
        print(x)
    sys.exit(1 if found else 0)
