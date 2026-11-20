"""PR guard: report what a change touches that the merge policy cares about, and fail on weakened tests.

    python scripts/pr_guard.py --base BASE_SHA --head HEAD_SHA [--repo DIR]

Diffs merge-base(BASE, HEAD)..HEAD with rename detection (git diff -M), so a moved test is not a deletion and
commits that landed on the base branch after the PR forked are not counted against it. It only runs git and
parses content (ast.parse, configparser, tomllib); nothing from the PR is executed, imported or installed. Only
.py files and pytest config files (pytest.ini, pytest.toml, .pytest.toml, setup.cfg, tox.ini, pyproject.toml,
conftest.py) are read, as bytes; any other path (images, binaries) is only named. Prints a JSON summary (also appended to
$GITHUB_STEP_SUMMARY when set):

- protected_paths_touched: tests/, verification/, .github/, AGENTS.md, scripts/pr_guard.py
- golden_or_engine_changed (+ golden_or_engine_detail): ENGINE_VERSION (asiimov/sim.py) or a module-level
  GOLDEN* constant (tests/test_world.py) changed
- weakening, each with a kind:
  - "test file deleted" / "test file renamed out of collection" (out of pytest's test_*.py / *_test.py)
  - "test function removed": an existing test function is gone. When N test functions are removed and N added
    in one file and its effective assert count did not drop, it is listed under `renamed` instead.
  - "asserts removed": fewer effective asserts in an existing test file. An assert on a constant truthy
    literal (`assert True`, `assert 1`), one with an `or <truthy literal>` operand, and one under a dead
    branch (`if False:`, `if 0:`, the `else` of `if True:`) does not count as effective.
  - "trivial assert added" / "dead assert added": more of those in an existing test file.
  - "skip/xfail added": a new reference to pytest.mark.skip/skipif/xfail, pytest.skip/xfail/importorskip
    (and their .Exception), unittest.skip*/expectedFailure/SkipTest, import aliases resolved, in code that
    reaches existing tests: an existing test function, module or class level code, a helper or fixture, a
    non-test module. A skip inside a test function the PR adds (or in a newly added test file) cannot weaken
    an existing test and is listed under `new_test_skips` instead.
  - "test shadowed by duplicate name": a test function name defined twice in one scope (only the last runs).
  - "parametrize cases reduced": an existing test's literal @pytest.mark.parametrize case count went down.
  - "pytest config": a change to addopts, testpaths, python_files, python_functions, python_classes or
    norecursedirs in pytest.ini / pytest.toml / setup.cfg / tox.ini / pyproject.toml, or an added --deselect, -k not,
    --ignore or collect_ignore line in those or a conftest.py.
  - "collection hook added or changed": pytest_collection_modifyitems added or changed in a conftest.py.
  - "unparseable after change": a .py file that parsed before no longer does.

Exit 1 when there is weakening, unless the PR carries the label `guard-override` (read from the event payload
at $GITHUB_EVENT_PATH, no network; an unreadable payload means no labels); then it is still reported and the
exit is 0. Protected paths and golden/engine changes alone never fail: they are reported for the merge policy
to escalate. Known limits: docs/merge-gate.md.
"""
import argparse
import ast
import configparser
import difflib
import json
import os
import re
import subprocess
import sys

try:
    import tomllib
except ImportError:  # Python < 3.11: pyproject.toml falls back to the line check
    tomllib = None

PROTECTED_DIRS = ("tests/", "verification/", ".github/")
PROTECTED_FILES = ("AGENTS.md", "scripts/pr_guard.py")
OVERRIDE_LABEL = "guard-override"
SKIPS = {"pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail", "pytest.skip", "pytest.xfail",
         "pytest.importorskip", "pytest.skip.Exception", "pytest.xfail.Exception", "unittest.skip",
         "unittest.skipIf", "unittest.skipUnless", "unittest.expectedFailure", "unittest.SkipTest",
         "unittest.case.SkipTest"}
PARAMETRIZE = "pytest.mark.parametrize"
PYTEST_CONFIG = ("pytest.ini", "pytest.toml", ".pytest.toml", "setup.cfg", "tox.ini", "pyproject.toml", "conftest.py")
INI_SECTIONS = {"pytest.ini": ("pytest",), "tox.ini": ("pytest",), "setup.cfg": ("tool:pytest",)}
COLLECTION_KEYS = ("addopts", "testpaths", "python_files", "python_functions", "python_classes", "norecursedirs")
COLLECTION_KEY_LINE = re.compile(r"\b(" + "|".join(COLLECTION_KEYS) + r")\b")
DESELECT = re.compile(r"--deselect|-k\s*[\"']?\s*not\b|--ignore|collect_ignore")
COLLECTION_HOOKS = ("pytest_collection_modifyitems",)


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def show(repo, sha, path):
    """File bytes at sha, or None when absent. Bytes, never decoded here: a file need not be UTF-8."""
    p = subprocess.run(["git", "-C", repo, "show", f"{sha}:{path}"], capture_output=True)
    return p.stdout if p.returncode == 0 else None


def text(data):
    return None if data is None else data.decode("utf-8", errors="replace")


def changes(repo, base, head):
    """[(status letter, old path, new path)] for base..head, renames followed."""
    out = git(repo, "diff", "--name-status", "-M", "-z", base, head).split("\0")
    res, i = [], 0
    while i < len(out) and out[i]:
        st = out[i][0]
        if st in "RC":
            res.append((st, out[i + 1], out[i + 2]))
            i += 3
        else:
            res.append((st, out[i + 1], out[i + 1]))
            i += 2
    return res


def protected(path):
    return path.startswith(PROTECTED_DIRS) or path in PROTECTED_FILES


def is_test_file(path):
    name = os.path.basename(path)
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def is_config(path):
    return os.path.basename(path) in PYTEST_CONFIG


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def parse(data):
    """ast of the bytes (coding declarations honoured), or None when absent or unparseable."""
    if data is None:
        return None
    try:
        return ast.parse(data)
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return None


def test_defs(tree):
    """[(qualified name, node)] of test functions, duplicates kept: test_x, or Class.test_x for methods."""
    out = []

    def walk(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
                out.append((prefix + n.name, n))
            elif isinstance(n, ast.ClassDef):
                walk(n.body, prefix + n.name + ".")
    walk(tree.body, "")
    return out


def test_names(tree):
    return {name for name, _ in test_defs(tree)}


def aliases(tree):
    alias = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            alias.update({a.asname or a.name: a.name for a in n.names})
        elif isinstance(n, ast.ImportFrom) and n.module:
            alias.update({a.asname or a.name: f"{n.module}.{a.name}" for a in n.names})
    return alias


def resolve(node, alias):
    name = dotted(node)
    if not name:
        return None
    root, _, rest = name.partition(".")
    return alias.get(root, root) + ("." + rest if rest else "")


def skip_refs(tree):
    """[owner] per skip/xfail reference: the qualified test function it sits in (decorators included), or None
    for anything else (module or class level, helpers, fixtures)."""
    alias = aliases(tree)
    owner = {}
    for name, fn in test_defs(tree):
        for n in ast.walk(fn):
            owner[id(n)] = name
    refs = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.Attribute, ast.Name)) and isinstance(getattr(n, "ctx", None), ast.Load):
            if resolve(n, alias) in SKIPS:
                refs.append(owner.get(id(n)))
    return refs


def truthy_constant(node):
    return isinstance(node, ast.Constant) and bool(node.value)


def trivial(test):
    """True for an assert that can't fail on its value: a truthy literal, or an `or` with a truthy literal."""
    if truthy_constant(test):
        return True
    return isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or) and any(map(truthy_constant, test.values))


def assert_counts(tree):
    """{"effective", "trivial", "dead"} assert counts; dead = inside the never-taken branch of `if <literal>:`."""
    counts = {"effective": 0, "trivial": 0, "dead": 0}

    def visit(node, dead):
        if isinstance(node, ast.Assert):
            counts["dead" if dead else "trivial" if trivial(node.test) else "effective"] += 1
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
            taken = bool(node.test.value)
            for c in node.body:
                visit(c, dead or not taken)
            for c in node.orelse:
                visit(c, dead or taken)
            return
        for c in ast.iter_child_nodes(node):
            visit(c, dead)
    visit(tree, False)
    return counts


def param_cases(fn, alias):
    """Product of literal @pytest.mark.parametrize case counts (1 when none), or None if any is not a literal."""
    total = 1
    for d in fn.decorator_list:
        if not (isinstance(d, ast.Call) and resolve(d.func, alias) == PARAMETRIZE):
            continue
        values = d.args[1] if len(d.args) > 1 else next((k.value for k in d.keywords if k.arg == "argvalues"), None)
        if not isinstance(values, (ast.List, ast.Tuple, ast.Set)):
            return None
        total *= len(values.elts)
    return total


def hooks(tree):
    """{hook name: [ast dump]} for module-level collection hooks."""
    out = {}
    for n in tree.body if tree else []:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in COLLECTION_HOOKS:
            out.setdefault(n.name, []).append(ast.dump(n))
    return out


def module_constants(tree, want):
    """{name: literal value or source text} for module-level assignments whose name passes want(name)."""
    out = {}
    for n in tree.body if tree else []:
        targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
        for t in targets:
            if isinstance(t, ast.Name) and want(t.id) and n.value is not None:
                try:
                    out[t.id] = ast.literal_eval(n.value)
                except ValueError:
                    out[t.id] = ast.unparse(n.value)
    return out


def golden_or_engine(name):
    return name == "ENGINE_VERSION" or name.startswith("GOLDEN")


def const_diff(name, old, new):
    if isinstance(old, dict) and isinstance(new, dict):
        d = {k: [old.get(k), new.get(k)] for k in sorted(set(old) | set(new), key=str) if old.get(k) != new.get(k)}
        return {"name": name, "changed_keys": d}
    return {"name": name, "old": old, "new": new}


def pytest_options(src, basename):
    """{key: value} of the pytest section, {} when the file or section is absent, None when unparseable."""
    if src is None:
        return {}
    try:
        if basename.endswith(".toml"):
            if tomllib is None:
                return None
            doc = tomllib.loads(src)
            table = doc.get("tool", {}).get("pytest", {}) if basename == "pyproject.toml" else doc.get("pytest", {})
            # pytest 9 reads native [tool.pytest] / [pytest] keys as well as the older ini_options table.
            opts = {**{k: v for k, v in table.items() if k != "ini_options"}, **table.get("ini_options", {})}
            return {k: json.dumps(v, sort_keys=True) for k, v in opts.items()}
        cp = configparser.ConfigParser(interpolation=None, strict=False)
        cp.read_string(src)
        for section in INI_SECTIONS.get(basename, ()):
            if cp.has_section(section):
                return {k: " ".join(v.split()) for k, v in cp.items(section)}
        return {}
    except Exception:  # noqa: BLE001 - any parse failure means "use the line check"
        return None


def config_weakening(where, old, new):
    """At most one "pytest config" entry per file: the collection keys that changed and the suspect diff lines."""
    basename = os.path.basename(where)
    keys, lines, structured = {}, [], False
    if basename != "conftest.py":
        o, n = pytest_options(old, basename), pytest_options(new, basename)
        if o is not None and n is not None:
            keys = {k: [o.get(k), n.get(k)] for k in COLLECTION_KEYS if o.get(k) != n.get(k)}
            structured = True
    for line in difflib.unified_diff((old or "").splitlines(), (new or "").splitlines(), lineterm="", n=0):
        if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
            continue
        if (line.startswith("+") and DESELECT.search(line)) or \
                (basename != "conftest.py" and not structured and COLLECTION_KEY_LINE.search(line)):
            lines.append(line)
    if not (keys or lines):
        return []
    return [{"kind": "pytest config", "path": where, "keys": keys, "lines": lines}]


def check(repo, base, head):
    diff_base = git(repo, "merge-base", base, head).strip()
    touched, golden, weak, renamed, new_skips = [], [], [], [], []
    for st, old_path, new_path in changes(repo, diff_base, head):
        for p in {old_path, new_path}:
            if protected(p) and p not in touched:
                touched.append(p)
        py = old_path.endswith(".py") or new_path.endswith(".py")
        if not (py or is_config(old_path) or is_config(new_path)):
            continue  # never read: images, binaries, docs
        old = show(repo, diff_base, old_path) if st != "A" else None
        new = show(repo, head, new_path) if st != "D" else None
        where = new_path if st != "D" else old_path
        if is_config(old_path) or is_config(new_path):
            weak += config_weakening(where, text(old), text(new))
        if not where.endswith(".py"):
            continue
        t_old, t_new = parse(old), parse(new)
        if t_old is not None and new is not None and t_new is None:
            weak.append({"kind": "unparseable after change", "path": where})
            continue
        if t_new is not None and os.path.basename(new_path) == "conftest.py":
            h_old, h_new = hooks(t_old), hooks(t_new)
            for k in sorted(h_new):
                if h_new[k] != h_old.get(k):
                    weak.append({"kind": "collection hook added or changed", "path": where, "hook": k})
        if t_new is not None:
            c_old, c_new = module_constants(t_old, golden_or_engine), module_constants(t_new, golden_or_engine)
            for k in sorted(set(c_old) | set(c_new)):
                if c_old.get(k) != c_new.get(k):
                    golden.append({"path": where, **const_diff(k, c_old.get(k), c_new.get(k))})
        if st == "D":
            if is_test_file(old_path):
                weak.append({"kind": "test file deleted", "path": old_path})
            continue
        if t_new is None:
            continue
        test_new = is_test_file(new_path)
        existing = st != "A" and t_old is not None
        if test_new:
            names_new = [n for n, _ in test_defs(t_new)]
            dup_old = {}
            if existing:
                for n, _ in test_defs(t_old):
                    dup_old[n] = dup_old.get(n, 0) + 1
            shadowed = sorted({n for n in names_new if names_new.count(n) > 1 and names_new.count(n) > dup_old.get(n, 0)})
            if shadowed:
                weak.append({"kind": "test shadowed by duplicate name", "path": where, "names": shadowed})
        if existing and is_test_file(old_path) and not test_new:
            weak.append({"kind": "test file renamed out of collection", "path": old_path, "to": new_path})
            continue
        old_names = test_names(t_old) if existing else set()
        new_names = test_names(t_new)
        gone, added = sorted(old_names - new_names), sorted(new_names - old_names)
        is_rename = False
        if existing and test_new and is_test_file(old_path):
            a_old, a_new = assert_counts(t_old), assert_counts(t_new)
            is_rename = bool(gone) and len(gone) == len(added) and a_new["effective"] >= a_old["effective"]
            if is_rename:
                renamed.append({"path": where, "from": gone, "to": added})
            elif gone:
                weak.append({"kind": "test function removed", "path": where, "names": gone})
            if a_new["effective"] < a_old["effective"]:
                weak.append({"kind": "asserts removed", "path": where, "before": a_old["effective"],
                             "after": a_new["effective"]})
            for k, kind in (("trivial", "trivial assert added"), ("dead", "dead assert added")):
                if a_new[k] > a_old[k]:
                    weak.append({"kind": kind, "path": where, "before": a_old[k], "after": a_new[k]})
            alias_old, alias_new = aliases(t_old), aliases(t_new)
            fns_old = dict(test_defs(t_old))
            for name, fn in test_defs(t_new):
                if name in fns_old:
                    p_old, p_new = param_cases(fns_old[name], alias_old), param_cases(fn, alias_new)
                    if p_old is not None and p_new is not None and p_new < p_old:
                        weak.append({"kind": "parametrize cases reduced", "path": where, "name": name,
                                     "before": p_old, "after": p_new})
        # Skips: count those that can reach a test that already existed.
        if test_new and not existing:
            fresh = len(skip_refs(t_new))
            if fresh:
                new_skips.append({"path": where, "count": fresh})
            continue
        if not test_new:
            r_old, r_new = skip_refs(t_old) if existing else [], skip_refs(t_new)
            reach_old, reach_new = len(r_old), len(r_new)
            fresh = 0
        else:
            keep = (old_names | new_names) if is_rename else (old_names & new_names)
            r_old, r_new = skip_refs(t_old), skip_refs(t_new)
            reach_old = sum(o is None or o in keep for o in r_old)
            reach_new = sum(o is None or o in keep for o in r_new)
            fresh = sum(o is not None and o not in keep and o not in old_names for o in r_new)
        if reach_new > reach_old:
            weak.append({"kind": "skip/xfail added", "path": where, "count": reach_new - reach_old})
        if fresh:
            new_skips.append({"path": where, "count": fresh})
    return {"diff_base": diff_base, "protected_paths_touched": touched,
            "golden_or_engine_changed": bool(golden), "golden_or_engine_detail": golden, "weakening": weak,
            "renamed": renamed, "new_test_skips": new_skips}


def labels(event_path):
    """Label names from the event payload; none when it is missing or unreadable (so no override)."""
    try:
        with open(event_path) as f:
            ev = json.load(f)
        return [lb.get("name") for lb in (ev.get("pull_request") or {}).get("labels") or []]
    except (TypeError, OSError, ValueError, AttributeError):
        return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--event", default=os.environ.get("GITHUB_EVENT_PATH"))
    ap.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    a = ap.parse_args(argv)
    res = {"base": a.base, "head": a.head, **check(a.repo, a.base, a.head)}
    res["override"] = OVERRIDE_LABEL in labels(a.event)
    res["result"] = "pass" if not res["weakening"] else "overridden" if res["override"] else "fail"
    out = json.dumps(res, indent=2, default=str)
    print(out)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write(f"## PR guard: {res['result']}\n\n```json\n{out}\n```\n")
    return 1 if res["result"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
