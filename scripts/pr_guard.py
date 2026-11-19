"""PR guard: report what a change touches that the merge policy cares about, and fail on weakened tests.

    python scripts/pr_guard.py --base BASE_SHA --head HEAD_SHA [--repo DIR]

Diffs merge-base(BASE, HEAD)..HEAD with rename detection (git diff -M), so a moved test is not a deletion and
commits that landed on the base branch after the PR forked are not counted against it. Prints a JSON summary
(also appended to $GITHUB_STEP_SUMMARY when set):

- protected_paths_touched: tests/, verification/, .github/, AGENTS.md, scripts/pr_guard.py
- golden_or_engine_changed (+ golden_or_engine_detail): ENGINE_VERSION (asiimov/sim.py) or a module-level
  GOLDEN* constant (tests/test_world.py) changed
- weakening: a deleted test file (or one renamed out of pytest's test_*.py / *_test.py pattern), a removed
  test function, fewer `assert` statements in a test file (counted with ast), a newly added skip/xfail
  (pytest.mark.skip/skipif/xfail, pytest.skip/xfail/importorskip, unittest.skip*/expectedFailure), or a
  pytest config change that touches addopts or adds --deselect, -k not, --ignore or collect_ignore.

Exit 1 when there is weakening, unless the PR carries the label `guard-override` (read from the event payload
at $GITHUB_EVENT_PATH, no network); then it is still reported and the exit is 0. Protected paths and
golden/engine changes alone never fail: they are reported for the merge policy to escalate.
"""
import argparse
import ast
import difflib
import json
import os
import re
import subprocess
import sys

PROTECTED_DIRS = ("tests/", "verification/", ".github/")
PROTECTED_FILES = ("AGENTS.md", "scripts/pr_guard.py")
OVERRIDE_LABEL = "guard-override"
SKIPS = {"pytest.mark.skip", "pytest.mark.skipif", "pytest.mark.xfail", "pytest.skip", "pytest.xfail",
         "pytest.importorskip", "unittest.skip", "unittest.skipIf", "unittest.skipUnless",
         "unittest.expectedFailure"}
PYTEST_CONFIG = ("pytest.ini", "setup.cfg", "tox.ini", "pyproject.toml", "conftest.py")
DESELECT = re.compile(r"--deselect|-k\s*[\"']?\s*not\b|--ignore|collect_ignore")


def git(repo, *args):
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.strip()}")
    return p.stdout


def show(repo, sha, path):
    """File text at sha, or None when absent."""
    p = subprocess.run(["git", "-C", repo, "show", f"{sha}:{path}"], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


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


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def parse(text):
    try:
        return ast.parse(text or "")
    except SyntaxError:
        return None


def test_names(tree):
    """Qualified names of test functions: test_x, or Class.test_x for methods."""
    names = set()

    def walk(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
                names.add(prefix + n.name)
            elif isinstance(n, ast.ClassDef):
                walk(n.body, prefix + n.name + ".")
    walk(tree.body, "")
    return names


def count_asserts(tree):
    return sum(isinstance(n, ast.Assert) for n in ast.walk(tree))


def count_skips(tree):
    """Skip/xfail references, with import aliases resolved (import pytest as _pytest; from pytest import mark)."""
    alias = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            alias.update({a.asname or a.name: a.name for a in n.names})
        elif isinstance(n, ast.ImportFrom) and n.module:
            alias.update({a.asname or a.name: f"{n.module}.{a.name}" for a in n.names})
    count = 0
    for n in ast.walk(tree):
        if isinstance(n, (ast.Attribute, ast.Name)) and isinstance(getattr(n, "ctx", None), ast.Load):
            name = dotted(n)
            if name:
                root, _, rest = name.partition(".")
                full = alias.get(root, root) + ("." + rest if rest else "")
                count += full in SKIPS
    return count


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


def check(repo, base, head):
    diff_base = git(repo, "merge-base", base, head).strip()
    touched, golden, weak = [], [], []
    for st, old_path, new_path in changes(repo, diff_base, head):
        for p in {old_path, new_path}:
            if protected(p) and p not in touched:
                touched.append(p)
        old = show(repo, diff_base, old_path) if st != "A" else None
        new = show(repo, head, new_path) if st != "D" else None
        if not (old_path.endswith(".py") or new_path.endswith(".py")) and \
                os.path.basename(new_path) not in PYTEST_CONFIG:
            continue
        where = new_path if st != "D" else old_path
        if os.path.basename(new_path) in PYTEST_CONFIG or os.path.basename(old_path) in PYTEST_CONFIG:
            for line in difflib.unified_diff((old or "").splitlines(), (new or "").splitlines(), lineterm="", n=0):
                if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
                    continue
                if "addopts" in line or (line.startswith("+") and DESELECT.search(line)):
                    weak.append({"kind": "pytest config", "path": where, "line": line})
        if not where.endswith(".py"):
            continue
        t_old, t_new = parse(old), parse(new)
        if old is not None and new is not None and old_path.endswith(".py") and t_new is None:
            weak.append({"kind": "unparseable after change", "path": where})
            continue
        if t_old is not None and t_new is not None:
            c_old, c_new = module_constants(t_old, golden_or_engine), module_constants(t_new, golden_or_engine)
            for k in sorted(set(c_old) | set(c_new)):
                if c_old.get(k) != c_new.get(k):
                    golden.append({"path": where, **const_diff(k, c_old.get(k), c_new.get(k))})
            added = count_skips(t_new) - count_skips(t_old)
            if added > 0:
                weak.append({"kind": "skip/xfail added", "path": where, "count": added})
        if not is_test_file(old_path) or st == "A":
            continue
        if st == "D":
            weak.append({"kind": "test file deleted", "path": old_path})
            continue
        if not is_test_file(new_path):
            weak.append({"kind": "test file renamed out of collection", "path": old_path, "to": new_path})
            continue
        if t_old is None:
            continue
        gone = sorted(test_names(t_old) - test_names(t_new))
        if gone:
            weak.append({"kind": "test function removed", "path": where, "names": gone})
        a_old, a_new = count_asserts(t_old), count_asserts(t_new)
        if a_new < a_old:
            weak.append({"kind": "asserts removed", "path": where, "before": a_old, "after": a_new})
    return {"diff_base": diff_base, "protected_paths_touched": touched,
            "golden_or_engine_changed": bool(golden), "golden_or_engine_detail": golden, "weakening": weak}


def labels(event_path):
    if not event_path or not os.path.exists(event_path):
        return []
    with open(event_path) as f:
        ev = json.load(f)
    return [lb.get("name") for lb in (ev.get("pull_request") or {}).get("labels") or []]


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
    text = json.dumps(res, indent=2, default=str)
    print(text)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write(f"## PR guard: {res['result']}\n\n```json\n{text}\n```\n")
    return 1 if res["result"] == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
