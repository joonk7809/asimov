"""scripts/pr_guard.py on throwaway git repos: weakening fails, renames and clean changes pass."""
import importlib.util
import json
import os
import subprocess

import pytest

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("pr_guard", os.path.join(HERE, "..", "scripts", "pr_guard.py"))
pr_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pr_guard)

TEST_A = '''import pytest


def test_one():
    assert 1 == 1
    assert 2 == 2


def test_two():
    assert 3 == 3
'''
SIM = 'ENGINE_VERSION = "35"\n\n\ndef step(x):\n    return x + 1\n'
WORLD = 'GOLDEN = {"34": "aa", "35": "bb"}\n\n\ndef test_golden():\n    assert GOLDEN\n'


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _write(repo, files):
    for path, text in files.items():
        p = repo / path
        if text is None:
            p.unlink()
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)


def _commit(repo, files, msg="c"):
    _write(repo, files)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q")
    base = _commit(r, {"tests/test_a.py": TEST_A, "asiimov/sim.py": SIM, "tests/test_world.py": WORLD,
                       "asiimov/other.py": "X = 1\n"})
    return r, base


def _run(r, base, head, tmp_path, label=None, capsys=None):
    ev = tmp_path / "event.json"
    ev.write_text(json.dumps({"pull_request": {"labels": [{"name": label}] if label else []}}))
    rc = pr_guard.main(["--base", base, "--head", head, "--repo", str(r), "--event", str(ev), "--summary", ""])
    out = json.loads(capsys.readouterr().out)
    return rc, out


def _kinds(out):
    return [w["kind"] for w in out["weakening"]]


def test_clean_change_passes(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"asiimov/other.py": "X = 2\n"})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["result"] == "pass"
    assert out["weakening"] == [] and out["protected_paths_touched"] == []
    assert out["golden_or_engine_changed"] is False


def test_identical_base_head_flags_nothing(repo, tmp_path, capsys):
    r, base = repo
    rc, out = _run(r, base, base, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == [] and out["protected_paths_touched"] == []


def test_planted_weakening_is_seen(repo, tmp_path, capsys):
    # Negative control: the same harness that passes above must see a real diff.
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("    assert 2 == 2\n", "")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["asserts removed"]
    assert out["protected_paths_touched"] == ["tests/test_a.py"]


def test_deleted_test_file_fails(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": None})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["test file deleted"]


def test_removed_test_function_fails(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("def test_two():\n    assert 3 == 3\n",
                                                         "def helper():\n    assert 3 == 3\n")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["test function removed"]
    assert out["weakening"][0]["names"] == ["test_two"]


def test_removed_asserts_fail(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("    assert 1 == 1\n", "    pass\n")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and out["weakening"][0]["before"] == 3 and out["weakening"][0]["after"] == 2


@pytest.mark.parametrize("deco", ["@pytest.mark.skip(reason='x')", "@pytest.mark.skipif(True, reason='x')",
                                  "@pytest.mark.xfail", "@unittest.skip('x')"])
def test_added_skip_or_xfail_fails(repo, tmp_path, capsys, deco):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": "import unittest\n" + TEST_A.replace("def test_two", deco + "\ndef test_two")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["skip/xfail added"]


@pytest.mark.parametrize("imp,use", [("import pytest as _pytest", "@_pytest.mark.skipif(True, reason='x')"),
                                     ("from pytest import mark", "@mark.xfail"),
                                     ("from unittest import skip", "@skip('x')")])
def test_aliased_skip_fails(repo, tmp_path, capsys, imp, use):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": imp + "\n" + TEST_A.replace("def test_two", use + "\ndef test_two")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["skip/xfail added"]


def test_skip_call_in_body_fails(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("    assert 3 == 3\n", "    pytest.skip('later')\n    assert 3 == 3\n")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["skip/xfail added"]


def test_pytest_config_deselect_fails(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"pytest.ini": "[pytest]\naddopts = --deselect tests/test_a.py::test_two\n"})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["pytest config"]


def test_rename_passes(repo, tmp_path, capsys):
    r, base = repo
    _git(r, "mv", "tests/test_a.py", "tests/test_moved.py")
    head = _commit(r, {})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == []
    assert set(out["protected_paths_touched"]) == {"tests/test_a.py", "tests/test_moved.py"}


def test_rename_out_of_collection_fails(repo, tmp_path, capsys):
    r, base = repo
    _git(r, "mv", "tests/test_a.py", "tests/a_helpers.py")
    head = _commit(r, {})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["test file renamed out of collection"]


def test_override_label_downgrades(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": None})
    rc, out = _run(r, base, head, tmp_path, label="guard-override", capsys=capsys)
    assert rc == 0 and out["result"] == "overridden" and _kinds(out) == ["test file deleted"]
    rc, out = _run(r, base, head, tmp_path, label="something-else", capsys=capsys)
    assert rc == 1 and out["result"] == "fail"


def test_golden_and_engine_change_detected_not_failed(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"asiimov/sim.py": SIM.replace('"35"', '"36"'),
                       "tests/test_world.py": WORLD.replace('"bb"}', '"bb", "36": "cc"}')})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == [] and out["golden_or_engine_changed"] is True
    detail = {d["name"]: d for d in out["golden_or_engine_detail"]}
    assert detail["ENGINE_VERSION"]["old"] == "35" and detail["ENGINE_VERSION"]["new"] == "36"
    assert detail["GOLDEN"]["changed_keys"] == {"36": [None, "cc"]}
    assert out["protected_paths_touched"] == ["tests/test_world.py"]


def test_diff_is_from_merge_base(repo, tmp_path, capsys):
    # A test deleted on the base branch after the PR forked is not the PR's doing.
    r, base = repo
    _git(r, "checkout", "-q", "-b", "pr")
    head = _commit(r, {"asiimov/other.py": "X = 3\n"})
    _git(r, "checkout", "-q", "-")
    moved_base = _commit(r, {"tests/test_a.py": None})
    rc, out = _run(r, moved_base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == [] and out["diff_base"] == base


# --- Binary and non-UTF-8 content (PR #76 added client/godot/art/*.png and crashed the first guard) ---

def _commit_bytes(repo, files, msg="c"):
    for path, data in files.items():
        p = repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe\x80"


def test_binary_files_are_named_not_read(repo, tmp_path, capsys):
    r, base = repo
    mid = _commit_bytes(r, {"client/art/a.png": PNG, "tests/data/b.bin": b"\xff\x00\x80"})
    head = _commit_bytes(r, {"client/art/a.png": PNG + b"\x99", "client/art/c.png": PNG})
    for b in (base, mid):
        rc, out = _run(r, b, head, tmp_path, capsys=capsys)
        assert rc == 0 and out["result"] == "pass" and out["weakening"] == []
    rc, out = _run(r, base, mid, tmp_path, capsys=capsys)
    assert rc == 0 and out["protected_paths_touched"] == ["tests/data/b.bin"]


def test_non_utf8_python_is_parsed_from_bytes(repo, tmp_path, capsys):
    # A latin-1 test file with a coding line parses; an undecodable one is "unparseable", never a crash.
    r, base = repo
    latin = b"# -*- coding: latin-1 -*-\n" + TEST_A.encode() + b"\ndef test_three():\n    assert '\xe9' == '\xe9'\n"
    head = _commit_bytes(r, {"tests/test_a.py": latin})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == []
    bad = _commit_bytes(r, {"tests/test_a.py": TEST_A.encode() + b"\nX = '\xff\xfe'\n"})
    rc, out = _run(r, base, bad, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["unparseable after change"]


def test_pr_content_is_never_executed(repo, tmp_path, capsys):
    r, base = repo
    marker = tmp_path / "ran"
    evil = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('x')\n"
    head = _commit(r, {"conftest.py": evil, "tests/test_evil.py": evil + "\ndef test_e():\n    assert 1 == 1\n",
                       "scripts/pr_guard.py": evil, "pyproject.toml": "[project]\nname = 'x'\n"})
    _run(r, base, head, tmp_path, capsys=capsys)
    assert not marker.exists()


def test_unreadable_event_means_no_override(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": None})
    for content in ("", "not json", "[]"):
        ev = tmp_path / "bad.json"
        ev.write_text(content)
        rc = pr_guard.main(["--base", base, "--head", head, "--repo", str(r), "--event", str(ev), "--summary", ""])
        assert rc == 1 and json.loads(capsys.readouterr().out)["override"] is False


# --- Renames: N test functions out, N in, effective asserts not down (PRs #96 #78 #71 #42 #33) ---

def test_renamed_test_functions_pass(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("def test_two", "def test_two_renamed")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == []
    assert out["renamed"] == [{"path": "tests/test_a.py", "from": ["test_two"], "to": ["test_two_renamed"]}]


def test_rename_that_drops_asserts_fails(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("def test_two():\n    assert 3 == 3\n",
                                                         "def test_two_renamed():\n    pass\n")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["test function removed", "asserts removed"] and out["renamed"] == []


def test_rename_with_fewer_new_names_fails(repo, tmp_path, capsys):
    # Two out, one in: not a rename, even with the asserts kept.
    r, base = repo
    merged = TEST_A.replace("def test_one():", "def test_both():").replace("\n\n\ndef test_two():\n", "\n")
    head = _commit(r, {"tests/test_a.py": merged})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["test function removed"]


# --- Asserts that can't fail ---

@pytest.mark.parametrize("new", ["assert True", "assert 1", "assert 'yes'", "assert 1 == 1 or True",
                                 "assert False or 1"])
def test_trivial_assert_replacing_a_real_one_fails(repo, tmp_path, capsys, new):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("assert 2 == 2", new)})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["asserts removed", "trivial assert added"]


@pytest.mark.parametrize("new", ["assert 2 == 2 or 3 == 3", "assert x or y", "assert not x"])
def test_non_literal_asserts_pass(repo, tmp_path, capsys, new):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("assert 2 == 2", new)})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == []


@pytest.mark.parametrize("wrap", ["if False:", "if 0:", "if None:", "if '':"])
def test_assert_under_dead_branch_fails(repo, tmp_path, capsys, wrap):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("    assert 2 == 2\n", f"    {wrap}\n        assert 2 == 2\n")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["asserts removed", "dead assert added"]


def test_assert_under_else_of_if_true_fails(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace(
        "    assert 2 == 2\n", "    if True:\n        pass\n    else:\n        assert 2 == 2\n")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["asserts removed", "dead assert added"]


@pytest.mark.parametrize("wrap", ["if True:", "if x:", "if 1:"])
def test_assert_under_live_branch_passes(repo, tmp_path, capsys, wrap):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("    assert 2 == 2\n", f"    {wrap}\n        assert 2 == 2\n")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == []


# --- Raised skips, and skips that can't reach an existing test ---

@pytest.mark.parametrize("imp,stmt", [("import unittest", "raise unittest.SkipTest('x')"),
                                      ("import pytest", "raise pytest.skip.Exception('x')"),
                                      ("from unittest import SkipTest", "raise SkipTest('x')")])
def test_raised_skip_fails(repo, tmp_path, capsys, imp, stmt):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": imp + "\n" + TEST_A.replace("    assert 3 == 3\n", f"    {stmt}\n    assert 3 == 3\n")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["skip/xfail added"]


def test_raised_other_exception_passes(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("    assert 3 == 3\n", "    if 3 != 3:\n        raise ValueError('x')\n    assert 3 == 3\n")})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == []


def test_skip_in_a_new_test_is_reported_not_failed(repo, tmp_path, capsys):
    r, base = repo
    new_test = "\n\n@pytest.mark.skipif(True, reason='x')\ndef test_new():\n    pytest.skip('no data')\n    assert 4 == 4\n"
    head = _commit(r, {"tests/test_a.py": TEST_A + new_test,
                       "tests/test_fresh.py": "import pytest\n\n\ndef test_f():\n    pytest.skip('x')\n"})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == []
    assert sorted(out["new_test_skips"], key=lambda d: d["path"]) == [{"path": "tests/test_a.py", "count": 2},
                                                                      {"path": "tests/test_fresh.py", "count": 1}]


def test_skip_in_a_helper_or_module_level_fails(repo, tmp_path, capsys):
    # A helper or module-level skip reaches existing tests, even in the same PR that adds a new test.
    r, base = repo
    for extra in ("\n\ndef _setup():\n    pytest.skip('x')\n", "\n\npytestmark = pytest.mark.skip\n"):
        head = _commit(r, {"tests/test_a.py": TEST_A + extra})
        rc, out = _run(r, base, head, tmp_path, capsys=capsys)
        assert rc == 1 and _kinds(out) == ["skip/xfail added"]


# --- Shadowed test names ---

def test_duplicate_test_name_fails(repo, tmp_path, capsys):
    r, base = repo
    head = _commit(r, {"tests/test_a.py": TEST_A + "\n\ndef test_one():\n    assert 5 == 5\n    assert 6 == 6\n"})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["test shadowed by duplicate name"] and out["weakening"][0]["names"] == ["test_one"]


def test_same_name_in_two_classes_passes(repo, tmp_path, capsys):
    r, base = repo
    classes = "\n\nclass TestA:\n    def test_x(self):\n        assert 1 == 1\n\n\nclass TestB:\n    def test_x(self):\n        assert 1 == 1\n"
    head = _commit(r, {"tests/test_a.py": TEST_A + classes})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == []


# --- Parametrize case counts ---

PARAM = '''import pytest


@pytest.mark.parametrize("x", [1, 2, 3])
@pytest.mark.parametrize("y", ["a", "b"])
def test_p(x, y):
    assert x
'''


def test_parametrize_cases_reduced_fails(repo, tmp_path, capsys):
    r, base = repo
    base = _commit(r, {"tests/test_p.py": PARAM})
    for new, after in ((PARAM.replace("[1, 2, 3]", "[1, 2]"), 4), (PARAM.replace('@pytest.mark.parametrize("y", ["a", "b"])\n', ""), 3)):
        head = _commit(r, {"tests/test_p.py": new})
        rc, out = _run(r, base, head, tmp_path, capsys=capsys)
        assert rc == 1 and _kinds(out) == ["parametrize cases reduced"]
        assert out["weakening"][0]["before"] == 6 and out["weakening"][0]["after"] == after


def test_parametrize_cases_kept_or_grown_passes(repo, tmp_path, capsys):
    r, base = repo
    base = _commit(r, {"tests/test_p.py": PARAM})
    for new in (PARAM.replace("[1, 2, 3]", "[1, 2, 4]"), PARAM.replace("[1, 2, 3]", "[1, 2, 3, 4]"),
                PARAM.replace("[1, 2, 3]", "CASES")):  # not a literal: not judged (docs/merge-gate.md)
        head = _commit(r, {"tests/test_p.py": new})
        rc, out = _run(r, base, head, tmp_path, capsys=capsys)
        assert rc == 0 and out["weakening"] == []


# --- Collection config ---

@pytest.mark.parametrize("path,before,after", [
    ("pytest.ini", "[pytest]\ntestpaths = tests\n", "[pytest]\ntestpaths = tests/fast\n"),
    ("pytest.ini", "[pytest]\ntestpaths =\n    tests\n    client\n", "[pytest]\ntestpaths =\n    tests\n"),
    ("tox.ini", "[pytest]\n", "[pytest]\npython_functions = check_*\n"),
    ("setup.cfg", "[tool:pytest]\nnorecursedirs = .git\n", "[tool:pytest]\nnorecursedirs = .git tests\n"),
    ("pyproject.toml", "[tool.pytest.ini_options]\npython_files = ['test_*.py']\n",
     "[tool.pytest.ini_options]\npython_files = ['test_a*.py']\n"),
    ("pyproject.toml", "[tool.pytest]\n", "[tool.pytest]\ntestpaths = ['x']\n"),
    ("pytest.ini", "[pytest]\n", "[pytest]\npython_classes = Nothing\n"),
])
def test_collection_config_change_fails(repo, tmp_path, capsys, path, before, after):
    r, base = repo
    base = _commit(r, {path: before})
    head = _commit(r, {path: after})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["pytest config"]


@pytest.mark.parametrize("path,before,after", [
    ("pytest.ini", "[pytest]\ntestpaths = tests\n", "[pytest]\ntestpaths = tests\nmarkers =\n    slow: slow\n"),
    ("pyproject.toml", "[project]\nname = 'a'\n", "[project]\nname = 'b'\n"),
    ("setup.cfg", "[metadata]\nname = a\n[tool:pytest]\ntestpaths = tests\n",
     "[metadata]\nname = b\n[tool:pytest]\ntestpaths = tests\n"),
])
def test_other_config_change_passes(repo, tmp_path, capsys, path, before, after):
    r, base = repo
    base = _commit(r, {path: before})
    head = _commit(r, {path: after})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == []


HOOK = "def pytest_collection_modifyitems(config, items):\n    items[:] = [i for i in items if 'slow' not in i.name]\n"


def test_conftest_collection_hook_added_or_changed_fails(repo, tmp_path, capsys):
    r, base = repo
    added = _commit(r, {"tests/conftest.py": HOOK})
    rc, out = _run(r, base, added, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["collection hook added or changed"]
    changed = _commit(r, {"tests/conftest.py": HOOK.replace("'slow'", "'s'")})
    rc, out = _run(r, added, changed, tmp_path, capsys=capsys)
    assert rc == 1 and _kinds(out) == ["collection hook added or changed"]


def test_conftest_other_change_passes(repo, tmp_path, capsys):
    r, base = repo
    base = _commit(r, {"tests/conftest.py": HOOK})
    head = _commit(r, {"tests/conftest.py": HOOK + "\n\nimport pytest\n\n\n@pytest.fixture\ndef world():\n    return 1\n"})
    rc, out = _run(r, base, head, tmp_path, capsys=capsys)
    assert rc == 0 and out["weakening"] == []
