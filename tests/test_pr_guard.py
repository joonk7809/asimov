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
    assert True
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
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("def test_two():\n    assert True\n",
                                                         "def helper():\n    assert True\n")})
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
    head = _commit(r, {"tests/test_a.py": TEST_A.replace("    assert True\n", "    pytest.skip('later')\n    assert True\n")})
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
