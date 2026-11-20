# Merge gate

Three signals on every PR, each checkable on the PR's head commit instead of taken on trust:

- `verify` (.github/workflows/verify.yml): `python verification/run.py --out DIR` with pinned z3-solver,
  pytest, hypothesis and jsonschema. Fails on a nonzero exit or a receipt whose status is not `pass`; the
  receipt is uploaded as the `verification-receipt` artifact. Scope and limits: verification/README.md.
- `guard` (.github/workflows/guard.yml): `scripts/pr_guard.py` diffs merge-base..head and prints JSON with
  `protected_paths_touched`, `golden_or_engine_changed`, `weakening`, `renamed` and `new_test_skips`. Weakening
  (the kinds are listed in the script's docstring: a deleted test file, a removed test function, fewer
  effective asserts, a trivial or dead-branch assert, an added skip/xfail reaching an existing test, a
  shadowed test name, fewer literal parametrize cases, a pytest collection-config change, a changed
  `pytest_collection_modifyitems`) fails the job unless the PR has the `guard-override` label. Protected paths
  and golden/engine changes are reported, not failed; the merge policy escalates on them.
- `review/adversarial` (commit status): after an independent review of a specific head SHA, the reviewer runs

      python scripts/review_status.py --pr N --sha HEAD_SHA --state success|failure --blocking K --summary "..." [--url URL]

  It refuses (exit 2) if HEAD_SHA is no longer the PR's head, or `success` with K > 0. `--dry-run` prints the
  `gh api` request without posting. A new push leaves the status behind on the old SHA, so the review must
  be redone for the new head.

Making these required checks is a branch-protection setting on GitHub; this repo only provides them.

## Who runs whose code

- `guard` runs on `pull_request_target`: GitHub takes the workflow file from the base branch, so a PR can't
  edit guard.yml or scripts/pr_guard.py to pass itself. The job checks out the base (`persist-credentials:
  false`, token `contents: read`, no secrets), fetches `refs/pull/N/head` as git objects only, checks the
  fetched head is the event's head SHA, and runs the base's `scripts/pr_guard.py`, which only runs git and
  parses PR content (ast.parse, configparser, tomllib); it never executes, imports or installs anything from
  the PR. SHAs and the PR number reach the shell through `env`, never `${{ }}` inside `run`. If the base has
  no scripts/pr_guard.py the job fails; it never falls back to the PR's copy. Consequence: the PR that first
  adds guard.yml and pr_guard.py gets no guard run, and a change to the guard only takes effect for PRs opened
  against a base that already contains it.
- `verify` and `test` necessarily run the PR's own code (`verification/run.py`, the tests, `asiimov/`), and
  run their own workflow file under `pull_request`. They have `contents: read` and no secrets, but their
  verdict on a PR is only as trustworthy as the PR's own copy of those files. So a PR that touches `.github/`
  or `verification/` (both reported in `protected_paths_touched`) requires escalation to the owner; a green
  `verify` on such a PR is not evidence on its own.
- The base the guard trusts is the PR's base branch. A PR aimed at a branch other than `main` is judged by
  that branch's guard.

## Known limits of pr_guard.py

It compares syntax, not behaviour. Weakenings it does not see:

- an early `return` (or any unconditional exit) placed before asserts, and asserts after it;
- asserts wrapped in `try: ... except: pass` (or `except AssertionError`), or `with pytest.raises(...)` around
  them, or `contextlib.suppress`;
- asserts moved into a helper (counted per file, so a helper in another file moves them out of view; a
  helper in the same file keeps the count but nothing checks it is still called);
- dynamic skips: `pytest.skip` reached through `getattr`, a variable, a fixture in another file that the
  guard can't tie to the test, `pytest.mark.skip` built at runtime, markers added by a plugin, `-m`/`-p`
  given on the command line in CI rather than in config, or a skip condition changed (`skipif(False)` to
  `skipif(True)`) without adding a reference;
- a constant condition that is not a literal (`if DEBUG:` with `DEBUG = False`, `if 1 > 2:`), and trivial
  asserts that are not literals (`assert x == x`, `assert not False`);
- parametrize cases given by a name or call rather than a literal list (not judged), and class- or
  module-level `pytestmark` parametrize;
- collection hooks other than `pytest_collection_modifyitems` (`pytest_ignore_collect`, `pytest_collect_file`,
  `pytest_deselected`), and any hook or plugin outside a conftest.py;
- pytest.toml / pyproject.toml are only parsed structurally on Python >= 3.11 (tomllib); on older Pythons
  they fall back to a line check that misses multi-line value changes;
- renames are matched by count only: N test functions removed and N added in one file with the effective
  assert count not lower is reported as `renamed`, whatever the new tests test; N removed and more than N
  added is still reported as a removal;
- skips in newly added tests are not weakening (they can't weaken a test that existed); they are listed in
  `new_test_skips` for the reviewer;
- a PR whose head isn't a descendant of a commit shared with the base (an unrelated history) fails at
  `git merge-base`.
