# Merge gate

Three signals on every PR, each checkable on the PR's head commit instead of taken on trust:

- `verify` (.github/workflows/verify.yml): `python verification/run.py --out DIR` with pinned z3-solver,
  pytest, hypothesis and jsonschema. Fails on a nonzero exit or a receipt whose status is not `pass`; the
  receipt is uploaded as the `verification-receipt` artifact. Scope and limits: verification/README.md.
- `guard` (.github/workflows/guard.yml): `scripts/pr_guard.py` diffs merge-base..head and prints JSON with
  `protected_paths_touched`, `golden_or_engine_changed` and `weakening`. Weakening (a deleted test file, a
  removed test function, fewer asserts, an added skip/xfail, a pytest config deselect) fails the job unless
  the PR has the `guard-override` label. Protected paths and golden/engine changes are reported, not failed;
  the merge policy escalates on them. The job runs the base branch's copy of the guard.
- `review/adversarial` (commit status): after an independent review of a specific head SHA, the reviewer runs

      python scripts/review_status.py --pr N --sha HEAD_SHA --state success|failure --blocking K --summary "..." [--url URL]

  It refuses (exit 2) if HEAD_SHA is no longer the PR's head, or `success` with K > 0. `--dry-run` prints the
  `gh api` request without posting. A new push leaves the status behind on the old SHA, so the review must
  be redone for the new head.

Making these required checks is a branch-protection setting on GitHub; this repo only provides them.
