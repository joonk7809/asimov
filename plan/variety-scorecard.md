# Variety scorecard: preregistered pass thresholds

Date: 2026-11-03. Implements the six tests in the shared folder's `plan/agent-individuality.md`.
The numbers live in `plan/variety-thresholds.json` (single source; the code reads it and the report
records its sha256). They were committed before any real-model run was graded. Changing a number
after seeing real-model results voids that test: bump `version` and log the reason here.

Run: `python -m asiimov.scorecard --mind stub --out runs/scorecard-stub` (or `--mind llm --pod ID`).
It runs three arms on the same seed and population and writes `report.md` and `report.json`.
`--grade-only` re-grades arm logs already in `--out` without calling a model.

## Protocol
- 100 agents, 7 sim-days, seed 1, a player fire on family 3 on day 1 (the standard demo world).
- Arms: **full**; **no_text** (soul and core blanked in every prompt, params kept);
  **flat_params** (disposition params set to the population median: traits, rhythm, body, need
  weights, cognition; each prompt still carries the agent's original soul). Circumstance (role,
  family, age, starting wealth) is kept in every arm. All arms give agents names from two gendered
  pools assigned at random, independent of params (needed for test 6).
- Unit of observation: an agent's waking period (wake to sleep). The first period is skipped as
  warm-up; periods cut off by the end of the run are dropped.
- Graded behavior features per period (all mind-influenced): hours at work, socializing and
  exploring; distinct places visited; distinct people talked with. Wake time is reported but not
  graded in test 1: it is set by code (chronotype + jitter), not by the mind.

## Tests and thresholds

| # | Test | Pass when | Why this number |
|---|---|---|---|
| 1 | Stable within, different between | Median ICC(1) over the 5 graded features is in [0.40, 0.90] and no feature is below 0.10 | Day-to-day behavior in diary studies typically gives ICCs of 0.3 to 0.6. Below 0.4 is mostly noise; above 0.9 is clockwork (no two days should be identical). Below 0.1 a feature carries no character at all. |
| 2 | Trait predicts behavior | Every testable trait has at least one preregistered behavior at Spearman rho >= 0.20 in the predicted direction, and none at rho <= -0.20 in the wrong direction | Real trait-to-single-behavior correlations sit around 0.1 to 0.3; with n of about 100, rho 0.20 is about p 0.05 two-sided. Pairs: extraversion with social hours and talks started; honesty against steal rate (agents who deliberated); conscientiousness with work hours (working roles); openness with explore hours; chronotype with wake time (a wiring check). Testable needs at least 20 agents, and for steal rate at least 5 steal decisions. Emotionality and agreeableness have no mind-chosen behavior in P0, so they're listed as untested, not passed. |
| 3 | Soul ablation | Text contribution >= 0.10, where text contribution = 1 - mean over features of (between-agent SD of agent means in no_text / same in full) | If blanking soul and core cuts less than 10% of between-agent spread, the markdown layer isn't earning its tokens. Params contribution is computed the same way from flat_params and reported (no threshold: code-driven variety is expected). |
| 4 | No lockstep | Over daytime hours 08 to 20 with at least 10 awake: top-activity share never above 0.85, mean top share at most 0.65, mean normalized activity entropy at least 0.50 | X = 85% is roughly every working-age agent doing one thing at once. A realistic midday (70% work, rest spread) gives entropy about 0.56 and top share 0.7; mornings and evenings are more mixed, so the mean should sit lower. Synchronized hour-mark transitions are reported as a diagnostic only: plans are whole hours by schema, so it would fail for every mind. |
| 5 | No homogenization | (a) Mean cross-agent text similarity in full is at most 0.85 of that in no_text; (b) it doesn't rise over the run: last-third daily mean minus first-third at most +0.05 | The no_text arm is the "single generic persona" reference: same world, no soul. Texts are each agent's free text per period (deliberation reasons, reflection summaries and core memories); P0 talk has no words yet. Similarity is cosine over word unigram and bigram counts with numbers and agent names removed. That's a lexical proxy: it catches template collapse, not paraphrased sameness. An embedding model can replace it later; that would be a version bump. Needs at least 20 agents with text per day. |
| 6 | No demographic bias | No feature has both \|Cohen's d\| >= 0.30 between name-gender groups and permutation p < 0.05 / 7 (2000 label permutations, seeded) | Labels are randomized, so under no bias any gap is chance, which the permutation test measures. d 0.3 is a small-to-medium effect worth acting on. Features: the 5 graded ones, deliberations and steal decisions per period. Computed on the full arm. |

Overall verdict: pass only if all six pass. A test that can't be computed (missing arm, too little
data) is "inconclusive", which is not a pass. Per the plan, the cost-ladder tier to build on is the
cheapest one that passes tests 1 to 5.

## Stand-in mind result (validation, not a target)
See the PR that added this file for the `StubMind` report. The stub reads params directly and
ignores prompt text, so it is expected to fail test 3 (text contribution 0) and test 5 (templated
text). Those failures show the tests can fail for the right reason.

## Change log
- v1 (2026-11-03): initial thresholds.
