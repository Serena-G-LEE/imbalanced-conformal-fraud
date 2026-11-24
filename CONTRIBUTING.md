# Contributing

## Branching strategy

`main` holds only verified, reproducible states. All work happens on short-lived branches that are
merged back with `--no-ff`, so a feature boundary stays visible in history and can be reverted as a
unit.

| Prefix | Purpose | Examples |
| --- | --- | --- |
| `feat/` | new method or pipeline capability | `feat/conformal-triage-pipeline`, `feat/weighted-cp` |
| `exp/` | experiment runs and analysis on one dataset | `exp/ulb-baseline`, `exp/ieee-cis` |
| `fix/` | bug fixes | `fix/empty-stratum-quantile` |
| `docs/` | documentation only | `docs/branch-strategy` |
| `test/` | tests only | `test/quantile-edge-cases` |

Workflow:

1. Branch from an up-to-date `main`:

   ```bash
   git switch main && git pull --ff-only
   git switch -c feat/<scope>
   ```

2. Commit in Conventional Commits format and keep each commit one logical change.

3. Verify locally before opening the merge:

   ```bash
   python3 -m py_compile $(git ls-files '*.py')
   python3 -m unittest discover -s tests
   ```

4. Merge without squashing so the work stays attributable:

   ```bash
   git switch main
   git merge --no-ff feat/<scope>
   ```

   The branch is kept on the remote for traceability and deleted locally once merged.

5. Roll back a merged feature as a unit with `git revert -m 1 <merge-commit>`.

`exp/` branches carry config overrides and result summaries only; datasets and raw result files are
ignored by `.gitignore` and never enter version control.

## Commit messages

Conventional Commits, imperative mood, subject at most 72 characters, body explaining *what* and *why*
(not *how*):

```
feat(metrics): add class-conditional coverage report
test: cover the empty calibration stratum
docs: document branch strategy and reproduction steps
chore: pin minimum runtime dependency versions
```

## Reproducibility requirements

- Random seeds come from the config (`42 43 44 45 46`); a change to the seed set belongs in its own
  commit so the resulting tables stay comparable.
- Report results as mean ± standard deviation across seeds, and never commit data files, checkpoints or
  `results/` outputs.
- Any change to `src/conformal.py` must keep `python3 -m unittest discover -s tests` green, since the
  coverage tests encode the guarantee the repository claims.
