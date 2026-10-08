# Contributing

Work on a branch and open a pull request into `main`. Keep each pull request focused, describe the scientific assumption it changes, and include a test or explain why no automated test applies.

Before opening a pull request, run:

```bash
PYTHONPATH=src:submission python -m unittest discover -s tests
python scripts/build_submission.py
python reference/starting_kit/tools/check_submission_zip.py dist/submission.zip
python reference/starting_kit/tools/run_smoke_test.py submission/
```

Do not commit credentials, private evaluation feedback, hidden outcomes, model caches, generated artifacts, or data whose license does not permit redistribution. Record new dependencies in `pyproject.toml`; submission-time dependencies also belong in `submission/requirements.txt` and must comply with organizer policy.

For experiments, commit the configuration and analysis code. Store raw outputs under `results/`, which Git ignores, and publish only reviewed summary tables or figures. Name the data snapshot and Git revision in every report.
