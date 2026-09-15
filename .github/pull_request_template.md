## What changed

<!-- One or two sentences. What is different now, and why. -->

## How it was tested

<!-- Be specific, and be honest about what was NOT tested. -->

- [ ] `pytest` passes locally (152 tests, coverage floor 70% enforced by `pytest.ini`)
- [ ] Ran the API and hit the changed endpoint(s) by hand
- [ ] Not applicable / cannot test — explain below

## Risk

<!-- What could break? What is the blast radius? "None, it's a comment fix" is a valid answer. -->

## Checklist

- [ ] No large artefacts committed (`models/`, `data_raw/`, `clean_data/`, `*.joblib`, `*.csv` are gitignored)
- [ ] `IMPROVEMENTS.md` updated if this completes a tracked item
- [ ] Secrets are not hardcoded — config comes from env vars (`.env` is gitignored)
