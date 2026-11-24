---
name: update-docs
description: Bring this repository's documentation (CHANGELOG.md, PROJECT_SUMMARY.md, CLAUDE.md, AGENTS.md, CONTRIBUTING.md, README.md and the MkDocs site in docs/) in line with the current code. Use before a commit, after any change to code, commands, pipelines, notebooks or conventions, or when the user asks to update the docs.
argument-hint: "[optional: what changed, or a git range]"
---

# Update the documentation

Make the documentation of this repository match the code as it is now. Edit the files directly; do not ask for confirmation first and do not commit.

$ARGUMENTS

## 1. Find out what changed

Run these and read the results before writing anything:

- `git status --short` and `git diff HEAD` for uncommitted work (include untracked files; read new files in full).
- `git log --oneline -15`, and `git diff <last documented commit>..HEAD` if commits landed since the newest `CHANGELOG.md` entry.
- The current signatures and constants, straight from the source: `grep -n "^def \|^[A-Z_]* = " cmb_lensing/*.py`, and the `add_argument` lines of any command line interface that changed. Names and defaults in this repository change often, so never write one from memory.
- The settings of any HPC driver that changed: the `name=${name:-default}` lines of the `.sh` driver and the positional arguments of its job script.
- The executed notebooks in `docs/` if they changed: read their cells and stored outputs with `nbformat`, since their markdown cells quote numbers from the outputs.

## 2. Update each file

Only touch a file when something in it is wrong or missing. Keep the existing structure and tone of each file.

- **`CHANGELOG.md`** (Keep a Changelog format, everything under `## [Unreleased]`, newest first). Add one entry per user-visible change, starting with the date it landed in bold (`- **YYYY-MM-DD** ...`), under `### Added`, `### Changed`, `### Removed` or `### Fixed`. Say what changed and where, naming the functions and files. Correct an existing entry of the same unreleased work when it has become false (a renamed function, a changed default) instead of stacking a contradiction on top. Do not log pure refactors nobody would notice, or edits to these docs themselves beyond a new file or page.
- **`PROJECT_SUMMARY.md`** (the human-readable description of what the code base and the repository do). Check every section against the code: what the package does, the walk through the pipeline, the sampler's steps, the forecast, the measurement pipelines, the repository layout, the data, the tests and the list of what is not validated. It is written for a person reading top to bottom: plain prose, terms defined on first use, no change history, and less detail than `CLAUDE.md`. Add to it when a new module, pipeline or entry point appears; do not copy agent-level detail into it.
- **`CLAUDE.md`** (the detailed briefing and working rules for Claude Code). Update the commands, the repository layout, the architecture sections, "Removed features" (when something is deleted, record what it was, how it worked and the last commit that has it), "Local data", "Known issues" (delete entries that are fixed) and the code style or operating rules when a new standing preference appeared. Date new facts and measurements in absolute form.
- **`AGENTS.md`** is the same briefing for other coding agents. After editing `CLAUDE.md`, make `AGENTS.md` identical to it except for its first heading (`# AGENTS.md`) and the agent named in its opening sentence. `diff AGENTS.md CLAUDE.md` must show those two lines only.
- **`CONTRIBUTING.md`** and **`README.md`**. Update the install steps, commands, the "What it does" list, the repository layout table, the description of the tests and of the data when they change.
- **The documentation site** (`docs/`, navigation in `mkdocs.yml`). The pages are a user-facing digest of `CLAUDE.md`; update the page that matches each change:
  - `docs/index.md` and `docs/getting-started/` (`installation.md`: install, the 5D CAMB grid, HPC setup; `data.md`: the Hugging Face dataset and the link to the results website; `unit-tests.md`);
  - `docs/concepts/basic-concepts.md` (fields, operators, `load_sim`);
  - `docs/pipelines/*.md`, one page per HPC pipeline: its commands, driver settings, outputs and merge step;
  - `docs/api/*.md`: the text above each mkdocstrings block. The reference itself is generated from the Google-style docstrings of `load_sim`, `map_joint`, `sample_joint`, `fisher_forecast.forecast`, `chain_analysis` (`main`, `load_chains`, `get_avg_covariance_matrix`, `get_forecasted_covariance`) and the field / operator / data set classes, so a changed argument is documented by updating that docstring (the one source-code edit this skill makes);
  - `mkdocs.yml` when a page is added, removed or renamed;
  - `experimental_results/README.md` and `experimental_results/phi_noise/README.md` when the layout of the results folders or the results website changes.
  - Finish with `mkdocs build --strict` (in `~/venvs/myenv`) and fix every warning it reports.

## 3. Rules

- **Never edit `CODE_REVIEW_FINDINGS.md`** as part of this skill, even when a finding listed there was fixed. Mention it in the report instead.
- Every number, name, default and file path written into the docs must be checked against the code, a driver script or a stored notebook output in this session. If something could not be checked, leave it out or say it is unverified.
- State what has not been validated plainly (for example "never run on the cluster", "nside 64 only"). Do not describe code as working because it exists.
- Follow the repository's writing conventions: dates in absolute form, quantities spelled out (`<|phi_mean|^2>`, not a letter shorthand), comments in code blocks without a space after `#`.
- Do not edit source code beyond the docstrings named above, and do not edit or re-execute the notebooks. If the docs cannot be made true without one of those (stale notebook commentary, a broken example), report it instead.
- Do not edit the files of the results website (`docs/results/`, `experimental_results/build_results_site.py`), the `SUMMARY.md` files of the data sets or `paper/`: they are content, not documentation of the code.

## 4. Report

Finish with a short summary: which files changed and the main points of each, the result of `mkdocs build --strict`, anything found that is out of date but outside these files (including `CODE_REVIEW_FINDINGS.md` and notebook text), and anything left unverified.
