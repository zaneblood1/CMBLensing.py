# Code review findings

Two read-only reviews of the whole repository, both without executing any code:

- **First review, 2026-11-11**, of commit `b50d4cc`.
- **Second review, 2026-11-11**, of the working tree after that day's refactors (uncommitted on top of `b50d4cc`): `chain_analysis.py` moved into the package with a CLI; `fisher_forecast.py` narrowed to the measured-delensing forecast and extended, with its three input pipelines, to pol `I` / `P` / `IP`; shared helpers moved to `camb_spectra.py` / `matrix_plots.py`; the 1D CAMB splines and the rotated-coordinate sampler removed.

This document is the state after the second review. Nothing listed as open has been fixed.

## How to read an entry

- **Status.** Sections 2 and 3 list what was **fixed** and what is **new**. Everything from section 4 on is **open**: either carried over from the first review or new (new items are marked **NEW**).
- **Effort.** Every area is split into **Quick fixes** (a few lines in one place, no design decision, no new runs) and **Involved fixes** (a design decision, several files, or new runs / validation). An item that has both a quick stop-gap and an involved proper fix says so.
- **Severity:** `bug` (wrong today), `latent bug` (wrong as soon as a currently unused path is used), `fragile`, `design`, `cleanup`.
- **Confidence** is the reviewer's (high / medium / low). ✔ marks an entry re-read against the source a second time.
- **Line numbers** are current for the files changed by the refactors (`sample_lcdm.py`, `fisher_forecast.py`, `camb_spectra.py`, `delensed_covariance.py`, `chain_analysis.py`, the phi-noise modules and their scripts). For files not touched since `b50d4cc` they are the first review's and still valid.

Delete entries as they are fixed.

## Contents

1. [Where things stand](#1-where-things-stand)
2. [Fixed since the first review](#2-fixed-since-the-first-review)
3. [New since the first review](#3-new-since-the-first-review)
4. [Sampler](#4-sampler)
5. [Core numerics](#5-core-numerics)
6. [CAMB grid](#6-camb-grid)
7. [Forecast and measurement modules](#7-forecast-and-measurement-modules)
8. [Chain analysis](#8-chain-analysis)
9. [MCMC diagnostics](#9-mcmc-diagnostics)
10. [HPC drivers and job scripts](#10-hpc-drivers-and-job-scripts)
11. [Runtime comparison](#11-runtime-comparison)
12. [Tests](#12-tests)
13. [Packaging, CI and repository hygiene](#13-packaging-ci-and-repository-hygiene)
14. [Documentation drift](#14-documentation-drift)
15. [Dead code, stale comments, magic numbers](#15-dead-code-stale-comments-magic-numbers)
16. [Checked and found correct](#16-checked-and-found-correct)

---

## 1. Where things stand

**Before committing the refactor** (quick): four new files are untracked and not ignored, and a commit without them breaks the package or the docs build: `cmb_lensing/camb_spectra.py`, `cmb_lensing/matrix_plots.py`, `docs/api/chain-analysis.md`, `docs/chain_analysis_tutorial.ipynb`. This file is untracked too and is referenced from `CLAUDE.md`.

### The quick fixes most worth doing

| Problem | Where | Entry |
|---|---|---|
| Pol "P" Python timing jobs crash on a wrong field name | `runtime_comparison/python_performance_test.py:181-189` | 11.1 |
| Out-of-bounds Metropolis rejections are not counted in the printed acceptance rate | `sample_lcdm.py:263-264` | 4.1 |
| `sample_joint` mutates the caller's `param_init` | `sample_lcdm.py:712` | 4.2 |
| Chain histories are appended (stop-gap: refuse or truncate an existing file) | `sample_lcdm.py:1057-1058` | 4.3 |
| The chain seed of an HPC chain is never recorded | `run_single_lcdm_chain.py:111` | 4.5 |
| `primal_log_det` returns a finite wrong value for a negative entry | `util.py:39-42` | 5.1 |
| Q and U swapped in the lensing adjoint's field cotangent | `lense_flow.py:140-152` | 5.2 |
| Merged sampler phi-noise files do not record which estimator made them | `sampler_phi_noise.py:291-299` | 7.1 |
| Forecast outputs with and without BB, or from different phi-noise files, overwrite each other | `fisher_forecast.py:99-101` | 3.6 |
| One straggling chain truncates a whole map in the sampler phi-noise merge | `sampler_phi_noise.py:255` | 7.2 |
| `CITATION.cff` is invalid | `CITATION.cff` | 13.1 |
| Tighten the test tolerances from 100 % to a multiple of the measured error | `tests/test_*.py` | 12.1 |

### The involved fixes most worth doing

| Problem | Where | Entry |
|---|---|---|
| A negative measured N_phi means three different things to its three consumers | forecast, delensed covariance, phi-noise plots | 3.1 |
| The `IP` forecast's default N_phi is the EB estimator's, about 80× the TT one on the test box | `camb_spectra.qe_noise_matrix` | 3.2 |
| Resuming a chain is not a continuation, and no script can resume | `sample_lcdm.py`, `run_single_lcdm_chain.py` | 4.9 |
| The product-of-KDEs width is biased wide by a few percent, the size of the quoted errors | `chain_analysis.py:339` | 8.4 |
| `load_sim` turns any CAMB or programming error into silent NaN spectra | `simulate.py:511-530` | 5.9 |
| The 5D grid merge has no checks: one failed CAMB node NaNs a spectrum table | `merge_camb_grid.py` | 6.3 |
| Drivers fail silently (no `set -e`, unknown `--name=value` accepted, unbounded retry) | all drivers, `hpc_submit.sh` | 10.6-10.8 |
| Nothing tests the sampler, forecast, measurement modules or diagnostics | `tests/` | 12.9 |
| The polarization forecast path has no chain or test behind it | forecast + three pipelines | 3.9 |
| Non-trivial masks and beams (a goal, once the rest is settled): the beam is counted twice in the QE noise, and Python has never been compared with Julia for either | `simulate.py`, `camb_spectra.py`, `tests/` | 3.4, 12.11 |

---

## 2. Fixed since the first review

Removed with their code, or corrected by the refactors. Old entry numbers are the first review's.

**Sampler**
- (old 2.1) `reparameterize_lcdm` read a forecast file name the forecast never wrote: the rotated-coordinate sampler is removed.
- (old 2.16, part) The `USE_CAMB_GRID` / `USE_CAMB_SPLINE` module switches, the import-time `raise` and the import of a script by file path are gone; the 5D grid is the only Cl source.
- (old 13) The duplicated sampled-names list and duplicated `USE_CAMB_GRID` check; the rotated branch falling through into the unrotated loop.

**CAMB grid and splines**
- (old 4.1, 4.5) The 1D spline accuracy check with an extrapolated node, and `precompute_camb_1d.py`'s housekeeping issues: the script is deleted.

**Forecast and measurement**
- (second review's 3.3) `delensed_covariance.DEFAULT_STEP_SIGMA` was 0.5 while the driver passes 0.05: both are 0.05 now, with a comment saying the value is only checked at nside 128 / 2.5' / 2.5-10 uK and how to re-check it.
- (old 5.4) The delensed-covariance merge recomputed the steps from today's `PARAM_SIGMA`: it now copies the steps the jobs stored (`delensed_covariance.py:349`).
- (old 5.7) Three lmax conventions: the forecast, `chain_analysis`, the delensed covariance and the phi-noise plots all default to `CAMB_LMAX`.
- (old 5.10, three bullets) The wasted CAMB delensing under a measured T block, the `--stability` cost, and `fisher_forecast.py` doing three jobs (helpers now in `camb_spectra.py`, `matrix_plots.py`; `fixed_theta_chain.py` no longer imports the forecast).
- The forecast and the three measurement pipelines were temperature-only: all four now take `pol`.
- Four hand-maintained parameter orderings: everything is in `constants.PARAM_ORDER`.

**Chain analysis** (rewritten)
- `chain[0:-1:stride]` dropped the final sample; a map with one surviving sub-chain was dropped from everything; a missing history file raised only when `verbose`; unequal-length files were truncated after concatenation, shifting sub-chains against each other; `ddof` mismatch in the std of widths; rectangle vs trapezoid rule; absolute grid pad; `120(2)` printed for ±20; disabled R-hat pruning; hardcoded `__main__` paths; `int(nan)` for the IAT of a stuck chain.

**Documentation drift and dead code**
- `_check_centre` / `same_cosmology`, `load_stored_nphi`'s claimed cosmology check, the `--nphi_source` default, the pre-rename script names, the "~2.3 GB" comment, the delensed driver's `qe_filter` default, "args 32-42" attribution.
- `band_average`, `leave_one_out_grids`, unused `verbose` parameters, unnecessary `allow_pickle`, `sigma_scale = 2.4`, `sweep % 50`, `map_joint_steps = 30`, the forecast's unnamed constants, stale `DEFAULT_DELTA_ELL` and `fixed_theta_chain` docstring comments.

---

## 3. New since the first review

Introduced, exposed or left behind by the refactors.

### Quick fixes

**3.5 The QE norm stores "no response" as 0** — fragile, medium. `simulate.py:406, 421`: `jnp.nan_to_num(1 / qe_sum, nan=0, posinf=0, neginf=0)`. The sampler's `pinv` reads 0 as no information, but the forecast (`fisher_forecast.py:119`) reads N_phi = 0 as a perfect measurement and `delensed_covariance.py:233-240` gets W = 1. Harmless on the current boxes (only the origin is hit, where C_phi = 0); it bites on a box whose corner exceeds `l_cutoff`, or with a real mask. `qe_noise_matrix` should map 0 to NaN away from the origin. This becomes live as soon as non-trivial masks are supported (3.4).

**3.6 Forecast outputs still overwrite each other** — design, high. `fisher_forecast.py:99-101`: the suffix `_{pol}_{nphi_label}_{noise:g}_uk` omits `include_bb`, so `--no_bb` overwrites the with-BB files, and every `--phi_noise` file is labelled `measured_nphi`, so a map_joint and a sampler forecast overwrite each other. It also still omits `--params`, nside, theta_pix and the delensed covariance file. The parameter `nphi_label` shadows the function of the same name defined just above.

**3.7 The merged delensed covariance cannot say which N_phi built its W** — design, high. `delensed_covariance.py:250-252, 269-283, 357-359`: `wiener_nphi_file` is written by each job but is not in `_read_config` and is dropped from the merged file, so jobs built from different N_phi files merge together (`CLAUDE.md` says "the merge refuses to mix"). `qe_filter` is recorded, and required to match, even when W came from a file and the filter is meaningless.

**3.8 Smaller items** — cleanup / fragile, high unless noted.
- `chain_analysis.py:316`: the bootstrap's docstring says it is valid "because the chains are thinned", but with `--no_thin` it runs on autocorrelated samples and its errors are too small by about sqrt(IAT). `stride = max(1, int(iat))` (line 228) floors, so an IAT of 1.9 gives no thinning.
- `phi_noise_analysis.py:70-83`: `qe_noise_matrices` never passes `pol`, so the QE reference is always the TT estimator even though merged files now carry `pol`.
- `delensed_covariance.py:180-197`: `load_stored_nphi` checks the box but not `pol`; a pol "I" N_phi file feeds an "IP" run silently. (Consistent with the "no checks" rule, but the box check next to it is the same kind of check.)
- `fixed_theta_chain.py:54, 82-87`: `posterior_mixed_draws` still takes four operators that `mixed_theta_matrices` now reads off the data set, then `replace`s them back in; its `on_draw is None` branch has no caller; `sampler_phi_noise.py:173` recomputes `pinv(G) * mixed_phi`, which the sweep's `unmix` has just produced.
- `camb_spectra.py:52-64`: `camb_cls_at_params` re-implements the body of `simulate._run_camb`; calling `_run_camb` would stop the two drifting. `load_sim_cosmology` duplicates `sample_lcdm.to_camb_naming_conv` and the hand-built dict in `sample_lcdm.__main__`.
- `delensed_covariance.py:146-156`: `if second_fourier is first_fourier:` uses object identity to choose `abs()**2` over `Re(x conj x)` for autos (kept for bit-identity with old files; low confidence that it matters).
- `delensed_covariance.py:77-78`: a custom merge `--out_name` such as `delensed_covariance_v2.npz` matches `REALIZATION_GLOB` and is picked up as a job file by the next merge. Same for `map_joint_phi_noise_*.npz`.
- The "delensed_covariance is None" refusal is written four times (`fisher_forecast.py:159-161, 389-390`, `chain_analysis.py:847-849, 927-928`); the forecast CLI uses `default = None` plus `parser.error` where `required = True` would do.
- `chain_analysis.py:592`: the colourbar of the covariance heat map is labelled "Value" but shows the correlation-normalized matrix. `matrix_plots.py:66` saves with `plt.savefig` instead of `fig.savefig`.
- `chain_analysis.py:611`: `triangle_plot`'s `sigmas = (1,)` is never passed by any caller.
- `chain_analysis.py:142`: an empty or wrong `chain_dir` raises an opaque `TypeError` from `set.intersection(*[])`.
- Guard rails the house style says to strip survive in the refactored files: `check_box`, the raises in `load_stored_nphi`, the shape `ValueError` in `sampler_phi_noise.store`.
- Your local, gitignored `cmb_lensing/debug_fisher_covar_vs_cls.py` calls the removed forecast API and imports `chain_analysis` by its old path; it cannot run.

### Involved fixes

**3.1 A negative measured N_phi has three meanings** — likely bug, medium (code path high). Carried over from the first review (old 5.3) and now sharper. The response estimator (`map_joint_phi_noise.py:115-121`, used by the sampler merge) is not bounded by r² ≤ 1, so a well-measured mode can be stored with `nphi < 0`.
- `fisher_forecast.py:180, 211-213` keeps it whenever `C_phi + N_phi > 0`, so the mode carries more than cosmic-variance-limited information.
- `delensed_covariance.py:230` maps it to infinity, so W = 0 and the best-measured mode gets no delensing.
- `phi_noise_analysis.py:62, 137` drops it from the band median, biasing the median high.

With 100 realizations this is plausible on the lowest-L modes. It needs one convention (clip to 0, or NaN) applied once, in the merge. Quick once the convention is chosen.

**3.2 The `IP` forecast's default N_phi is the EB estimator's** — design, high. `camb_spectra.qe_noise_matrix` uses `polar_quadratic_estimate` for both "P" and "IP", as `load_sim` does. At nside 64 / 2.5' / 5 uK the EB N0 measured about 80× the TT N0 (unlensed filter), so the `IP` phi block carries far less lensing information than the `I` one, and the default `IP` forecast came out with a slightly wider omch2 than temperature alone. A combined TT + EB N0 does not exist in the code. Also unexamined: the unlensed-filter EB N0 puts a zero BB in its filter, which is very optimistic. Until resolved, use a measured `--phi_noise` for `IP`.

**3.4 Non-trivial beams and masks: the forecast's beam is counted twice, and nothing has been tested with either** — latent bug, medium-high. To be revisited: **supporting non-trivial masks and beams is a goal**, to be taken up once the other issues in this document are worked out, so the fix here is not to drop the `beam_fwhm` argument but to make beams and masks work end to end.
- **What is wrong today.** There are two equivalent ways to write the power of a beamed, noisy map: beam on the signal, `B² C + N`, or beam divided out, `C + N / B²`. The code mixes them: `noise_cls` (`simulate.py:65-76`) returns the noise already divided by the beam, and `scalar_quadratic_estimate` (`simulate.py:397-399`) forms `(mask · beam)² C + noise`, so with a beam the estimator's filter sees `B² C + N / B²` and the signal-to-noise falls as B⁴ instead of B². `camb_spectra.instrument_matrices` (`camb_spectra.py:91-93`) passes one `beam_fwhm` to both, so `forecast(..., beam_fwhm)` / `--beam_fwhm` inherits it. The same pairing exists inside `load_sim` (a beam-deconvolved `noise_cls` and a separate beam operator on the signal), where it is never exercised because the beam FWHM is always 0.
- **Why it has not bitten.** Every run so far has the beam equal to the identity: `load_sim` calls `noise_cls` and `get_beam` without a beam, `delensed_covariance.py` hard-codes a beam of 0 for its Wiener filter, `sampler_phi_noise.run_job` stores `beam_fwhm = 0.0`. With B = 1 the two conventions agree. A forecast run with a beam today would also be inconsistent with its inputs, which were all measured on beam-free maps.
- **What supporting them needs.** One convention for the beam chosen in `simulate.py` and used by the noise, the data model, the Wiener filter, both QE norms and the forecast; a `beam_fwhm` (and a mask choice) threaded through `load_sim`'s callers, the three measurement pipelines and their drivers; 3.5 fixed for masked modes; and the Julia comparison of 12.11 run first, since that is where deeper problems would show.
- Related: `instrument_matrices`' `lmax_prime` parameter (`camb_spectra.py:83-90`) has no caller.

**3.9 The polarization path is verified by reading and one small local run only** — design. The 2×2 TT/TE/EE block, the EB N0, the polarized delensed covariance and the `pol` option of the two phi-noise pipelines ran once at nside 64 with 4 realizations. No P or IP chain exists to compare a forecast with, nothing ran on the cluster, and there is no test. The block contraction itself was checked against a dense per-mode loop.

**3.10 A single map or a stuck chain crashes `chain_analysis` in the formatting code** — fragile, high. With one map `widths_std` and the jackknife errors are NaN and `standard_notation` reaches `int(np.floor(np.log10(nan)))` (`chain_analysis.py:411-412, 430`), after all per-parameter plots have been written; a sub-chain that never moves makes `gaussian_kde` raise a singular-matrix error. The guards for this were removed on purpose (no run has one map). Listed so the behaviour is known: "needs at least two maps" surfaces as a formatting crash. Involved only in the sense that it is a decision, not a bug.

---

## 4. Sampler

Files: `cmb_lensing/sample_lcdm.py` (1129 lines now), `get_lcdm_samples/run_single_lcdm_chain.py`.

### Quick fixes

**4.1 Out-of-bounds rejections are not counted in the acceptance rate** — bug in the diagnostic, high ✔. `sample_lcdm.py:263-264`: `if not (lo <= theta_prop <= hi): continue` skips the `lcdm_acceptance` bookkeeping, so the printed rate is biased high near the box edges.

**4.2 `to_camb_naming_conv` mutates the caller's `param_init`** — fragile, high. `sample_lcdm.py:712` (function at 678-683, with `del param_init["theta_MC_100"]`). A second `sample_joint` call with the same dict fails at line 892.

**4.3 History files are appended (stop-gap)** — likely bug, high ✔. `sample_lcdm.py:1057-1058` opens `{hpc_path}{theta}_map_..._history.txt` with `"a"`, and `sample_lcdm.sh:79` names the folder only by date, so a same-day resubmission or rerun silently appends a second chain to the first; `submission_settings.txt` is rewritten with `cat >` (`sample_lcdm.sh:126`). Quick stop-gap: refuse or truncate an existing history file at chain start. The proper fix (a run id in the folder name) is involved.

**4.4 Resuming with the original `seed` replays the original key stream** — fragile, high ✔. `sample_lcdm.py:921-924`: the saved `sub_key` is used only when `seed` is None.

**4.5 HPC chains cannot be reproduced** — design, high. `run_single_lcdm_chain.py:111` passes `seed = None`; `sample_lcdm.py:926` draws `np.random.randint(0, 2**31)` and never records it (`sampler_phi_noise` stores its `chain_seed`). The comment at `run_single_lcdm_chain.py:50` ("using the chain index as the seed") is stale.

**4.6 A divergent HMC trajectory is accepted during always-accept burn-in** — fragile, medium. `sample_lcdm.py:193`: `accept = jnp.logical_or(always_accept, log(u) < delta_h)`; phi then stays NaN for the rest of the chain. Dormant on the HPC path (`num_burn_in_always_accept = 0`).

**4.7 Off-by-one in both burn-in counters** — cleanup, high. `sample_lcdm.py:988` (`iter >= num_burn_in_fix_theta`) and `:155` (`iter < num_burn_in_always_accept`) with `iter` starting at 1 give N-1 sweeps each; the history file and `param_vals` then have different lengths.

**4.8 Smaller items**
- `sample_lcdm.py:989-990`: a parameter with `should_sample = True` but no `proposal_sigmas` entry is never moved, yet its history is written as a constant; `should_sample[theta]` is a hard index here but `.get(name, False)` at line 888.
- `sample_lcdm.py:1013`: `update_args_after_sample` runs every sweep after burn-in even when no move was accepted.
- `sample_lcdm.py:1020-1052`: both progress plots are redrawn from the full history every iteration (O(n²) over a chain); `plt.legend()` at 1031 has no labelled artist; the title "(mean - sample)/sigma" (1045) does not match what is plotted, (sample - truth)/sigma.
- `sample_lcdm.py:668-674, 706`: `args["lmax"]`, `["lmax_prime"]`, `["ells"]`, `["cphi_fid"]` are never read, so `sample_joint(lmax = ...)` has no effect although its docstring documents it; the comment at 704-705 ("needed for the G mixing matrix") is stale.
- Comments at `sample_lcdm.py:36` and `:289` say `eval_logpdf_batch` is jitted; it is not.

### Involved fixes

**4.9 Resuming a chain is not a continuation, and no script can resume** — likely bug, high ✔.
- `sample_lcdm.py:961, 988, 155`: a resumed chain restarts `iter` at 1, so theta is frozen again for `num_burn_in_fix_theta - 1` sweeps and the history file gains about 99 duplicate lines mid-chain.
- `sample_lcdm.py:882-885, 712-717`: the QE norm (hence G and the phi mass matrix) is rebuilt at the resumed parameters, so the kernel differs from the first segment.
- `run_single_lcdm_chain.py:116-121`: the resume npz is written once, after `sample_joint` returns; a job killed at the wall clock leaves none, and the script has no resume argument. Its key `"phi_map"` holds the Fourier matrix.

A sweep counter and the seed stored in the resume npz, written periodically, plus a resume flag in the driver, fix these together with 4.4.

**4.10 The throwaway `load_sim` at start-up** — design. `sample_lcdm.py:715-717`: a full simulation, seeded from the global `np.random`, runs only to obtain the lensed covariance for the QE norm.

**4.11 `gibbs_sample_f` may be only an approximate draw** — suspected, not verified. `sample_lcdm.py:141-143` calls `wiener_filter` with its default `tol = 1e-1`, an absolute threshold (5.8).

**4.12 Performance of the theta step**
- `sample_lcdm.py:266`: the current-point logpdf is recomputed for every parameter and every step although the fields have not changed. About half the theta-step cost. (high)
- `make_eval_logpdf_batch` (282-437) is rebuilt at line 450 on every call and is not jitted. (medium, not profiled)

**4.13 Structure**
- **Duplication.** The Cl-to-covariance blocks are written out about four times: `single_logpdf`'s three branches (355-431), the three `_recompute_cosmo_matrices*` kernels (471-588), `get_new_cf_matrix` (623-661); the teb / eb / scalar draw twice in `gibbs_sample_f` (99-130).
- **Dead configuration axes.** `bb_is_zero = False` (see 6.1), `refresh_qe` (never set by `sample_joint`) and `fixed_fields` are threaded through every signature.
- **Import weight.** `fixed_theta_chain.py:79` imports `sample_lcdm`, which sets `matplotlib.use("Agg")` and imports `huggingface_hub`.
- **Editable-install only.** `constants.py:81` puts `camb_splines/` at the repository root; `sample_lcdm.py:1032, 1049` and `fisher_forecast.py:91` write inside the package directory.

---

## 5. Core numerics

Files not touched by the refactors; line numbers are the first review's.

### Quick fixes

**5.1 `primal_log_det` is not a log-determinant for a negative entry** — latent bug, high on the maths. `util.py:39-42` multiplies the whole sum by `prod(sign(...))`, so one negative covariance or Schur-complement entry gives a finite, badly wrong logpdf instead of a rejection. Return NaN when any entry is negative.

**5.2 Q and U are swapped in `output_like_input`** — latent bug, high. `lense_flow.py:140-152`: `input_field.replace(polar_matrix_1 = u_matrix, polar_matrix_2 = q_matrix)` while `spin_vector_matrix_data` reads Q from `polar_matrix_1`. Harmless today (only phi is differentiated); any future gradient with respect to a polarized field through `lense_flow_wrapper` would be silently wrong.

**5.3 Circular import between `statistics.py` and `gradients.py`** — fragile, high. `statistics.py:11` star-imports `gradients`, which imports `logpdf` from `statistics`; it works only because of the order in `__init__.py`. `statistics.py` uses nothing from `gradients` or `dataset`, and its unused `import matplotlib.pyplot` (line 5) loads pyplot before `sample_lcdm`'s "must be first" `matplotlib.use("Agg")`.

**5.4 Opposite positional order of G and D** — fragile, high. `mixed_logpdf(..., mixing_g, mixing_d)` (`statistics.py:173-175`) vs `mixed_grad_phi_logpdf(..., mixing_d, mixing_g)` (`gradients.py:41-43`). Silent for T-only. All current call sites are correct. Make them keyword-only.

**5.5 `get_identity_like` is not an identity for `BlockTEB`** — bug in dead code, high. `matrix_operators.py:191-195` sets the TE / ET blocks to ones. Unused; delete it.

**5.6 `_matrix_names()` returns a set on operators** — fragile, high. `matrix_operators.py:63, 99, 147` (a list in `fields.py:101`); set order varies with `PYTHONHASHSEED`. No bug today.

**5.7 JAX arrays created inside a `pure_callback` host function** — fragile, medium. `simulate.py:506, 521-529`; can deadlock on GPU. Return numpy.

**5.8 Wiener filter housekeeping** — cleanup, high. `wiener_filter.py:46-49, 71-99`: `A_matrix_operator`'s `data` parameter is overwritten immediately; the CG state carries two always-equal residuals; `tol = 1e-1` is an absolute threshold on `r·M⁻¹r` (it scales with nside and units), unnamed, and non-convergence is silent.

**Smaller**
- `dataset.py:70, 100, 115`: wrong type annotations and defaults for `mixing_g` and `DataSetTEB.phi`.
- Names arriving only through chained star imports: `np` in `statistics.py`; `jfft`, `Basis`, `Parametrization` in `fields.py`; several constants in `simulate.py`; `plt`, `np`, `jnp`, `jfft`, `partial` in `sample_lcdm.py`. An `__all__` in `util.py` would break three modules.
- `simulate.py:1005`: `get_g_matrix(cphi, qe, a_phi, a_phi = a_phi)` is identically 1, so every `load_sim` data set carries G = identity.

### Involved fixes

**5.9 `load_sim`'s `H0` argument cannot work, and errors become NaN** — bug, high. `simulate.py:799-805, 511-530`: with `H0` set CAMB raises "Set H0=None when setting theta.", swallowed by a bare `except Exception` into NaN spectra; so is any programming error. Catching only CAMB's own exceptions keeps the sampler's reject-on-NaN behaviour. (The catch itself is quick; deciding what `H0` should do is not.)

**5.10 Field metadata are differentiable pytree leaves** — design, high. `fields.py:22, 98`, `gradients.py:34-37` (the TODO there): `basis`, `parametrization` and `fourier_weights` are leaves, not `pytree_node = False`. This is why `allow_int = True` is needed, why `grad_phi_logpdf` patches three fields back, and why `field.basis` is a tracer under jit. Making them static removes the workarounds.

**5.11 `n`, `direction`, `adjoint` are differentiable primals of the custom VJP** — design / performance. `lense_flow.py:10-13, 37-46`: they are tracers, so the RK4 loop lowers to a `while_loop`, the backward pass re-traces a forward lensing to produce zero cotangents, and the backward rule ignores `adjoint`. `dt0` is computed and never used (57-74, 305-322).

**5.12 The `map_joint` line search** — fragile, medium. `map_joint.py:81-87`: unbounded BFGS with finite-difference gradients on a 1D problem, `optimizer.success` never checked; a NaN alpha poisons every remaining iteration. Julia uses a bounded Brent search (`minimize_scalar` is imported, unused).

**5.13 Tensor spectra above ℓ = 600 at default settings** — likely bug for r > 0, medium, not verified. `simulate.py:916-917`, `constants.py:31`: with `max_l_tensor = 600` the zeroed tensor spectra go through the log interpolation and `cf_tensor` would be NaN above ℓ ≈ 600. The sampler runs at r = 0 and is safe; `batch_simulated_trials`, `get_avg_cls` and `simulate.__main__` default to r = 0.2.

**5.14 Two Nyquist conventions** — fragile, medium. `simulate.py:308-354, 380-386` rebuild the k-grid three times with numpy's positive Nyquist while `util.get_k_meshgrid` flips it negative to match Julia; axis naming is opposite between helpers.

**5.15 Operator arithmetic gaps** — design, high. `fields.py:65-73`, `matrix_operators.py:74, 174`: `field * operator` silently returns a field whose matrices are operator objects; `__neg__` and `__radd__` are missing.

**5.16 Performance**
- T-only gradients integrate zero Q/U (`lense_flow.py:95-100, 175-187`): about three times the FFTs needed in every RK4 stage of every leapfrog step. (high)
- About two of six lensing solves in `mixed_grad_phi_logpdf` are repeats (`gradients.py:41-62`). (medium)
- Eager callers recompile the RK4 loop on every call (`lense_flow.py:289-333`, `util.py:184-199`); so do the 14 eager `lax.cond` calls in `dl2cl`. (medium)
- The polar QE computes 48 zero terms of 64 through traced indices (`simulate.py:410-442`). (medium-high)

**5.17 Low-confidence suspicions**
- `simulate.py:373-387`: `(1j * KX)**num_x` with a traced exponent relies on complex `0**0 == 1`; a NaN would be silently zeroed.
- `statistics.py:73`: `primal_power_spectra` weights modes by `(2 ell + 1) / 2` within a bin. Test plots only.
- `statistics.py:200-213`: `dot_backwards` returns the cotangent of `dot(phi, phi_covar)`; nothing in the code marks it.
- Geometry is carried redundantly on every field, operator and data set, with 256 / 2.5' defaults that make `FlatS0(scalar_matrix = ...)` silently wrong at other sizes.
- Importing any `cmb_lensing.*` module runs `__init__`, which pulls in JAX, flax, camb, scipy and pyplot.

---

## 6. CAMB grid

### Quick fixes

**6.1 `bb_is_zero` is hardcoded True** — design, high ✔. `merge_camb_grid.py:114`; no `coeff_bb` is ever written. The `if not bb_is_zero` branches in `sample_lcdm.py` (350-351, 375-378, 410-413, 524-527, 566-569) are unreachable. Quick to delete the branches; involved if unlensed BB is ever to be supported.

**6.2 The theta → H0 inversion is repeated per spectrum** — performance, high. `camb_grid_interp.py:173`: `grid_points` (a 2000-point spline line per row) runs once per spectrum per call, 2 to 4 identical inversions per logpdf batch. Cache the last result.

### Involved fixes

**6.3 The grid merge trusts every job** — fragile, high. `run_single_camb_grid.py:59, 104-105, 139` promise checks ("the merge step can verify and flag that") and save `ok`; `merge_camb_grid.py` reads neither. A failed H0 node NaNs that spectrum's whole table; a missing slab stays at lnCl = 0 (`np.zeros`, line 92); `cl_te_lensed` is computed, stored and never merged. `tests/test_camb_grid.py` is the only guard.

**6.4 theta(H0) is assumed monotone** — fragile, medium, suspected only. `camb_grid_interp.py:138-145`: `np.interp(theta, curve, h0_dense)` returns a wrong H0 silently if the spline wiggles near an edge.

---

## 7. Forecast and measurement modules

See also section 3: most open items here are new.

### Quick fixes

**7.1 Two estimators under one file name** — likely bug or stale doc, high. `convert_old_sampler_phi_noise.py:40-41` uses the default `estimator = "correlation"`; its docstring says "exactly as sampler_phi_noise.merge_phi_noise does", which uses `"response"` (line 3 still cites the removed `--nphi_source sample_joint`). The merged dict (`sampler_phi_noise.py:291-299`) has no `estimator` key. `phi_noise_analysis.py:193` plots the converted (`_old`) files as "Gibbs Sampler", and its docstring at line 91 hard-codes the correlation formula. Add an `estimator` key.

**7.2 One straggling chain truncates a whole map** — fragile, high. `sampler_phi_noise.py:255`: `n_post_burn_in = min(n_done) - burn_in` over a map's chains. A chain just past the burn-in cuts all of them to a handful of sweeps instead of being left out. Matters because the merge is advertised to run on live jobs.

**7.3 Smaller items**
- `sampler_phi_noise.py:236, 242`: every sidecar `NpzFile` stays open for the whole merge; `:150` zeroes the `.npy` slightly before `:169` resets `n_done`, so a concurrent merge can read zeros.
- `sampler_phi_noise.py:263-264`: R-hat is computed on unprojected chain moments, so the kx = 0 / Nyquist columns include the unphysical degrees of freedom.
- `plot_sampler_convergence.py:44-45, 62, 79`: `np.errstate` does not silence `nanmean`'s warning; the pre-2026-11-09 merged files lack `rhat_*` / `n_chains` and raise `KeyError`.
- `map_joint_phi_noise.py:72-73`: `cross_power_grid` round-trips numpy arrays through `jnp`.
- `phi_noise_analysis.py:8` imports the whole forecast (CAMB, matplotlib Agg) for `fourier_weights`, a three-line wrapper of `util.get_fourier_weights`.

### Involved fixes

**7.4 `map_joint_phi_noise` may keep anti-Hermitian content** — suspected, low-medium, not verified. `map_joint_phi_noise.py:103-104` uses `map_joint`'s phi as-is, without the Hermitian projection `sampler_phi_noise.hermitian_part` applies. The fix is two lines; knowing whether it matters needs a run.

(3.1, 3.2 and 3.9 are the other involved items in this area.)

---

## 8. Chain analysis

`cmb_lensing/chain_analysis.py`. See also 3.8 and 3.10.

### Quick fixes

**8.1 The trace plot is of thinned, spliced chains** — cleanup. `chain_analysis.py:387`: `plot_traces` receives one thinned array per map (its sub-chains concatenated), so each trace is five sub-chains spliced together.

**8.2 The acceptance rate is only the Metropolis rate for one move per sweep** — design, medium. `chain_analysis.py:179-186` counts sweeps in which the value changed, which equals the acceptance rate only for `metropolis_num_steps = 1`. Stated in the docstring. Exact if the sampler logged its accepts.

### Involved fixes

**8.3 The dashed "± 1 jackknife" ellipses are not a bound** — design, medium. `chain_analysis.py:643-646`: shifting every covariance element by its own error is not a contour in any defined sense, and the shifted matrix need not be positive semi-definite (`:691, :708` then give NaN axes and the ellipse silently disappears).

**8.4 The product of KDEs is biased wide** — design, medium-high. `chain_analysis.py:339`: each per-map `gaussian_kde` has variance s²(1 + h²) with Scott's h = n^(-1/5), 3-4 % extra width for 500-1000 thinned samples, and the product inherits it. "Product Std" is biased high and the z-score low by about that much, which is the size of the quoted errors; the bootstrap re-applies the same smoothing and does not cover it. Options: deconvolve the bandwidth, or multiply Gaussians fitted to each map's mean and std.

**8.5 Switch the triangle plot to GetDist** — future work, not a bug (noted 2026-11-12). `chain_analysis.triangle_plot` draws its panels and ellipses by hand. GetDist is the standard cosmology library for this and takes a covariance directly: `getdist.gaussian_mixtures.GaussianND(mean, covariance, names = ..., labels = ...)` for the forecast, `getdist.MCSamples(samples = ..., names = ..., labels = ...)` for the chains, and `getdist.plots.get_subplot_plotter().triangle_plot([chains, forecast], filled = True)` overlays the two. It adds one dependency and would remove the hand-rolled ellipse code, the unnamed layout literals (section 15), the unused `sigmas` argument and entry 8.3's dashed ellipses. To settle first: the current chain contours are the ellipse of the per-map covariance averaged over maps, whereas `MCSamples` on pooled samples would show the spread of all maps together (wider, since each map's posterior is centred differently). Either pass the averaged covariance as a second `GaussianND`, or recentre each map's samples on its own mean before pooling.

---

## 9. MCMC diagnostics

`cmb_lensing/mcmc_diagnostics.py`, unchanged. All quick.

- `:38`: the docstring says "truncate at lag M where M < c * tau(M)"; the code (correctly) uses `m >= c * tau`.
- `:52-54`: a constant chain gives variance 0, a NaN autocorrelation and a NaN tau.
- `:39-44`: a noisy autocorrelation can drive tau to 0 or below and return at once; if the window never closes the full noisy sum is returned with no signal.
- `:53`: `np.correlate(..., "full")` is O(n²); an FFT autocorrelation is O(n log n).
- `:20`: `gelman_rubin_from_moments` raises `ZeroDivisionError` for one chain; `gelman_rubin` is unguarded (both current call sites guard themselves).

---

## 10. HPC drivers and job scripts

The positional plumbing of all six pipelines lines up (section 16), including the three whose arguments changed.

### Quick fixes

**10.1 The example local file overrides the documented defaults** — fragile, high. `hpc_local.example.sh:19-20` exports `MAX_QUEUED_JOBS=89` and `SUBMIT_POLL_SECONDS=600`, labelled "optional" but live, replacing the documented 85 / 60 s; 89 is one below the ~90 cap.

**10.2 Memory requests** — design, high. Every JAX job script asks `--mem-per-cpu=16G` with 4 CPUs: 64 GB against 1.3 to 4.5 GB measured. `get_sampler_phi_noise.sh:66` calls `chain_mem` "memory per job" but passes it as `--mem-per-cpu`.

**10.3 `true` is case-sensitive in the driver but not in Python** — likely bug (minor), high. `sample_lcdm.sh:183` (`[ "$sampled" = true ]`) vs `run_single_lcdm_chain.py:47` (`value.lower() == "true"`): `--sample_ns=True` makes Python sample ns while the driver starts it at the truth and records that.

**10.4 A failed `conda activate` is not caught** — fragile, high. Job scripts do `source ".../hpc_local.sh" || exit 1`, which only catches a missing file; the job then runs under whatever `python3` is on PATH.

**10.5 Smaller items**
- No SBATCH `--output` or `--job-name`: `slurm-<jobid>.out` files land in the driver's tracked folder and do not identify map or chain; `.gitignore` covers neither them nor `submit.log`.
- Default output folders not gitignored: `get_delensed_field_covariance/delensed_covariance_output/`, `measure_phi_noise/get_sampler_phi_noise/sampler_phi_noise_output/`, `camb_grid_generation/multi_param_CAMB_grid/`, `runtime_comparison/performance_results/`.
- Seed conventions differ: `sample_lcdm.sh:187` uses `map_prefix * map` from 1; the measurement drivers use `seed_prefix + m` from 0, so only the first chain map coincides with a measured realization despite the "MATCH sample_lcdm.sh" comments.
- `run_single_lcdm_chain.sh:14`: `export JULIA_PKG_PRECOMPILE_AUTO=0` and the comments on lines 8, 10-11 are leftovers from the Julia scripts.
- `get_sampler_phi_noise.sh:38-39` says "set burn_in in it first"; the merge takes `--burn_in=`.
- `get_sampler_phi_noise_1_chain.py:42-43` says "the analysis rejects duplicate seeds"; the merge groups by `map_seed` and would pool such files as chains.

### Involved fixes

**10.6 `--name=value` parsing accepts anything** — fragile / design, high. All drivers: `for arg in "$@"; do declare "${arg#--}"; done`. A misspelled name is silently accepted (`--noise=10` runs the default noise); `--nside 64` prints errors and carries on; any variable can be clobbered (`--PATH=`); `--out_dir=` is overwritten later; `declare` evaluates array subscripts; `--MAX_QUEUED_JOBS=` is overridden by `hpc_local.sh`. Needs an allow-list of settings per driver.

**10.7 No `set -euo pipefail` in any driver or job script** — fragile, high. A failed `mkdir`, `bc` or audit write does not stop submission. Adding it is one line per script, but each script needs checking for commands that legitimately return non-zero.

**10.8 The throttled submit loop** — fragile, high on mechanics. `hpc_submit.sh:22-29`: a failing `squeue` yields 0 and opens the throttle; the `sbatch` retry is unbounded, so a permanent error loops forever under `nohup`; check-then-submit is not atomic across two drivers. Untested on the cluster.

**10.9 Structure**
- Ground truth and box bounds are retyped as shell literals in `sample_lcdm.sh:28-52` and `camb_grid.sh:41-48`, duplicating `constants.py`.
- 9 to 42 positional arguments are passed only to be turned back into `--flags`; passing `--name value` pairs straight through with `"$@"` would remove the index bookkeeping.

---

## 11. Runtime comparison

### Quick fixes

**11.1 Every pol "P" Python timing job crashes** — bug, high. `runtime_comparison/python_performance_test.py:181-189`: `noise_covariance.replace(polar_matrix_1 = ..., polar_matrix_2 = ...)` on a `DiagonalEB`, whose fields are `matrix_EE` / `matrix_BB`; `dataclasses.replace` raises `TypeError`. Same for `field_covariance`, `mask`, `beam` in that branch.

**11.2 Smaller items**
- `python_performance_test.py:10-12` uses a private juliapkg internal and `shutil.which("julia")`, which is None when julia is not on PATH.
- `performance_analysis.py:15-16`: `BASE_DIR = "/media/zane-blood/KINGSTON/"`; seeds, trials and map sizes are hardcoded; four of five `__main__` calls are commented out.
- `julia_performance_test.jl:89-100` writes .npy format under an `.npz` name.
- `run_performance_test.sh:75` and the job scripts leave `$1`..`$6` unquoted; the defaults submit 2400 jobs.

### Involved fixes

**11.3 Julia may be running single-threaded** — suspected, medium. `julia_performance_test.sh:19` starts `julia` with no `-t` on a 4-CPU allocation where JAX uses all four cores. If CMBLensing.jl's FFTW threading follows `Threads.nthreads()`, the timings behind `paper/run_time_comparison.png` are biased. Checking and re-running the timings is the cost.

---

## 12. Tests

Unchanged by the refactors. 49 tests collected; only `test_logpdf.py` (3) and `test_camb_grid.py` (1, skipped without the 8 GB grid) have regression-grade assertions.

| File | Tests | Asserts that can catch a real numeric bug | Tolerance |
|---|---|---|---|
| `test_lensing.py` | 12 | 0 | relative norm ≤ 1; mean correlation > 0 |
| `test_gradients.py` | 9 | 0 | same |
| `test_wiener_filter.py` | 3 (12 comparisons) | 0 | relative norm ≤ 1 |
| `test_map_joint.py` | 3 | 0 | relative norm ≤ 1, field only |
| `test_logpdf.py` | 3 | 3 | abs diff < 1 on ~1e6 |
| `test_simulated_cls.py` | 3 | 0 | `assert 1 == 1` |
| `test_covariance_matrices.py` | 15 | 0 (weak) | relative norm ≤ 1; abs ≤ 1 on zero blocks |
| `test_camb_grid.py` | 1 | 1 | per-spectrum 5e-4 to 1e-2 |

### Quick fixes

**12.1 Tolerance is 100 % relative error** — design, high ✔. `MAX_NORM_DIFF = 1`, `MIN_AVG_CORRELATION = 0` in five test modules; an all-zero prediction passes. The 2 % two-column gradient bug described in `CLAUDE.md` passes by a factor of about 50. Replace with per-quantity tolerances at 3-10× the currently measured error; add a per-column metric for the gradients (columns 0 and -1) and a per-mode log-ratio for the covariances.

**12.2 `test_map_joint_*` never asserts on phi** — test gap, high. `test_map_joint.py:170, 383, 667`.

**12.3 Missing or weak asserts**
- `test_lensing.py:292`: `test_inverse_adjoint_lensing_intensity_only` computes a correlation and never asserts on it.
- `test_covariance_matrices.py`: Cphi, D and G are dumped by Julia but never compared; the all-zero blocks (`:112, 155, 198`) are compared by absolute norm ≤ 1 where `jnp.all(x == 0)` is the real check.

**12.4 Portability**
- `GROUND_TRUTH = os.getcwd() + "/tests/ground_truth_data/"` read at import in seven modules: `cd tests && pytest` fails at collection. `conftest.py:6` already computes `TESTS_DIR`.
- `GROUND_TRUTH` and `CAMB_LMAX` shadow the package constants after the wildcard imports.
- `matplotlib.use("Agg")` is set in two modules and not the other four; one call in `conftest.py` covers all.
- `_preamble.py`: bare `KeyError` if `CMBLENSING_JL_PATH` is unset; unused `using PythonPlot`.
- `generate_simulated_cls.py:21-28` and `test_simulated_cls.py` are unseeded, so every run rewrites files and figures.

### Involved fixes

**12.5 `test_simulated_cls.py` cannot fail** — design, high ✔. `assert 1 == 1` (lines 81, 160, 266) after 100 `load_sim` calls at nside 256. A real assertion: the binned ratio to Julia within a few `sqrt(log_c_ell_variance / num_trials)`.

**12.6 The mixed parametrization is only ever tested with G = identity** — high ✔. `test_gradients.py:676, 763, 882` all load `g_I.npz`, which is identically 1; no `g_P` / `g_IP` exists. Needs new Julia dumps with a non-trivial G, or a Julia-free round-trip test.

**12.7 The Wiener-filter ground truth is loosely converged** — design, medium. `generate_wiener_filter.py:12` uses `tol=1e-1`, like Python; a tight tolerance needs regeneration at about 1e-6.

**12.8 `--generate` is fragile** — high. `conftest.py:29-47` runs generation inside `pytest_collection_modifyitems` (after the test modules are imported, also under `--collect-only`); per-module generation regenerates only that module's outputs, so a partial run mixes two sims; 17 generated files are never loaded; Julia globals leak between `seval` blocks.

**12.9 Untested code that is cheap to test without Julia** — design, high. No test imports the sampler, the forecast, the measurement modules, the diagnostics or the RNG helpers. Candidates, each milliseconds to seconds:
- `gelman_rubin` vs `gelman_rubin_from_moments`; `integrated_autocorrelation_time` on an AR(1) chain.
- The phi-noise estimators and their jackknife on synthetic `phi_hat = W phi + noise`.
- `spline_coefficients` / `evaluate_spline` on a tiny grid against `scipy.interpolate.CubicSpline`, including the out-of-box NaN contract the sampler relies on.
- `fisher_from_blocks` on analytic blocks (the reductions used to check it on 2026-11-11: TE = 0, and a dense per-mode loop); `load_merged_covariance` on both file formats.
- `chain_analysis.load_chains` on synthetic history files; `standard_notation`.
- `metropolis_sample_theta` with a stub Gaussian logpdf.
- The common-random-number property of `field_from_covar`; `chain_key` distinct from `load_sim`'s keys.
- `mix` / `unmix` round trip with a non-trivial G and D (covers 12.6); `pinv` / `log_det` of `BlockTEB` against a dense inverse.
- Lensing self-consistency (`<L f, g> == <f, L† g>`), and a finite-difference check of the phi gradient including the imaginary DC / Nyquist directions, which is the only way to pin the Fourier-gradient fix besides 12.1.
- A sampler smoke test with stub `GridPredictors`.

**12.11 The Python and Julia code bases have never been compared with a non-trivial mask or beam** — test gap, high that the gap exists; what it hides is unknown. Every Julia comparison so far (the `tests/ground_truth_data` dumps and the tests built on them) was made with the beam set to the identity and the mask set to a low-pass in ℓ, and every sampler, forecast and measurement run uses an all-ones mask and no beam. So the port has only ever been checked in the regime where the beam drops out and the mask is a smooth isotropic cut. Before masks and beams are supported (3.4), the two code bases should be diffed with both made non-trivial: a finite beam FWHM, and a mask that is not a function of |ℓ| alone (CMBLensing.jl's pixel-space and Fourier-space masks). Places where a difference could be hiding:
- the beam convention itself (3.4): which of the noise, the data model and the estimators carry B, and whether `noise_cls`' deconvolved noise matches what `load_sim` in CMBLensing.jl builds;
- every place the mask and beam enter as one "transfer function" product: the data model `mask * beam * lensed + noise`, the Wiener filter's operator and preconditioner, `logpdf` and both gradients, the D mixing matrix, and the scalar and polar QE norms (`tf = m * b`, `(mask * beam)²`);
- `BlockTEB` masks and beams, whose TE blocks are assumed zero, and the `op_shared_matrix` shortcut in `sample_lcdm.py` that passes a single mask / beam block to the jitted kernels for every component;
- the QE norm's "no response" modes stored as 0 (3.5), which only appear once a mask removes modes;
- anything that assumes the mask commutes with lensing or is diagonal in Fourier space, which a pixel-space mask is not.

This needs new Julia dumps (`tests/generate_julia_data/`), new comparison tests with real tolerances (12.1), and probably fixes on the Python side; it is the prerequisite for 3.4, and it is possible that it turns up porting errors deeper than the ones listed in this document.

**12.10 Duplication** — cleanup, high. About 4300 lines in four test files are hand-copied variants of one pattern; a data-set loader, one plot helper and `pytest.mark.parametrize` over pol would shrink them to a few hundred. The duplication already produced drift (12.3, 12.6).

---

## 13. Packaging, CI and repository hygiene

### Quick fixes

**13.1 `CITATION.cff` is invalid** — bug, high. It lacks the required `title` and `authors`, so GitHub's "Cite this repository" will not render.

**13.2 `setuptools-scm` pulls every tracked file into an sdist** — likely bug, high. `pyproject.toml:2` requires it while the version is static; its only effect is the file finder (about 117 MB in an sdist). Remove it.

**13.3 Other `pyproject.toml` items** — `license = "MIT"` (PEP 639) needs setuptools ≥ 77 while the floor is 68; no `authors` / `urls`; `jaxlib` listed separately; `camb>=1.5.0` while the grid was built with CAMB 2.0.0; `packages.find` discovers data directories as namespace packages; `juliacall` is in no extra.

**13.4 Git state after the refactor** (also at the top of section 1)
- Untracked and needed: `cmb_lensing/camb_spectra.py`, `cmb_lensing/matrix_plots.py`, `docs/api/chain-analysis.md`, `docs/chain_analysis_tutorial.ipynb`.
- `cmb_lensing/sample_lcdm_output/{lcdm_progress,logpdf_progress}.png` are tracked and rewritten by every `python -m cmb_lensing.sample_lcdm` run.
- `.gitignore` ignores a source file by name, `convert_old_sampler_phi_noise.py`, which `CLAUDE.md` documents as part of the layout; it has no trailing newline.
- `get_lcdm_samples/chain_analysis_output/` and `lcdm_chain_plots/` are leftovers of the old script.
- `mkdocs.yml:3` `site_url` points at `zaneblood1.github.io` while README and CONTRIBUTING link to `zaneblooddev.com`; no `docs/CNAME`.

**13.5 The docs CI deploys without `--strict`** — `.github/workflows/docs.yml` runs `mkdocs gh-deploy --force` directly; its dependency list duplicates the `docs` extra with different bounds.

### Involved fixes

**13.6 CI covers only the docs** — design, high. No pytest, import smoke test, shellcheck or lint; this is how 11.1 went unnoticed for a month.

**13.7 The package only works as an editable install** — design, high. See the last bullet of 4.13.

**13.8 The repository doubles as a data store** — `.git` is 907 MiB for about 117 MB of content. 1050 tracked chain-history files sit under the gitignored `get_lcdm_samples/lcdm_sampler_data/` and are byte-identical duplicates of files under `experimental_results/`; 48 more sit under the ignored, obsolete `lcdm_chain_plots/`; 363 test figures (50 MB) are tracked and rewritten by every test run. Untracking needs the duplicates confirmed first; shrinking history needs a rewrite.

**13.9 `tests/index.html`** depends on server directory listings and shows no numbers.

No secrets, tokens or e-mail addresses were found in tracked files.

---

## 14. Documentation drift

All quick. `AGENTS.md` mirrors `CLAUDE.md`, so each `CLAUDE.md` row applies to both.

| Document says | Code says |
|---|---|
| `CLAUDE.md:180, 204`: `get_g_matrix_lcdm(cphi_fid, cphi_curr, nphi, cn_tt)`, normalised against `cphi_fid` | `get_g_matrix_lcdm(cphi_curr, nphi)` (`simulate.py:284`) ✔ |
| `CLAUDE.md:380`: `lense_flow` default `n = 12` | `n = 10` (`lense_flow.py:12, 289`) ✔ |
| `CLAUDE.md:175`: the mask is forced to all ones, including in the Julia tests | `load_sim` has `use_mask` (`simulate.py:962`); the tests use a real mask ✔ |
| `CLAUDE.md:392`: every merge refuses mixed configurations and duplicate seeds, and sets unfinished files aside | Only the delensed-covariance merge checks anything |
| `CLAUDE.md:215`: `sample_lcdm_output/` is untracked | It is tracked |
| `CLAUDE.md:161`: the `get_*` folders are no longer gitignored | Their data subfolders are |
| `CLAUDE.md:46`, `docs/getting-started/hpc-setup.md:17`: throttle defaults 85 / 60 s | `hpc_local.example.sh` exports 89 / 600 s |
| **NEW** `CLAUDE.md:151`: `get_map_joint_phi_noise/map_joint_phi_noise/` | On disk: `map_joint_phi_noise_new/` and `_old/` |
| **NEW** `CLAUDE.md:279`: `nphi_leave_one_out` is read by `chain_analysis.py` | It is read by `phi_noise_analysis.py:114, 145` |
| **NEW** `CLAUDE.md:388`: the forecast calls `fisher_forecast.camb_results_at_params` | Removed; it is `camb_spectra.camb_cls_at_params` |
| **NEW** `CLAUDE.md:190`, `constants.py:59-60`: `PARAM_SIGMA` sets the forecast's steps | The forecast takes its steps from the merged file |
| **NEW** `CLAUDE.md:132`: `--pol` only labels the triangle plot | It also selects the forecast's pol (`chain_analysis.py:868`) |
| **NEW** `CLAUDE.md:412`: "(default Cl source)"; the graph omits `sample_lcdm → camb_splines` | The grid is the only source |
| **NEW** `CLAUDE.md:442`: "Known issues (as of 2026-11-04)" | Stale date |
| **NEW** `docs/index.md:34`: "the empirical delensed T covariance stencil" | Every pol now |
| **NEW** `docs/getting-started/installation.md:11`: JAX pinned to `== 0.9.2` | `pyproject.toml` has `>= 0.9.2` |
| **NEW** `docs/pipelines/lcdm-chains.md:33`: history files hold "every parameter" | Only sampled parameters get a file |
| `docs/getting-started/installation.md:44`: "The example script is gitignored" | `hpc_local.sh` is the ignored one |
| **NEW** `simulate.py:479-481, 849`: docstrings cite `fisher_forecast` keeping results for `get_partially_lensed_cls`, and `fisher_forecast.camb_cls_at_params` | Removed / moved to `camb_spectra` |
| **NEW** `forecast`'s docstring (`fisher_forecast.py:236-253`) | Does not document `l_cutoff` |
| `map_joint_phi_noise.py:7-13`, `convert_old_sampler_phi_noise.py:7`, `CLAUDE.md:279`, `docs/pipelines/map-joint-phi-noise.md:6, 12` | Use A / B / D letter shorthand for the moments instead of spelling them out |
| Section 16 of the first review: map_joint 8 args, sampler 12 | Now 9 and 13 |

---

## 15. Dead code, stale comments, magic numbers

All quick.

### Dead code

- `util.py`: `reshape_diagonal`, `loess` and its helpers (about 100 lines).
- `statistics.py`: `calculate_unbinned_cl`.
- `simulate.py`: `get_d_matrix`, `interpolate_cls`, `f_sky`, `log_c_ell_variance`, `import multiprocessing`, the `None` branches of `get_specific_derivative`. `load_sim` for pol "I" still builds the lensed EE / BB / TE covariances; for pol "P" it still draws and lenses T and builds the TEB D matrix.
- `constants.py:19, 93, 106-109`: `MAX_NUM_SAMPLED`, `MAP_MATRIX_DEFAULT`, `class Polarity`; `:12` `BEAM_TRANSFER_SCALAR = 8*jnp.log(2)` allocates a device array at import.
- `matrix_operators.py`: `get_identity_like` (5.5).
- `sample_lcdm.py:248, 186`: `symplectic_integrate` returns `p`, never used. `fixed_theta_chain.py:82-83`: redundant `data_set.replace(...)`.
- `map_joint.py:12`: `minimize_scalar` imported, unused.
- `performance_analysis.py:77`: unnecessary `allow_pickle`.
- **NEW** `camb_spectra.py:83-90`: `instrument_matrices`' `lmax_prime`; `chain_analysis.py:611`: `triangle_plot`'s `sigmas`; `fixed_theta_chain`'s `on_draw is None` branch (3.8).

### Stale or contradicting comments

- `lense_flow.py:250` ("laplacian sum minus the divergence term"; the code adds) and `:347` ("-1*divergence"; the code is +).
- `statistics.py:167-168` mentions subtracting logdet(G) inside `logpdf`; it happens in `mixed_logpdf`.
- `simulate.py:889-890` ("the emulator override"), `run_single_camb_grid.py:41` (`EMULATOR_MAX_ELL`), `sample_lcdm.py:677` ("between emulator and load_sim() naming convetion").
- `constants.py:86-87`: "2 arcminute" above `THETA_PIX_DEFAULT = 2.5`.
- `util.py:76` ("2N x N"); `statistics.py:24` (`primal_percent_diff_2d` returns a fraction); `simulate.py:142` (parameter `seed` is a PRNG key); `util.py:90` (`get_fourier_weights(field_shape)` uses only the row count).
- `sample_lcdm.py:961, 151`: the loop variable and a parameter named `iter` shadow the builtin.

### Magic numbers

- The lensing step count `10`: `lense_flow.py:12, 289`, `gradients.py:76`, `simulate.py:599-606, 953`, `sample_lcdm.py:134` (`delensed_covariance.py:56` names it `LENSE_STEPS`); `logpdf` and `gradf_logpdf` rely on these staying equal.
- `jnp.deg2rad(5 / ARCMIN_PER_DEGREE)**2`, four times in `simulate.py:183-264`; `load_sim(nphi_fac = 2, lmax = 4000)` duplicating named constants.
- `eps = 1e-12` (`lense_flow.py:408`), `1e-10` (`simulate.py:87`), `delta_l = 50`, `17_000`, `40_000`; `2**31` at `sample_lcdm.py:716, 926` (`sampler_phi_noise.py:83` names it `MAX_CHAIN_SEED`).
- `constants.py:91`: `2 * ones.at[0].set(0.5).at[-1].set(0.5)` is right only through operator precedence.
- `wiener_filter.py`: `tol = 1e-1`, `maxiter = 500`.
- `phi_noise_analysis.py:168` (`set_ylim(0.0, 2.0)`), `:163, 176` (`dpi = 150`); `sampler_phi_noise.py:181` (`sweep % 100`) and its argparse defaults at 307-315.
- **NEW** `camb_spectra.py:121`: `polarization_noise = 2 * noise` while `fisher_forecast.py:86` names the same factor `POLARIZATION_NOISE_FACTOR` (and `noise_cls` already returns the E / B noise, which `instrument_matrices` discards). `STORED_NPHI_KEY` is defined in both `fisher_forecast.py` and `delensed_covariance.py`; `FIGURE_DPI` in both `matrix_plots.py` and `chain_analysis.py`. `--pol` choices are `list(POL_DESCRIPTIONS)` in `chain_analysis.py:898` but `POLS` elsewhere.
- **NEW** Unnamed layout literals in `chain_analysis.triangle_plot` (`1.5`, `0.8`, `72`, `0.04`, `1.05`, `0.03`, `0.97`, figure sizes) and in `fisher_forecast.py:298, 312, 326` (`0.5`, `3.0`, `"#4C72B0"`); the forecast CLI's 128 / 2.5 / 2.5 defaults are literals while `chain_analysis.py` names them.

---

## 16. Checked and found correct

So that these are not re-investigated.

**Sampler**
- HMC: leapfrog signs, the Hamiltonian, `log(u) < delta_h`.
- Metropolis: the acceptance test, NaN rejection, bounds rejection as a self-transition (symmetric proposal, detailed balance holds); the candidate logpdf rebuilds Cphi / Cf / G / D from the same predictors and frozen QE as `update_args_after_sample`.
- RNG: the key is advanced before every use in `sample_joint` and `posterior_mixed_draws`.
- The predictor singleton is still the same function objects after the 1D-spline removal, so the static-argument jit kernels do not retrace; no leftover reference to the removed spline or rotation code.

**Forecast (second review)**
- `fisher_from_blocks`: the weights broadcast along the half axis and sum to nside²; `[0,0] > 0` and `det > 0` is a correct positive-definite test for 1×1 and 2×2 blocks; the origin and NaN-N_phi modes are masked for every block; the trace is right.
- `field_blocks` looks components up by name, so it cannot mis-order against `POL_COMPONENTS`; TE is `Re(T conj E) / nside²` and symmetric in the block; the E / B noise is twice the TT noise, matching `noise_cls`.
- `load_merged_covariance` expands the old format on the correct axes; parameters are looked up by name.
- `qe_noise_matrix`: both estimator calls match their definitions and `load_sim`; the zero unlensed BB goes through `_covar_or_zeros`. `camb_cls_at_params` matches `_run_camb`.
- `delens()`: the `FlatS0` pass-through of `eb2qu` / `qu2eb`, the basis flags and n = 10 matching `load_sim`.
- `mixed_theta_matrices(data_set)`: `get_g_matrix_lcdm(cphi, qe)` matches the sampler's calls; `load_sim`'s G is the identity.
- The forecast's order equals the chains' order because both follow `PARAM_ORDER`.

**Statistics**
- The delete-one jackknife in `noise_with_jackknife`, `measured_errors`, `band_median_jackknife`; the Gelman-Rubin formula; the ellipse axis and angle maths.
- `chain_analysis.load_chains`: files are cut to a common length before burn-in; keys are sorted and intersected across parameters.

**Other**
- `camb_grid_interp.screen()`: the knot indices are right for not-a-knot cubic splines.
- The memmap flush-then-sidecar ordering; merge globs do not pick up temp files.
- The `ell**4` phi convention and the per-term `abs` in the QE normalisation match CMBLensing.jl.
- **Shell plumbing:** all six driver → job script → Python chains line up argument by argument. Current counts: `sample_lcdm.sh` 42, `camb_grid.sh` 20, delensed covariance 11 + params (position 9 is now `pol`), map_joint phi noise 9 (`pol` last), sampler phi noise 13 (`pol` at 12, chain seed at 13), performance test 7. An empty `phi_noise` and an empty trailing chain seed both arrive as `""` and become None.
- Every file a test loads is generated and present, with no key-name mismatches.
- No leftover mention of any removed name in tracked code, scripts, docs or notebooks, outside the "Removed features" record in `CLAUDE.md` and this file.
- `camb_splines.py`, `mixing.py` and `__init__.py` have no findings of their own.
