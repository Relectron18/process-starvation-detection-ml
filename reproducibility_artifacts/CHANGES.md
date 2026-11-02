# Methodology fixes (this revision)

A review of this package against the paper draft found four issues that
could bias reported numbers. All four are fixed in this revision; every
affected experiment was rerun. This file records what changed and by how
much, so the fixes are auditable rather than silently baked in.

## 1. Feature/scaling leakage (pipeline_core.py)

`fit_queueing_theory_stats()` (which builds `pk_wait_anomaly`), the
`waiting_time_ms` median (which builds `low_priority_high_wait`), and the
`LabelEncoder`/`StandardScaler` pair were all previously fit on the full
dataset **before** the train/test split, in `build_dataset()` +
`preprocess()`. Test rows leaked into the statistics used to build features
and scale inputs for the model later evaluated on those same rows.

**Fix:** `pipeline_core.split_raw()` now splits first (on row-wise-only
features), and `fit_split_stats()` / `fit_encoders_scaler()` fit everything
cross-row on the training partition only; `apply_split_stats()` /
`transform_with()` apply those fitted statistics to test (and validation)
without refitting. `build_dataset()`/`preprocess()` are kept only for the
non-split smoke tests in `tests/test_pipeline_core.py`, where leakage is
not a concern, and are documented as such. Regression tests were added
(`tests/test_pipeline_core.py`) that assert a test row's features/scaling
cannot change based on what else is in the test partition.

Used by: `train_model.py`, `multiseed_stability.py`,
`feature_access_and_model_family_check.py`.

## 2. Decision threshold tuned on the test set (train_model.py)

The threshold applied to real-hardware predictions (Table 2, Section
5.4-5.5) was chosen by maximizing F1 directly on `y_test`, then that same
test set's score was implicitly treated as an unbiased estimate.

**Fix:** `train_model.py` now carves a validation split out of the
training partition (60/20/20 train/val/test overall). Model selection,
hyperparameter tuning (`RandomizedSearchCV`), and threshold tuning all use
validation only. The chosen model+threshold is refit on train+val, and the
test set is touched exactly once, at the end, to report final metrics.

## 3. Selection bias in the HRRN comparison (multiseed_stability.py)

The ML side of the HRRN comparison (Section 5.2) previously took
`max(seed_f1.values())` — the best of 5 models' **test** F1 scores per
seed — before comparing to HRRN, a "winner's curse" that inflates the
margin.

**Fix:** the compared model is now selected by highest mean
cross-validated F1 on the training data (computed before test is touched),
exactly as a real deployment decision would be made. The margin over HRRN
is essentially unchanged (see below) because the effect size is large
enough that model choice barely matters — but the comparison is no longer
methodologically circular.

## 4. No uncertainty on small-denominator real-hardware precision/recall

Real-hardware precision/recall are each computed over ~17-26 samples per
trial; a per-trial "1.000 ± 0.000" reads as more solid than a 3-5-trial
std actually supports.

**Fix:** `aggregate_real_world.py` now reports a per-trial Wilson score
95% CI for precision and recall alongside the point estimate (see
`real_world_multitrial_results.csv`).

---

## What changed in the reported numbers

| Result | Before | After | Changed? |
|---|---|---|---|
| Table 1 (per-model F1, accuracy, etc.) | — | — | Point estimates ~unchanged (<0.5pt) |
| RF vs XGBoost F1, paired t-test | t=2.94, **p=0.043** | t=1.70, **p=0.165** | **No longer significant** |
| Best-ML vs HRRN F1, paired t-test | t=40.0, p=2.3e-6 | t=38.7, p=2.7e-6 | Unchanged (robust either way) |
| pk_wait_anomaly ablation (Sec 5.3), RF, 5 seeds | delta=-0.0004, p=0.777 | delta=-0.0023, p=0.126 | Still null, larger magnitude |
| pk_wait_anomaly x MLP (Sec 5.6a) | delta=-0.0019, p=0.431 | delta=-0.0028, p=0.356 | Still null |
| pk_wait_anomaly, reduced access (Sec 5.6b) | delta=+0.0043, **p=0.009** | delta=+0.0043, **p=0.051** | **No longer significant at α=0.05** |
| Real-hardware RT: Accuracy / Precision / Recall / F1 | 0.975 / 0.991 / 0.908 / 0.945 | 0.929 / 0.989 / 0.721 / 0.833 | Recall & F1 drop materially |
| Real-hardware nice: Accuracy / Precision / Recall / F1 | 0.941 / 1.000 / 0.773 / 0.872 | 0.913 / 1.000 / 0.667 / 0.800 | Recall & F1 drop materially |

The most consequential change in the first revision was **#3 in the table
above**: the paper's own positive resolution to its stated Future Work
question (Section 5.6b) was significant at p=0.009 under the leaky
pipeline and dropped to p=0.051 (not significant) once fixed, though the
direction and magnitude of the effect were unchanged. The real-hardware
recall/F1 drop in that revision was a side effect of fix #2: the
leak-free threshold, tuned on validation instead of directly on the
metric that used to get reported, sits at a more conservative operating
point. Precision was essentially unchanged (still ~0.99-1.00) — the model
became more conservative, not less accurate when it does flag something.
The second revision below (items 1-7) resolves the p=0.051 ambiguity with
more seeds and adds several new robustness checks.

---

# Second revision: statistical power, multiple comparisons, and three new robustness checks

The first revision (above) fixed leakage and threshold-tuning bugs but left several
things unresolved or unexamined. This revision addresses seven follow-up items.

## 1. Notebook synced with the fixed pipeline

`starvation_detection_project.ipynb` previously still ran the original leaky
pipeline (build stats on the full dataset, tune the threshold on the test
set) even after `pipeline_core.py`/`train_model.py` were fixed -- so the
notebook and the scripts disagreed. The notebook's split/preprocessing/
threshold-tuning cells now mirror the fixed scripts exactly (three-way
train/val/test split, stats fit on train only, threshold tuned on
validation, final refit on train+val before saving). A correction notice
was added near the top of the notebook. EDA plots that render *before* the
split still use full-dataset statistics for `pk_wait_anomaly`/
`low_priority_high_wait` -- clearly noted as fine for visualization, not
evaluation.

## 2. More seeds for the borderline p=0.051 result (Section 5.6b)

Reran `feature_access_and_model_family_check.py` with 12 seeds instead of
5 (same expanded seed set now used everywhere for consistency). Result:
**mean delta=+0.0038 F1, p=0.0020** (was p=0.051 at n=5) -- the effect is
real, not a coin flip that happened to land on the wrong side of 0.05.
Direction and magnitude are consistent with the 5-seed result; only the
statistical power changed.

A side effect of the larger, more consistent seed set surfaced a second,
unexpected change: **Section 5.3's own ablation (RF, full feature access)
is now also significant, in the *negative* direction** (mean delta=-0.0028,
p=0.0098) -- previously null at p=0.126 (5 seeds) and p=0.777 (an even
earlier single-seed check). With 12 seeds, the engineered feature set does
measurably (if slightly) *hurt* accuracy when the raw columns it overlaps
with are already available, rather than being merely neutral. This
sharpens rather than undermines the paper's redundancy story: not "no
effect," but a small real cost when redundant, and a real benefit when not.

**5.1 (Random Forest vs. XGBoost)** flipped again with the larger seed
set: p=0.165 (5 seeds) -> **p=0.0032** (12 seeds), now favoring Random
Forest. This is reported as a caution against reading too much into any
single seed-count's p-value in a small-sample paired design -- see item 3.

## 3. Multiple-comparisons correction

Added `significance_summary.py`, which applies Holm-Bonferroni correction
across all 5 formal significance tests in the paper. Results (12 seeds):
4 of 5 remain significant after correction (5.1, 5.2, 5.3, 5.6b); only
5.6(a) (the MLP check) stays null. See `multiple_comparisons_correction.csv`.

## 4. Leak-free single-seed checks (Section 5.3 i/ii)

`single_seed_ablation_check.py` reruns the original single-seed RF and LR
ablations (previously retained from the leaky notebook with a "predates
the fix" caveat) on the corrected pipeline. Both now show small negative
deltas (RF: -0.0013, LR: -0.0025), consistent in direction and magnitude
with the well-powered 12-seed result in item 2. The "predates the fix"
caveat is removed from the paper -- all three 5.3 checks are now leak-free
and agree.

## 5. Cross-seed real-hardware generalization check

Table 2 only ever validated a single trained model (seed=42) against real
hardware. `multiseed_real_world_check.py` trains two more models (seeds 7
and 123; reduced RandomizedSearchCV budget, n_iter=8 vs. 20, for compute
reasons -- noted as a confound below) and applies each, independently, to
all 8 real-hardware trials.

**Finding, reported honestly because it's a real limitation, not swept
under the rug:** the three seeds' real-hardware F1/precision/recall vary
substantially (RT scenario F1: 0.698-0.833; nice scenario F1: 0.800-1.000),
and two of the three seeds badly over-flag on the RT scenario (precision
0.54-0.55 vs. seed 42's 0.99). However, **ROC-AUC is consistently high
across all three seeds on both scenarios (0.975-1.000)** -- meaning the
underlying model score still ranks real-hardware windows correctly in all
three cases; what varies across seeds is the *decision threshold*
selected from synthetic validation data, which does not reliably transfer
to real hardware. This reframes Table 2 more precisely: the model's
discriminative signal appears to generalize robustly; the specific
operating threshold does not, and Table 2's headline F1/precision/recall
numbers should be read as characteristic of seed 42's model, not a
seed-independent property, until this is tested with matched tuning
budgets across seeds (the current check used a reduced budget for seeds 7
and 123, which may be a partial confound and does not by itself rule out
that a full tuning budget would close the gap).

## 6. Threshold-sensitivity check extended to all 8 trials

Previously run on 1 representative trial only. `aggregate_real_world.py`
now runs all 6 label-threshold combinations against all 8 real-hardware
trials (48 combinations total): F1 ranges 0.780-0.878 (mean 0.813, std
0.020), precision 0.889-1.000, n_starved 23-26 of 96 -- consistent with
the single-trial check, now with full trial coverage.

## 7. Permutation importance as direct evidence for the redundancy story

Added to `feature_access_and_model_family_check.py`: permutation
importance (Breiman 2001) of `pk_wait_anomaly`, computed on a held-out
slice of the training partition, under full vs. reduced feature access.
Mean importance jumps from 0.0087 (full access, average rank 8-9 of 19
features) to 0.076 (reduced access, average rank ~3-4 of 17 features)
across the 12 seeds -- a more direct, causal-ish signal than the
correlation-based argument alone, and consistent with the ablation-based
redundancy story in item 2/4.

## Updated before/after table (supersedes the first revision's table for items marked *)

| Result | First revision | This revision |
|---|---|---|
| RF vs XGBoost F1, paired t-test (n seeds) | t=1.70, p=0.165 (n=5) | *t=3.75, p=0.0032 (n=12), Holm-significant* |
| Best-ML vs HRRN F1, paired t-test | t=38.7, p=2.7e-6 (n=5) | t=87.6, p=5.4e-17 (n=12), Holm-significant |
| pk_wait_anomaly ablation, Sec 5.3 (RF, full access) | delta=-0.0023, p=0.126 (n=5) | *delta=-0.0028, p=0.0098 (n=12), Holm-significant, negative* |
| pk_wait_anomaly x MLP, Sec 5.6a (full access) | delta=-0.0028, p=0.356 (n=5) | delta=+0.0006, p=0.773 (n=12), still null |
| pk_wait_anomaly, Sec 5.6b (reduced access) | delta=+0.0043, p=0.051 (n=5) | *delta=+0.0038, p=0.0020 (n=12), Holm-significant* |


---

# Third revision: reviewer-facing polish, two new robustness checks, one bug fix

This revision responds to a pre-submission review pass that found one remaining
methodological confound worth closing, one useful additional control, and a few
presentation bugs. No previously reported numbers changed as a result of this
revision except where noted below.

## 1. Reference list numbering bug

The reference list had `[10]` and `[11]` each used twice (the EMC and Apple/
Callisto patents, then reused for the XGBoost paper and the Silberschatz
textbook). Renumbered: XGBoost is now `[12]`, Silberschatz is now `[13]`. No
in-text citations were affected (the paper cites by author name in prose, not
by bracket number, so this was a reference-list-only bug).

## 2. Figure 1 caption seed count

Figure 1's caption still read "F1 distribution across 5 independent
synthetic-data seeds" from before the seed count was expanded to 12 throughout
the paper. Corrected to 12, matching Table 1 and the surrounding text.

## 3. Real-hardware environment disclosure

`collect_real_data.py`'s own docstring notes it runs on "this sandbox's single
CPU core" -- the real-hardware data collection environment is a virtualized/
containerized sandbox, not dedicated bare-metal hardware. This was implied but
not stated explicitly in the paper. Added an explicit note in Section 4.2 and
Section 6 (Limitations): the host/hypervisor's own scheduling could in
principle introduce noise beyond what a dedicated machine would have, though
the RT scenario's SCHED_FIFO preemption is independently verified via
`os.sched_getscheduler` in `collect_real_data.py` and is a genuine kernel-level
effect regardless of virtualization.

## 4. New control for Section 5.6(b)'s redundancy claim

Section 5.6(b) shows that withholding `waiting_time_ms`/`wait_to_burst_ratio`
(columns pk_wait_anomaly is correlated with, r=0.27-0.39) turns
pk_wait_anomaly's ablation delta significantly positive. That's consistent
with a redundancy story, but withholding *any* two columns shrinks the
feature set by the same amount -- part of the effect could just be "fewer
competing features helps," not specifically redundancy.

Added `control_ablation_check.py`: identical reduced-access ablation, but
withholding two columns with near-zero correlation with pk_wait_anomaly
instead (`memory_usage_percent`, `num_threads`; |r| < 0.02 across all 12
seeds, vs. 0.27-0.39 for the columns withheld in 5.6b). Result: mean delta =
-0.0010 F1 (std=0.0018), t=-1.89, p=0.0851 -- not significant, and numerically
in the *opposite* direction from the redundant-column condition. This
directly supports the redundancy interpretation: removing unrelated columns
does not rescue pk_wait_anomaly's marginal value, while removing the columns
it overlaps with does. Reproduces `control_ablation.csv` and
`pk_wait_anomaly_correlations_control.csv`. Written up as a new paragraph in
the paper's Section 5.6(b).

## 5. Matched-tuning-budget rerun closes the cross-seed confound

Section 5.4's cross-seed real-hardware check (models trained on seeds 7 and
123, applied to all 8 real-hardware trials) originally used a reduced
RandomizedSearchCV budget (`n_iter=8`) for the two additional seeds vs. seed
42's `n_iter=20`, flagged as a partial confound in the paper's own
Limitations section.

Retrained both seeds with the full `n_iter=20` budget
(`models_seed7_matched/`, `models_seed123_matched/`) and reran
`multiseed_real_world_check.py`. Result: essentially unchanged from the
original reduced-budget run --

| | seed42 | seed7 (matched) | seed123 (matched) | seed7 (orig, n_iter=8) | seed123 (orig, n_iter=8) |
|---|---|---|---|---|---|
| RT F1 | 0.833 | 0.698 | 0.709 | ~0.70-0.83 range as reported | ~0.70-0.83 range as reported |
| RT precision | 0.989 | 0.537 | 0.549 | 0.54-0.55 | 0.54-0.55 |
| nice F1 | 0.800 | 1.000 | 1.000 | 0.800-1.000 | 0.800-1.000 |
| ROC-AUC (both scenarios) | 0.999-1.000 | 0.999/1.000 | 0.975/1.000 | 0.975-1.000 | 0.975-1.000 |

This rules out tuning budget as the explanation for the threshold-transfer
finding: the original reduced-budget run and the matched-budget rerun agree
closely. The threshold-transfer limitation reported in Sections 5.4 and 6 is
therefore a genuine property of the approach (the decision threshold selected
from synthetic validation data does not reliably transfer to real hardware
across independently trained models), not an artifact of unequal tuning
effort. The original reduced-budget run is kept for provenance as
`multiseed_real_world_results_reduced_budget_ORIGINAL.csv`;
`multiseed_real_world_results.csv` now holds the matched-budget numbers, which
are what Sections 5.4 and 6 report as of this revision. `models_seed7/` and
`models_seed123/` (the original reduced-budget artifacts) are kept alongside
`models_seed7_matched/`/`models_seed123_matched/` for the same reason.

## 6. Data and Code Availability section added

Added a short section before References pointing to this artifacts package,
consistent with most venues' reproducibility requirements. Author names,
affiliation, and archival/DOI location remain withheld pending venue
submission, as in the rest of this draft.

# Peer review revision

Changes made in response to the ICMLDE 2026 reviews. Code and results are in `revision/`.

1. Added 30 trials in an Ubuntu 24.04 virtual machine: replication of RT and nice, application style workloads, overcommit, cgroup throttling and a two CPU run.
2. Added an episode label for `victim_normal` that does not depend on model features. Under it the seed 42 model detected 0 of 289 windows, so the claim that learned models beat HRRN now covers synthetic data only.
3. Added class imbalance strategies, grouped and real hardware permutation importance, simple comparison rules on the same windows, and collection overhead measurements.
4. Corrected the feature importance rise from ninefold to about eightfold, and the scheduler description, since kernel 6.18.5 uses EEVDF rather than CFS.
5. Corrected the record of class weighting: Gradient Boosting and XGBoost were not class weighted.
