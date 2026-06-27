# Improving the EEG → Minesweeper algorithm

> **Purpose.** A working design document and *conversation starter*. Drop it into a
> new chat and say "let's work on item X". Every point is grounded in the current
> code ([main.py](../main.py)); function names are cited so you can jump straight in.
> It was written jointly by **Claude (Opus 4.8)** and **Codex (gpt‑5.5)** — see
> [How this was produced](#how-this-was-produced).

---

## 0. What the pipeline does today (snapshot)

`main.py` is a linear pipeline:

`extract_data()` → `processing()` → `windowing()` → `eeg_feature_extraction()` →
`creating_the_feature_matrix()` → `classification_prediction_and_evaluation()` →
`regression_prediction_and_evaluation()` → `metrics_visualization()`.

It turns **13 Minesweeper sessions** (one subject, Emotiv INSIGHT, 5 channels —
AF3/T7/Pz/T8/AF4 — amplitude at 128 Hz + `POW.*` band power) into **one feature
row per session**, then predicts **Outcome** (Win/Lose, classification) and
**Time** (completion time, regression).

The dataset reality (verified from the JSON annotations):

| Property | Value |
|---|---|
| Sessions (samples) | **13** |
| Outcome balance | **10 Win / 3 Lose** → majority-class baseline = **76.9 %** |
| Difficulty | 5 Easy, 5 Normal, 3 Advanced |
| Difficulty × Outcome | Easy → **5/5 Win**; losses only in Normal (1) & Advanced (2) |
| Fields / Mines | Perfectly determined by Difficulty (81/10, 256/40, 480/99) |
| Time range | 17 s – 293 s |

Everything below is prioritized: **fix correctness first, then evaluation, then
features.** With n = 13 the headline is sobering — treat results as a *pilot*, not
evidence that EEG predicts Minesweeper performance.

---

## 1. Correctness bugs (fix before anything else)

### 1.1 Target leakage through the game-annotation columns 🔴
`creating_the_feature_matrix()` appends the game annotations (`Time`, `Outcome`,
and optionally `Difficulty`/`Fields`/`Mines`) as feature columns. Then:

- `classification_prediction_and_evaluation()` pops **only** `Outcome` — so
  **`Time` (and Difficulty/Fields/Mines) stay in `X`** while predicting Win/Lose.
  Completion time is hugely predictive of outcome → the classifier can "win" by
  reading game metadata, not EEG.
- `regression_prediction_and_evaluation()` pops **only** `Time` — so **`Outcome`
  stays in `X`** while predicting completion time → same leak in reverse.

**Fix:** decide explicitly what each model is allowed to see and report **three
regimes**: (a) **EEG-only**, (b) **pre-game context-only** (difficulty), (c)
**EEG + pre-game context**. Never feed post-game outcomes (`Outcome`, `Time`) as
predictors of each other. This single change will likely *lower* the numbers and
*raise* their credibility.

### 1.2 Redundant / confounded context features
`Fields` and `Mines` are exact functions of `Difficulty` (see table) — keeping all
three triples a single signal. And **Difficulty is confounded with Outcome** (Easy
is always a win), so a "Difficulty-only" model is a strong baseline that any
EEG-based claim must beat.

### 1.3 The Random-Forest `n_estimators` no-op
In `classification_prediction_and_evaluation()`:
```python
param_grid["n_estimators"]: [100, 200]   # ← a type annotation, not an assignment
```
This line does nothing, so `n_estimators` is never tuned. Use `=`. (Low impact, but
it is silently dead code.)

### 1.4 `max_features` / grid sanity
The tree/RF grid was modernized to `["sqrt", "log2"]`. Keep grids *small* (see §5).

---

## 2. Evaluation & leakage methodology (highest-impact after §1)

The current scheme — `train_test_split(test_size=0.3)` repeated `number of runs`
times, with `GridSearchCV(cv=3)` inside the ~9-sample train split — is not
trustworthy at n = 13:

- The **test set is ~4 games**; one flipped prediction moves accuracy by 25 %.
- With `use fixed random state = True`, every "run" uses the **same split**, so the
  repeats add no information (they look like a distribution but aren't).
- `StandardScaler` and `SMOTE` are fit **before** `GridSearchCV`, so inner CV folds
  see information from the whole training set → optimistic tuning.

**Do instead:**

1. **Leave-Session-Out / LOOCV** (or repeated stratified k-fold) and **pool the
   out-of-fold predictions** into a single confusion matrix / report. With 13
   samples, LOOCV gives 13 honest held-out predictions.
2. Put preprocessing **inside** the CV via `imblearn.pipeline.Pipeline([scaler,
   smote, estimator])` (and a sklearn `Pipeline` for regression) so scaling/SMOTE
   are fit on train folds only.
3. If you keep hyperparameter search, use **nested CV** (outer LOOCV, inner small
   grid) — otherwise tuning leaks into the score.
4. Always report **baselines**: majority-class (76.9 %), Difficulty-only, and
   median-time for regression. An EEG model only matters if it beats these.
5. Add a **permutation test** (shuffle labels, re-run CV) to get a chance
   distribution — essential at this n.
6. Report **uncertainty**: exact binomial CIs for accuracy (they will be wide,
   that's the point); for regression report MAE/median-AE in **seconds** with
   per-session errors rather than R² on ~4 points.

`metrics_visualization()` currently **averages confusion matrices and
classification reports across runs** — replace with the single pooled out-of-fold
matrix.

---

## 3. Feature engineering

`eeg_feature_extraction()` is where most signal is left on the table.

### 3.1 Preprocess the raw signal first
There is **no filtering, detrending, baseline correction, or artifact handling**.
Raw amplitude mean/variance is dominated by DC offset and drift. The CSVs even ship
quality columns (`CQ.*`, `EQ.*`, `EEG.Interpolated`) that `processing()` discards.
Add band-pass + notch filtering, detrend, drop/----flag low-contact-quality windows,
and consider re-referencing.

### 3.2 The "correlation" and "cross-covariance" features are not what they sound like
- `Corr.*` comes from `np.corrcoef()` between **whole flattened windows**, then
  collapsed to two scalars (`Corr.mean`, `Corr.var`). That's window-to-window
  self-similarity, **not electrode connectivity**.
- `Xcov.*` uses `correlate2d(..., mode="valid")` — effectively an **uncentered dot
  product** between windows — then hard-divides by `/1e10` and `/1e19` to tame the
  scale.

**Fix:** compute real **channel-pair connectivity** (pairwise correlation,
coherence, phase-locking value, band-specific), Fisher-z transform it, and either
keep the pairs (only 10 for 5 channels) or compact graph summaries. Delete the
magic `/1e10`,`/1e19` scaling and let a `RobustScaler`/`StandardScaler` **inside the
CV pipeline** handle magnitudes.

### 3.3 Use band power properly
Today only **absolute per-channel band means** are used. Add:
- **relative** band power (band / total) and **log** power,
- ratios: **theta/beta**, alpha/beta, beta/(alpha+theta),
- **frontal alpha asymmetry** (AF4 − AF3) and temporal asymmetry (T8 − T7),
- per-band **variability and trend** across windows.

### 3.4 Keep temporal dynamics
`windowing()` makes 2 s windows but `eeg_feature_extraction()` **averages every
feature across all windows** into one vector — throwing away within-session
dynamics. Add window-level mean/std/median/IQR, **slope**, and **early-vs-late
deltas** (engagement often drifts over a game). Or model at the window level with
**session-grouped** CV (never split windows of one session across train/test).

---

## 4. Class imbalance

`SMOTE(k_neighbors=1)` synthesizes the minority "Lose" class from only ~2 training
examples in a high-dimensional space — not credible here. Prefer **class-weighted**
estimators (`class_weight="balanced"`), evaluate with **balanced accuracy / macro-F1
/ recall-on-Lose**, and — most of all — **collect more loss sessions**.

---

## 5. Regression target & models

- `Time` mixes two regimes: a win's **completion time** and a loss's **time-to-
  failure**. Model them separately, or jointly with Outcome. Consider `log(Time)`
  or difficulty-normalized time as the target.
- The "augmentation" `X_train + 0.01 * normal(...)` in
  `regression_prediction_and_evaluation()` perturbs **already-extracted features**
  (and any leaked context columns) and duplicates labels — unlikely to help. Drop
  it or move augmentation to the raw-signal stage (see [SYNTHETIC_DATA.md](SYNTHETIC_DATA.md)).
- Grids are **too large for the data** (SVC ≈ 24, tree/RF ≈ 32, MLP ≈ 48 combos with
  cv=3 on ~9 samples). Start with **regularized linear / SVC**, shallow trees, or
  fixed hyperparameters; expand only if honest CV justifies it.
- Regression tuning is **inconsistent**: `GridSearchCV` runs on *unscaled* augmented
  features, then the final model is fit on *scaled* features — selection and final
  fit use different distributions. Wrap everything in one `Pipeline`.

---

## 6. Reproducibility & hygiene

- `simplefilter("ignore")` hides important warnings (MLP non-convergence, invalid
  metrics). Keep warnings on during experiments.
- `extract_data()` sorts value/interval/JSON files **independently** and assumes
  index alignment. Match by **session id** instead.
- `remember_parameters()` logs parameters but not **split indices, chosen
  hyperparameters, package versions, data hash, or git commit** — add these for
  auditability.
- `scorings_for_grid_search()` contains typos (`precission_*`, duplicate
  `neg_mean_squared_log_error`); validation only checks this local list. Validate
  against sklearn's scorer registry.

---

## 7. Suggested first experiment (concrete)

1. Strip leakage: build an **EEG-only** feature matrix (§1.1) and a **Difficulty-
   only** baseline matrix.
2. Evaluate with **LOOCV**, pooled out-of-fold predictions, class-weighted SVC and
   logistic regression, scaling inside the fold.
3. Report balanced accuracy + macro-F1 + Lose-recall **vs** majority and Difficulty-
   only baselines, with a **permutation test** p-value.
4. For regression: ridge on `log(Time)`, LOOCV, MAE in seconds vs median baseline.
5. Only then revisit richer features (§3) and augmentation.

If EEG-only barely beats the majority/Difficulty baselines, that is the honest,
publishable finding for a 13-session pilot — and the strongest argument for
collecting more data.

---

## How this was produced

- **Codex (gpt‑5.5)** did a read-only pass over `main.py` and produced the bulk of
  the structured findings — notably the **target-leakage** discovery (§1.1), the
  connectivity/cross-covariance critique (§3.2), and the validation plan (§2).
- **Claude (Opus 4.8)** verified the dataset facts (§0 table), framed and
  prioritized the document, and integrated both sets of notes.

## Open questions to start a new chat

- Which regime do we actually care about: **EEG-only** prediction, or EEG **plus**
  pre-game difficulty?
- Is the goal a *publishable pilot* (rigorous, likely-negative) or a *demo*
  (engaging, less strict)? That decision changes almost every item above.
- Can we get more sessions — especially **more losses** and **balanced difficulty**?
