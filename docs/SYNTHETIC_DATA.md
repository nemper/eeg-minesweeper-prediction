# Can we generate more (synthetic) EEG input data?

> **Purpose.** A working discussion and *conversation starter* on whether synthetic
> EEG can meaningfully grow the training set for this project. Drop it into a new
> chat and pick a thread. A debate between **Claude (Opus 4.8)** and **Codex
> (gpt‑5.5)** — see [How this was produced](#how-this-was-produced). Companion to
> [ALGORITHM_IMPROVEMENTS.md](ALGORITHM_IMPROVEMENTS.md).

---

## TL;DR (the honest answer)

**Yes, you can synthesize EEG — but it won't buy you what you actually need.**

- Synthetic EEG can act as **regularization** and as a **stress test** for the
  pipeline. It can plausibly mimic EEG *spectra* and *autocorrelation*.
- It **cannot manufacture trustworthy `Win/Lose` or completion-time labels**. The
  link between brain signal and game outcome is exactly the thing we're trying to
  learn; a generator can only fabricate it.
- The real bottleneck is **n = 13 real sessions, one subject, 3 losses, and
  difficulty confounded with outcome** (see [ALGORITHM_IMPROVEMENTS.md §0](ALGORITHM_IMPROVEMENTS.md#0-what-the-pipeline-does-today-snapshot)).
  No generator fixes a confound or a missing subject. **The most valuable
  "augmentation" is more real data.**

So: use synthetic data for **robustness and null tests**, use *conservative*
augmentation to regularize **inside training folds only**, and keep every reported
number measured on **real held-out sessions**.

---

## The one non-negotiable rule

> **Synthetic samples may enter training folds only. They must never appear in
> validation or test, and augmentation must happen *after* the train/test split.**

If you augment before splitting (or let a generator trained on all data leak into
the test fold), apparent performance will rise and mean nothing. Evaluation stays on
**real, held-out sessions** (ideally leave-session-out). This is the difference
between "the model generalizes" and "we inflated the score".

---

## The spectrum of options

Ordered cheapest/safest → most effort/most fabrication. For each: what it
**preserves**, what it **fabricates**, the **label-fidelity risk**, and the verdict.

### (a) Feature-space augmentation — *on the extracted feature matrix*
Examples: small Gaussian **jitter** on standardized features (the repo already does
a crude version in `regression_prediction_and_evaluation()`), **SMOTE**, **mixup**,
**bootstrapping**.

- Preserves: marginal feature statistics, rough class structure.
- Fabricates: any feature interaction the noise model doesn't know about.
- Label risk: **low–moderate** if jitter is small and done inside the fold.
- Verdict: **cheap regularizer.** Worth trying, but it cannot create new
  *information* — only smooth decision boundaries. SMOTE on 2–3 minority samples
  (see [ALGORITHM_IMPROVEMENTS.md §4](ALGORITHM_IMPROVEMENTS.md#4-class-imbalance))
  is not credible; prefer class weights.

### (b) Signal-space augmentation — *on the raw windows before feature extraction*
Examples: amplitude scaling, additive **Gaussian / pink (1/f) noise**, small
**time-shift**, gentle **time-warp**, **channel dropout/attenuation**, within-session
**window bootstrap**.

- Preserves: most spectral/temporal structure (if perturbations are mild).
- Fabricates: fine phase relationships; aggressive warping distorts the very timing
  that completion-time regression depends on.
- Label risk: **moderate.** A noisier version of a *winning* session is still
  plausibly a win; a *time-warped* one is a dubious completion-time label.
- Verdict: **the most defensible augmentation here.** It models real sensor/biology
  variability and forces features to be robust. Keep it mild; keep it in-fold.
  ⚠️ **Avoid cross-session window recombination** — it destroys session identity and
  leaks across the split.

### (c) Generative models — *learn the distribution, sample from it*
Examples: **VAE / GAN / diffusion** for EEG; **AR / colored-noise (1/f) + band-
limited oscillators**; **phase-randomization / IAAFT surrogates**.

- Preserves: plausible spectra; surrogates preserve amplitude distribution and power
  spectrum exactly.
- Fabricates: neural dynamics, cross-channel coupling, and — critically — the
  cognition→outcome relationship. Surrogates deliberately **destroy** phase/temporal
  structure.
- Label risk: **high** for supervised use. A GAN trained on 13 sessions will mostly
  memorize/overfit.
- Verdict: **not for supervised augmentation here**, but **excellent for null
  tests**: generate **phase-randomized surrogates** and check whether the classifier
  *still* performs — if it does, it was exploiting spectral/session confounds, not
  real temporal structure. Simple **1/f + oscillator** simulators are great for
  *unit-testing* `eeg_feature_extraction()` (does relative band power behave?).

### (d) Physiology / forward-model simulation
Examples: neural-mass / oscillator networks, **MNE** source→sensor forward
simulation.

- Preserves: biologically plausible source/sensor structure; controllable rhythms
  and artifacts.
- Fabricates: this subject's Emotiv-specific physiology and the gameplay
  relationship entirely.
- Label risk: **very high** for our labels.
- Verdict: **high effort, low payoff** unless the project pivots to EEG *simulation*
  research. Skip for now.

---

## What to actually do (recommendation)

**Try first (in-fold, real-data evaluation):**
- Small feature-space jitter on standardized training features.
- Mild raw-signal amplitude scaling + Gaussian/pink-noise injection + light channel
  attenuation.
- Optional within-session window **bootstrap** for robustness estimates.
- Replace SMOTE with **class weighting**.

**Use as diagnostics, not training data:**
- **Phase-randomized / IAAFT surrogates** and **label-permutation** runs as null
  tests — "does the model beat its own chance distribution?" At n = 13 this matters
  more than any augmentation.
- **1/f + oscillator** synthetic signals to unit-test the feature code.

**Avoid for now:**
- GAN/VAE/diffusion EEG generation as training data.
- Cross-session window recombination; aggressive time-warping (especially for
  regression).
- Treating any synthetic sample as an independent "session".
- Reporting **any** metric measured on augmented validation/test data.

**The best "augmentation" is real data:** more sessions from this subject, **more
losses**, balanced difficulty×outcome combinations, ideally **more subjects**, and
repeated sessions across days to quantify session drift. Synthetic data can
regularize a model; it cannot create evidence that EEG predicts Minesweeper
outcome or time.

---

## A concrete starter experiment

1. Build the **EEG-only, leakage-free** matrix from
   [ALGORITHM_IMPROVEMENTS.md §1](ALGORITHM_IMPROVEMENTS.md#1-correctness-bugs-fix-before-anything-else).
2. Establish honest **leave-session-out** baselines (majority, difficulty-only).
3. Add **mild in-fold signal-space augmentation**; check whether LOOCV
   balanced-accuracy / MAE improves **vs no augmentation**.
4. Run a **permutation test** and a **phase-randomized surrogate** test; if scores
   survive surrogates, investigate the confound before celebrating.
5. Document the delta. If augmentation only stabilizes variance (not mean skill),
   say so — that is still a useful result.

---

## How this was produced

- **Codex (gpt‑5.5)** did a read-only analysis of the data format and pipeline and
  produced the option-by-option breakdown (preserves/fabricates/risk) and the
  "try first / avoid" lists.
- **Claude (Opus 4.8)** verified the dataset confounds, framed the non-negotiable
  evaluation rule and the TL;DR, and integrated both views.

## Open questions to start a new chat

- Is the aim to **boost a demo's apparent skill**, or to **honestly test** whether
  EEG carries signal? (Synthetic data helps the first far more than the second —
  and the second is the scientifically meaningful one.)
- Can we record **more real sessions / more subjects** instead? That dominates every
  synthetic option.
- Do we want to implement the **surrogate-based null test** first, as a reality
  check on the current ~75–100 % accuracies?
