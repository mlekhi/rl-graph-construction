# Code review findings — GraphHARE (June 2026)

*Written by Fadi after a full review of the
pipeline on branch `fadi/review-fixes`. Read this before looking at the
code changes — it explains **why** the current results cannot be used,
and what the evidence is. The companion file `CODE_REVIEW_EDITS.md`
documents every code change made in response.*

First, the important framing: **the infrastructure you built is good.**
The 4-way stratified split is correctly disjoint, the frozen-SAGE protocol
is clean, SAGE early-stops on `sage_val` only, the kNN-pool bookkeeping and
min-degree constraints are correct, and the PPO ratio bookkeeping (storing
the full candidate set per step) is done right. The problems below are
methodological — in the reward design and the evaluation protocol — and
they are the kind that have caught many published authors. They are also
all fixable, and the fixes are now in place.

---

## Finding 1 — The reported gains are a running max over noise (critical)

`best_macro_f1` is the maximum of ~500 noisy per-episode evaluations of
policy_val macro-F1, and `delta_macro_f1 = best − baseline` was reported
as the headline gain. This number goes up with episode count **even if the
policy learns nothing.**

The evidence, from your own `results.json` trajectories (Cora,
`rl_cora_normalized2`):

| run | mean over episodes | std | first-100 mean | last-100 mean | reported "best" |
|---|---|---|---|---|---|
| β=0.0 | 0.8667 | 0.0035 | 0.8663 | 0.8668 | 0.8794 |
| β=1.0 | 0.8663 | 0.0039 | 0.8670 | 0.8660 | 0.8816 |
| β=1.1 | 0.8667 | 0.0037 | 0.8670 | 0.8670 | 0.8820 |
| β=2.0 | 0.8664 | 0.0035 | 0.8662 | 0.8667 | 0.8776 |

Every trajectory is statistically flat at the baseline (0.8663): the
first-100 and last-100 means are identical to within noise — **the policy
never durably improved F1.** Meanwhile, the expected maximum of 500 draws
from N(0.8665, 0.0036) is ≈ baseline + 3.5σ ≈ 0.879 — exactly the range of
all the reported "bests". The "inverted-U with a peak at β=1.1" is the
ordering of noise maxima across runs with identical distributions.

Three independent confirmations:

1. **CiteSeer:** all 7 β values produced *exactly* `best = 0.7242`.
   Macro-F1 on a 334-node val set is quantized; every run's noise ceiling
   hit the same value. (You noticed this — "no F1 differentiation yet" —
   which was the right instinct; it was a red flag for the whole protocol,
   not just CiteSeer.)
2. **The "gain" scales as 1/√(val-set size)** — the signature of an
   extreme-value statistic, not of learning: Cora (271 val nodes) "+1.5pp",
   CiteSeer (334) "+1.1pp", PubMed (~1,970) "+0.2pp".
3. **`final_macro_f1` ≈ baseline in every run** — without cherry-picking
   the max, the improvement vanishes.

**The lesson — always run the null experiment.** Ask: "what would a random
policy's best-over-500-episodes look like?" Answer: baseline + ~1.3pp on
Cora — exactly what was reported. The sanity check you did run ("random
policy gets ~0 reward") could not catch this, because the problem is in
the *selection statistic*, not the reward. From now on: any time a result
comes from a max/min/argmax over noisy evaluations, compute what pure
noise would produce under the same selection before believing it.

**Also:** "beats GraphSAGE (0.866) at all β" compared the max-over-500-evals
on policy_val against a single evaluation of the *same 271 nodes*. The real
GraphSAGE baseline is its **test** F1 (0.866 ± 0.015 across 10 splits), and
no RL run was ever evaluated on test — the entire pipeline contained no
test-set evaluation.

## Finding 2 — The homophily reward leaked test labels (critical)

`_eval_homophily()` computed edge homophily with ground-truth labels over
**all** edges — including edges touching test nodes (~⅓ of edges at a 20%
test split). Every β>0 reward evaluation therefore read test labels. With
a kNN pool that is only ~52% same-label (Cora), a working policy could use
this oracle signal to learn which feature-similar pairs are truly
same-class — including around test nodes — and inflate test accuracy in a
way reviewers would correctly reject.

It did not bite *yet* — only because of Finding 1 (the policy wasn't
learning anything, so it couldn't exploit the leak). That makes it the
most dangerous kind of bug: it would have silently activated the moment
learning started working.

Note the distinction: using Zhu et al. (2020) edge homophily with full
ground truth is **correct as a reporting metric** (and the homophily
diagnostic gate is fine). The error was reusing that same function inside
the **reward**. Labels that may appear in a training signal: train,
sage_val, policy_val. Test labels: never.

## Finding 3 — The implemented reward didn't match the design (major)

Three divergences from RL_DESIGN.md, all in `graph_env.py`:

1. **Levels, not deltas.** The design says reward = change in F1/homophily
   *between consecutive evaluation points*; the code computed
   `current − original-graph-baseline` at every evaluation — a level, not
   a delta. (The unused `episode_start_macro_f1` variable was a fossil of
   the intended design.) With levels, one early good edit pays reward at
   every later evaluation, smearing credit across the episode.
2. **Welford normalization redefined β.** Dividing ΔF1 by the *running std
   of the policy's own deltas* z-scores what is mostly quantization noise
   (271-node F1 moves in ~0.2pp steps), while Δhomophily's std is mostly
   signal. The implemented reward was therefore not the paper's
   `ΔF1 + β·Δhom`, and the scale was non-stationary (it changed as the
   policy changed, plus a discontinuity when the 20-step warmup ended).
   The normalization *requirement* (my 05-29 email) stands — the fix is to
   calibrate **fixed** scales once from a random-policy pre-pass, then
   freeze them.
3. **EMA smoothing on the reward** mixed earlier evaluations into later
   rewards, further blurring credit assignment. Smoothing belongs in
   logging/plots, not in the reward the optimizer sees.

## Finding 4 — PPO mechanics issues (moderate)

- **Dropout mismatch:** rollouts ran under `policy.eval()` (dropout off)
  but updates under `policy.train()` (dropout 0.1 on). The PPO importance
  ratio `exp(new_logp − old_logp)` is then noise even at epoch 1, and
  clipping acts on that noise. PPO policies conventionally use no dropout.
- **Stale state:** the SAGE cache refreshed only every 5 steps, so for 4
  of 5 edits the agent acted on outdated embeddings/degrees — the design
  doc promises updated state per edit.
- **Critic input:** the value head sees only the sampled node's candidate
  set (mean-pooled), not the graph state — its predictions are close to
  meaningless, making advantages noisy. (Known limitation; documented but
  not restructured in this pass.)
- **`entropy_coef=0.05`** (the sweep setting) with near-zero advantages
  makes the entropy bonus the dominant gradient — which actively pushes
  the policy *toward* uniform random. Default restored to 0.01; Optuna now
  searches this properly.

## Finding 5 — Learnable β optimizes the wrong objective (major, for that feature)

In the learnable-β path, advantages are detached, so β receives gradients
**only through the critic's MSE loss**. The optimizer moves β to make
returns easier for the critic to predict — essentially shrinking reward
variance — not to improve performance. The observed decay 1.0 → 0.49 in 20
episodes is that artifact; β would drift the same way on almost any task.
"The network prefers less homophily weighting" is not a supported reading.
A meaningful learnable β needs a bilevel objective (update β to maximize
held-out improvement). The flag now prints a warning; do not report
learnable-β results until this is redesigned.

## Finding 6 — Reporting/process issues (minor but worth internalizing)

- The best *graph* (the actual artifact of the method) was never saved —
  only the policy weights — so even a legitimate best run could not be
  evaluated post-hoc. Now both `best_graph.pt` and `final_graph.pt` are saved.
- `results.json` mixed metrics from different graphs (best-episode F1 next
  to final-episode homophily).
- `05_compile_results.py` treated 0.0 as "missing" (`if d.get(...)`), and
  one stale-schema run (`beta0.1`) sat in the comparison directory.
- All conclusions rested on a single seed (42) — multi-seed was planned,
  but the β=1.1 narrative formed before it ran.

---

## What this means for the project

The honest current state: **no evidence yet that the RL agent improves the
graph** — for any β, including β=0. That is not a dead project; it is the
actual research problem, now visible: the per-edit reward signal on a
271-node quantized F1 is extremely weak, and learning under it is the
thing to solve (longer episodes, richer candidate pools, possibly a soft
classification signal alongside macro-F1, curriculum on the edit budget…).

The fixed pipeline gives you the tools to attack it honestly:

1. `02_freeze_sage.py` per seed → frozen SAGE = GraphSAGE baseline.
2. `08_tune_rl_optuna.py` → tune PPO at β=0 with an honest objective
   (mean of last-50 policy_val F1, not max).
3. `09_multiseed_eval.py` → 10 seeds × β sweep, MLP + GraphSAGE baselines
   on identical splits, test-set evaluation with paired per-seed deltas
   and 95% CIs.

A claim of "GraphHARE beats GraphSAGE" now requires: the paired per-seed
test-F1 delta CI excluding zero. Nothing weaker goes in the paper. If
β>0 and β=0 end up indistinguishable on test, remember the fallback
framing we agreed on in RL_DESIGN.md — that result is publishable too,
just positioned differently.

Finally: please read `CODE_REVIEW_EDITS.md` next, then walk through
`graph_env.py`'s new reward path until you can explain to me (a) why the
homophily reward excludes test-touching edges, (b) why the scales are
frozen after calibration, and (c) why we report mean-last-50 instead of
the max. That conversation is the goal of this review.
