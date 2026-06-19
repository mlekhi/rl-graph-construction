# GraphHARE — Weekly Summary
_June 11 – June 19, 2026_

---

## Headline: last week's "gains" were measurement artifacts

A pipeline audit found two methodological bugs that invalidated every RL result so
far. After fixing them and running an honest protocol, **GraphHARE matches the
frozen-SAGE baseline at every β on all three datasets — no significant gain, no
inverted-U.** The β=1.1 peak and the +1.57pp Cora result from last week were noise,
not signal.

This is a clean negative result on homophilous citation graphs. The current focus is
to finish the comparison table and move into writing.

---

## What Went Wrong (and is now fixed)

| Bug | Effect | Fix |
|---|---|---|
| **Reward leak** | homophily reward used ground-truth labels on test-incident edges (~1/3), leaking test info on every β > 0 run | reward homophily restricted to non-test edges; full-edge homophily kept for *reporting only* |
| **Noise-max metric** | `best_macro_f1` = max over ~500 noisy policy_val evals → selection-on-noise; manufactured the β peak/curve | honest `mean_last50` metric; **test set evaluated once** after training |
| no test eval | none of the numbers could go in the paper | `evaluate_graph(split="test")` on original/best/final graphs |
| reward non-stationarity | running-stat normalization drifted | consecutive deltas + fixed scales calibrated once via random-policy pre-pass |

Confirmed the fixes are in the merged code and the pipeline ran on them.

---

## Honest 10-Seed Results (test macro-F1, 95% CI, seeds 42–51)

| Dataset | MLP | GraphSAGE | GraphHARE (best β) | Δ vs SAGE |
|---|---|---|---|---|
| Cora | 0.7322 | 0.8659 | 0.8665 (β=1.0) | **+0.0005 ± 0.0025** |
| CiteSeer | 0.6819 | 0.7144 | 0.7148 (β=1.0) | **+0.0004 ± 0.0012** |
| PubMed | 0.8743 | 0.8834 | 0.8835 (β=1.0) | **+0.0001 ± 0.0002** |

Every delta's CI includes zero. β has no significant effect. Baselines reproduce our
earlier Optuna numbers exactly (Cora 13pp SAGE-MLP gap, CiteSeer 3pp, PubMed 1pp), so
the measurement chain is sound — the null result is real.

Pipeline: `08_tune_rl_optuna.py` (PPO Optuna tuning) → `09_multiseed_eval.py`
(10 seeds, MLP/SAGE baselines on identical splits, paired test deltas + CIs).

---

## Why Zero — Oracle Headroom Analysis

To check whether the null is a method failure or a property of the data, measured the
ceiling on test-F1 from graph editing under different information:

| Oracle (information used) | Cora | CiteSeer |
|---|---|---|
| full (all true labels, leaky) | +0.109 | +0.162 |
| fair (true labels, non-test edges only) | +0.010 | −0.004 |
| **pred-reachable (SAGE predictions only — what the policy has)** | **+0.009** | **~0** |

The graph *can* matter (~11–16pp under a perfect oracle), but almost all of it is
locked behind true labels the method cannot use. With only the frozen SAGE's own
predictions, reachable test gain is ~0.9pp on Cora and ~0 elsewhere. The policy
converges near the original graph because there is little prediction-reachable
signal on these already-homophilous graphs.

---

## Comparison Baselines (for the paper table)

Added five GNNs + a graph-autoencoder to the baseline harness, same Optuna +
multi-split protocol as SAGE/MLP. Smoke-tested locally (all run, correct shapes).
Full Optuna + 10-split runs are now running on gpu2.

- **GCN, GAT, APPNP, GCNII, GraphTrans** (PyG node-classification layers)
- **GraphAE** (GAE pretrain → linear probe on embeddings)
- **GraphRARE** — cited from its paper (RL graph method; reproduction out of scope).
  TODO: confirm it reports Cora/CiteSeer/PubMed (it targets heterophilic graphs, so it
  may not — may need to note this).

---

## Open Questions for Next Meeting

1. How to frame the homophilous null result: negative result, diagnostic result, or
   motivation for revising the reward/policy design?
2. Whether the paper should emphasize the protocol correction + oracle headroom
   analysis as the main finding for this phase.
3. Be ready to walk through the new reward path in `graph_env.py`.

---

## Next Steps

1. Finish extended baseline table (GCN/GAT/APPNP/GCNII/GraphTrans/GraphAE) on all 3 datasets, gpu2.
2. Pull/scp baseline summaries and assemble the comparison table.
3. Pull GraphRARE's reported numbers; confirm dataset coverage.
4. Begin writing: intro + related work first.
