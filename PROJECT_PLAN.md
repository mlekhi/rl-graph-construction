# project plan -- GraphHARE
_updated: 2026-05-28._
_target venue: NeurIPS (with learnable β + multi-dataset ablation + math formulation) or IEEE BigData/IJCNN as fallback._
_thesis submission: aug 15, 2026._

---

## milestones

### week 1-2 (may 1-12) -- DONE
- [x] env setup, baseline reproduced on sentiment
- [x] 4-way split refactor (sage_val / policy_val)
- [x] graphrare paper read + notes
- [x] dataset pivot to cora / pubmed
- [x] graphsage + mlp baselines on cora and pubmed (gpu2)
- [x] RL_DESIGN.md written and updated with fadi's revisions

### week 3-4 (may 13-26) -- DONE
- [x] `GraphEnv` class: wraps frozen graphsage, manages graph state, applies edge edits
- [x] edge candidate pool: existing citation edges + knn-10 feature pool
- [x] `PolicyNet`: MLP over edge pair states (151-dim, 72k params)
- [x] PPO training loop: node-first action selection, clean categorical log_prob ratio
- [x] smoke test passing on gpu2 (cora, 20 episodes)
- [x] full 500-episode PPO run on cora: baseline 0.8663 → best 0.8816 (+0.0153)

### week 5 (may 27 - jun 2) -- DONE: homophily pivot
- [x] homophily diagnostic: all 3 datasets pass (cora 3.6x, pubmed 2.1x, citeseer 3.2x above random)
- [x] update GraphEnv reward: `gamma * delta_minority_f1` → `beta * delta_homophily`
- [x] citeseer baselines: graphsage 0.761 ± 0.009, mlp 0.728 ± 0.008
- [x] citeseer frozen sage checkpoint (val_acc=0.779)
- [x] reply to fadi with progress update

### week 6 (jun 3-9) -- DONE: β ablation (empirical core)
- [x] reward normalization (Welford online algorithm) — β is now interpretable
- [x] full β ablation on cora: {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} — β=1.0 peaks at 0.8816
- [x] log all 3 homophily values per run (original, kNN pool, refined)
- [x] wandb wired in: reward curves, macro_f1, homophily, action histograms, grad norm
- [x] β=1.0 beats β=0 (+0.0153 vs +0.0131) — inverted-U confirmed (single seed)
- [x] experiment tracking system: config files, results_summary.csv, git commit + wall time per run
- [x] β ablation curve plot generated

### week 7 (jun 10-16) -- fine-tune β + multi-dataset ablation
**thursday meeting agenda:**
- [ ] discuss NeurIPS feasibility with fadi — which theoretical claims does he want to make?
- [ ] ask fadi to lead/co-write the mathematical formulation section
- [x] confirm learnable β is in scope for the paper ← CONFIRMED by fadi jun 5
- [x] run β = {1.1, 1.2, 1.3} on cora — peak at β=1.1 (0.8820, +0.0157)
- [x] read NoisyNet paper (noisy networks for exploration)
- [x] freeze sage for pubmed (val_acc=0.902, test_macro_f1=0.887)
- [x] full β ablation on pubmed: flat curve, +0.002, graph redundant as expected
- [x] full β ablation on citeseer: flat F1, +0.011, β doesn't differentiate
- [x] plot β curve for all 3 datasets side by side (runs/beta_curve_all_datasets.png)
- [x] fadi's theory confirmed: cora (7 classes) clear peak, citeseer (6) flat, pubmed (3) minimal gain
- [x] repo cleanup: scripts numbered 01-06, archive/ organized

### week 8 (jun 17-23) -- learnable β (NeurIPS upgrade)
- [ ] implement learnable β: make β a network parameter tuned through training (NoisyNet-inspired)
- [ ] compare learnable β vs fixed β=1.0 on cora
- [ ] mathematical formulation of homophily-aware reward with learnable β
- [ ] 10-seed CI on cora, pubmed, citeseer at best β
- [ ] comparison table vs graphsage baseline, mlp, graphrare (β=0 row)
- [ ] PPG if PPO plateaued

### weeks 9-10 (jun 24 - jul 7) -- buffer + wrap-up
- lock conference venue based on results
- all experiments must be done by jul 7 -- no new experiments after this
- freeze results, finalize figures and tables

### weeks 11-12 (jul 8-21) -- writing block 1
- [ ] related work: graphrare comparison, homophily in gnns, rl for graph structure
- [ ] methodology: GraphHARE formulation, GraphEnv, PolicyNet, homophily reward
- [ ] results: baseline table, β ablation curve, homophily of output graphs
- no new experiments during this block

### week 13 (jul 22-28) -- writing block 2 + fadi review
- [ ] intro + conclusion
- [ ] figures and tables finalized
- [ ] **full draft to fadi by jul 25**

### jul 29 - aug 8 -- revisions
- [ ] revisions from fadi feedback
- [ ] final proofread
- [ ] conference submission (venue TBD)

### aug 6-15 -- thesis submission
- [ ] thesis submission

---

## targets (cora)

| target | value | status |
|---|---|---|
| MLP baseline | 0.757 ± 0.012 | done |
| GraphSAGE (default graph) | 0.879 ± 0.013 | done |
| GraphHARE β=0 beats GraphSAGE | >0.879 | ✓ 0.8794 (single seed) |
| GraphHARE β>0 beats β=0 | TBD | ✓ β=1.0: 0.8816 vs 0.8794 (single seed, needs 10-seed CI) |

## targets (pubmed)

| target | value | status |
|---|---|---|
| MLP baseline | 0.876 ± 0.003 | done |
| GraphSAGE baseline | 0.886 ± 0.003 | done -- 1pp gap, secondary dataset |

## targets (citeseer)

| target | value | status |
|---|---|---|
| MLP baseline | 0.728 ± 0.008 | done |
| GraphSAGE baseline | 0.761 ± 0.009 | done |
| GraphHARE β ablation | TBD | pending (week 7) |

---

## risks

| risk | mitigation |
|---|---|
| β=0 and β>0 produce indistinguishable results | adjust β range; reframe contribution as "robustness of accuracy-only reward" |
| homophily diagnostic fails for a dataset | drop that dataset before investing gpu time |
| RL doesn't beat graphsage on cora | check action entropy, reward scale, episode length sweep |
| pubmed gap too small | treat as secondary; cora is primary contribution |
| PPO plateaus | switch to PPG (cobbe et al. 2021) |
| gpu2 unavailable | fall back to mac (cora ok on MPS) |
