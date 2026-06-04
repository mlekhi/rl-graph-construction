# project plan -- GraphHARE
_updated: 2026-05-28._
_conference submission: IEEE BigData or IJCNN (venue locked ~week 9 based on results)._
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

### week 5 (may 27 - jun 2) -- IN PROGRESS: homophily pivot
- [ ] **homophily diagnostic**: compute edge homophily of starting graph + kNN pool for cora, pubmed, citeseer. mandatory gate before more training.
- [ ] **update GraphEnv reward**: `gamma * delta_minority_f1` → `beta * delta_homophily`
- [ ] **citeseer baseline**: graphsage + mlp baselines on citeseer (10 splits)
- [ ] **β=0 run**: replicate accuracy-only baseline on cora (clean GraphRARE comparison point)
- [ ] reply to fadi with progress update + reschedule zoom

### week 6 (jun 3-9) -- β ablation (empirical core)
- [ ] train cora at β ∈ {0, 0.1, 0.5, 1.0, 2.0}
- [ ] log edge homophily of final graph at each β (alongside accuracy)
- [ ] confirm β=0 vs β>0 produce distinguishably different graphs
- [ ] wandb wired in: reward curves, val f1, graph homophily, action histograms
- [ ] checkpoint: does β>0 beat β=0 in F1 or produce higher-homophily output?

### week 7 (jun 10-16) -- pubmed + citeseer
- [ ] repeat full pipeline on pubmed (β ablation)
- [ ] citeseer run (β ablation)
- [ ] compare β trade-off curve across datasets

### week 8 (jun 17-23) -- final eval + learnable β
- [ ] 10-split CI on cora, pubmed, citeseer
- [ ] comparison table vs graphsage baseline, mlp, graphrare (β=0 row)
- [ ] learnable β: make β a trainable parameter (log_beta = nn.Parameter) updated via meta-optimizer on val performance -- if implemented, compare vs fixed β=1.0
- [ ] PPG if PPO plateaued
- [ ] episode length sweep: 20 / 50 / 100 if time allows

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
| GraphHARE β=0 beats GraphSAGE | >0.879 | in progress |
| GraphHARE β>0 beats β=0 | TBD | pending ablation |

## targets (pubmed)

| target | value | status |
|---|---|---|
| MLP baseline | 0.876 ± 0.003 | done |
| GraphSAGE baseline | 0.886 ± 0.003 | done -- 1pp gap, secondary dataset |

## targets (citeseer)

| target | value | status |
|---|---|---|
| MLP + GraphSAGE baselines | TBD | not run yet |

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
