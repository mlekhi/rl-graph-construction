# project plan -- rl graph construction
_updated: 2026-05-12. target submission: aug 15, 2026._

---

## milestones

### week 1-2 (may 1-12) -- DONE
- [x] env setup, baseline reproduced on sentiment
- [x] 4-way split refactor (sage_val / policy_val)
- [x] graphrare paper read + notes
- [x] dataset pivot to cora / pubmed
- [x] graphsage + mlp baselines on cora and pubmed (gpu2)
- [x] RL_DESIGN.md written and updated with fadi's revisions

### week 3 (may 13-19) -- GraphEnv
- [ ] `GraphEnv` class: wraps frozen graphsage, manages graph state, applies edge edits
- [ ] edge candidate pool: existing citation edges + knn-k feature pool
- [ ] reward function: delta macro-f1 on policy_val, computed every 5 steps with EMA
- [ ] sanity check: random policy on cora should get ~0 reward

### week 4 (may 20-26) -- PolicyNet + PPO
- [ ] `PolicyNet`: MLP over edge pair features (add/remove probabilities)
- [ ] PPO training loop (cleanrl-based)
- [ ] smoke test: 100 episodes on cora, confirm reward is moving
- [ ] wandb wired in: reward curves, val f1, action histograms

### week 5 (may 27 - jun 2) -- first real results
- [ ] full cora run: 500 episodes PPO
- [ ] compare to graphsage baseline with CI
- [ ] debug if reward not improving (check action entropy, reward scale)
- [ ] checkpoint: does RL beat graphsage baseline on cora?

### week 6 (jun 3-9) -- ablations
- [ ] episode length sweep: 20 / 50 / 100 edits per episode
- [ ] reward frequency sweep: every 1 / 5 / 10 steps
- [ ] action ablation: add-only vs remove-only vs both
- [ ] results table for ablation section of thesis

### week 7 (jun 10-16) -- pubmed + citeseer
- [ ] repeat full RL pipeline on pubmed
- [ ] citeseer run if time permits (breadth experiment)
- [ ] compare across all datasets

### week 8 (jun 17-23) -- final eval
- [ ] 10-split CI on cora + pubmed
- [ ] comparison table vs graphsage, MLP, graphrare
- [ ] PPG if PPO plateaued (switch and re-run)

### weeks 9-10 (jun 24 - jul 7) -- buffer + wrap-up
- buffer for re-runs, unexpected issues, fadi feedback
- all experiments must be done by jul 7 -- no new experiments after this
- freeze results, finalize figures and tables

### weeks 11-12 (jul 8-21) -- writing block 1 (minimum 2 weeks pure writing)
- [ ] related work: graphrare comparison, other graph RL methods
- [ ] methodology chapter: RL formulation, GraphEnv, PolicyNet, training setup
- [ ] results chapter: baseline table, RL results, ablations
- no new experiments during this block

### weeks 13-14 (jul 22 - aug 5) -- writing block 2 + fadi review
- [ ] intro + conclusion
- [ ] figures and tables finalized
- [ ] full draft to fadi by jul 28 for review pass

### aug 6-15 -- revisions + submission
- [ ] revisions from fadi feedback
- [ ] submission

---

## targets to beat (cora)

| target | value | status |
|---|---|---|
| MLP baseline | 0.757 ± 0.012 | done |
| graphsage baseline | 0.879 ± 0.013 | done |
| graphsage-RARE (graphrare paper) | ~0.890 | stretch goal |

## targets to beat (pubmed)

| target | value | status |
|---|---|---|
| MLP baseline | 0.876 ± 0.003 | done |
| graphsage baseline | 0.886 ± 0.003 | done -- note: 1pp gap, may be hard to improve |

---

## risks

| risk | mitigation |
|---|---|
| RL doesn't beat graphsage on cora | check action entropy, reward scale, episode length |
| pubmed gap too small for contribution | focus thesis on cora, use pubmed as secondary |
| PPO doesn't converge | switch to PPG, reduce episode length, increase reward frequency |
| gpu2 unavailable | fall back to mac (cora ok on MPS, pubmed slow but doable) |
