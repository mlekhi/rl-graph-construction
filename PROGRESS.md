# thesis progress log

## 2026-05-01

### repo setup
- initialized git repo at `/Users/mlekhi/THESIS`, connected to [mlekhi/rl-graph-construction](https://github.com/mlekhi/rl-graph-construction)
- added `.gitignore` excluding large binary artifacts: `cache/`, `preprocessed/*.npy`, `runs/*/model_outputs/`, `data/`
- committed: notebooks, handover doc, data splits, best params json, k-sweep metrics/plots, metadata

### baseline summary (inherited from fadi)
- pipeline: bert-base-uncased (frozen, mean pool) → pca(128) → z-score → smote → knn graph (K=15, cosine, inductive_directed) → graphsage
- test accuracy: **0.6823 ± 0.0022** (10 seeds)
- test macro-f1: **0.6856 ± 0.0021**
- best hyperparams: hidden_dim=256, num_layers=4, dropout=0.234, lr=1.71e-3, weight_decay=6.0e-5, aggr=mean
- k-sweep peak: K=50 → 0.6853 test acc (hard target for RL to beat)
- neutral class recall is weakest at ~60% — primary RL optimization target

---

## 2026-05-05

### plan: 4-way split refactor (prerequisite for RL)
- original split: 60% train / 20% val / 20% test
- new split: 60% train / 10% sage_val / 10% policy_val / 20% test
- rationale: if SAGE early-stops on val AND RL is rewarded on val, that's double-dipping (reward hacking). separate splits prevent this. test stays at 20% for comparability with fadi's baseline.

---

## 2026-05-06

### environment setup
- notebook was pinned to fadi's kernel `sentiment-rl-graph` (didn't exist locally)
- registered new kernel: `python3.11 -m ipykernel install --user --name sentiment-rl-graph`
- installed missing deps on python3.11: scikit-learn, imbalanced-learn, torch-geometric, optuna, seaborn, matplotlib

### 01_preprocessing.ipynb — 4-way split complete
- new split sizes: train=24,087 (balanced) | sage_val=2,481 | policy_val=2,482 | test=6,203

### RL design doc (RL_DESIGN.md)
- wrote policy sketch covering all open design questions

---

## 2026-05-07

### wandb setup
- installed wandb 0.26.1 on python3.11
- logged in to account: mlekhi-western-university

---

## 2026-05-08

### graphrare paper read (2312.09708v2)
- target: heterophilic graphs, co-trains GNN + DRL jointly
- reward: delta accuracy + lambda * delta loss on training set
- key finding: both adding AND removing edges matter, relative entropy prior critical
- our differentiators: frozen SAGE, held-out policy_val reward, homophilic datasets

---

## 2026-05-12

### dataset pivot: cora + pubmed (fadi's instruction)
- sentiment MLP already competitive with graphsage (~0.689 vs 0.686) — no headroom
- new primary: cora + pubmed + citeseer
- deprecated: sentiment pipeline, preprocessing notebooks as primary

### baseline results (cora + pubmed, 10 splits, 60/10/10/20 per-class)

| dataset | model | test acc | test f1 |
|---|---|---|---|
| cora | graphsage | 0.879 ± 0.013 | 0.866 ± 0.015 |
| cora | MLP | 0.757 ± 0.012 | 0.732 ± 0.016 |
| pubmed | graphsage | 0.886 ± 0.003 | 0.884 ± 0.003 |
| pubmed | MLP | 0.876 ± 0.003 | 0.874 ± 0.004 |

- cora gap: ~12pp — strong RL headroom
- pubmed gap: ~1pp — small, expected behavior per fadi (frame as "graph redundant" case)

---

## 2026-05-13 to 2026-05-19 (week 3)

### GraphEnv implemented (`src/graph_env.py`)
- wraps frozen graphsage, manages edge edits, knn-10 candidate pool
- 4-way split loaded from masks file
- dense reward every 5 steps with EMA smoothing
- node-first action selection: sample by structural entropy, evaluate ~25 candidates
- vectorized edge state computation: `get_edge_states_batch()` (151-dim per candidate)
- edge state: `[s_i || s_j || cosine_sim || in_original_graph || op_is_add]`
- node state: `[h_i || sage_logits_i || degree_i || homophily_i || entropy_i]`
- sanity check passed: random policy gets ~0 reward

### freeze_sage.py implemented
- trains graphsage with best optuna params on fixed split, saves checkpoint
- cora: val_acc=0.910, test_acc=0.895
- saves: `frozen_sage_seed42.pt`, `masks_seed42.pt` to `runs/rl_{dataset}/`

---

## 2026-05-20 to 2026-05-26 (week 4)

### PolicyNet implemented (`src/policy_net.py`)
- MLP over edge-pair states (151-dim input)
- shared trunk → actor head (scalar logit) + critic head (scalar value)
- 72,834 parameters, orthogonal init, LayerNorm + dropout
- AuxValueHead for PPG auxiliary phase (removed, kept PPO only)

### PPO training loop implemented (`src/train_rl.py`)
- clean categorical log_prob ratio (fixes old PPG correctness issues)
- node-first selection, value head receives full candidate tensor
- RolloutBuffer stores full candidate set per step for correct ratio computation
- output dir includes beta in name: `ppo_seed42_beta{beta}/`
- wandb logging: reward, macro_f1, homophily, action histograms, grad norm

### smoke test passing on gpu2
- cora, 20 episodes, β=0: baseline 0.8663 → best 0.8728
- cora, 20 episodes, β=0.5: baseline 0.8663 → best 0.8737

---

## 2026-05-27 to 2026-05-29 (week 5)

### fadi's revised RL_DESIGN.md received and merged
- method renamed: **GraphHARE** (Homophily-Aware Reward for Edges)
- single contribution: homophily-aware reward `r_t = delta_macro_f1 + beta * delta_homophily`
- drops minority class F1 bonus
- citeseer added as required dataset
- drops MIT URTC, targets IEEE BigData / IJCNN
- beta ablation is the empirical core

### homophily diagnostic (`src/homophily_diagnostic.py`)
- mandatory pre-training gate per fadi's design doc
- all 3 datasets pass:

| dataset | h_original | h_knn_pool | random_base | ratio |
|---|---|---|---|---|
| cora | 0.810 | 0.517 | 0.143 | 3.6x |
| pubmed | 0.802 | 0.713 | 0.333 | 2.1x |
| citeseer | 0.736 | 0.525 | 0.167 | 3.2x |

### GraphEnv reward updated to GraphHARE
- `gamma * delta_minority_f1` → `beta * delta_homophily`
- added `_eval_homophily()` using ground-truth labels (Zhu et al. 2020)
- `get_current_metrics()` returns (macro_f1, homophily)

### citeseer baselines (10 splits, 60/10/10/20 per-class)

| model | test acc | test f1 |
|---|---|---|
| graphsage | 0.761 ± 0.009 | 0.714 ± 0.015 |
| MLP | 0.728 ± 0.008 | 0.682 ± 0.009 |

### citeseer frozen sage
- val_acc=0.779, test_acc=0.754, test macro_f1=0.719

### pre-normalization beta ablation runs on cora (seed=42, 500ep, 100 steps)
- **note: reward terms not normalized — beta values not yet interpretable. directional only.**

| beta | best macro_f1 | delta | homophily trend |
|---|---|---|---|
| 0.0 | 0.8790 | +0.0127 | rises as byproduct (+0.003-0.005) |
| 0.1 | 0.8807 | +0.0145 | positive |
| 0.5 | 0.8831 | +0.0168 | consistently +0.003-0.006 |
| 1.0 | 0.8794 | +0.0131 | positive |

- β=0.5 beats β=0 on both accuracy and homophily — directional signal is good
- runs saved under `prenorm_` prefix

### citeseer pre-normalization run (seed=42, 500ep, 100 steps, β=0.5)
- baseline macro_f1=0.713 → best 0.722 (+0.009)
- saved under `prenorm_` prefix

---

## 2026-05-29 — fadi feedback (email)

### key technical changes required
1. **normalize reward terms** before beta sweep — delta_F1 and delta_homophily must be on comparable scale so beta means something. without this beta values are just raw magnitude ratios
2. **new beta sweep**: {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} — need to find peak and show it turn over on both sides
3. **homophily reporting**: all three reference points — original graph, kNN pool, refined graph. primary comparison: kNN pool vs refined graph
4. **experiment tracking system**: structured CSV/JSON per run with full config, metrics, git commit, wall time. single source of truth for paper tables

### fadi's dataset framing notes
- more classes = more room for GNN to contribute over MLP (cora 7 classes > pubmed 3 classes)
- pubmed likely can't beat MLP or only marginally — frame as "controlled case where graph is redundant," not a weak result
- cora and citeseer are the primary contribution datasets

### next steps
- [ ] implement reward normalization in GraphEnv
- [ ] build experiment tracking system (CSV + JSON per run)
- [ ] re-run full beta sweep {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} on cora with normalized reward
- [ ] run pubmed baselines + freeze sage for pubmed
- [ ] 10-seed CI runs at best beta on cora, citeseer, pubmed
- [ ] thursday 3pm meeting with fadi — bring cora beta curve

---

## 2026-06-03

### reward normalization implemented (GraphEnv)
- added Welford online algorithm to estimate running std of delta_F1 and delta_homophily
- reward now: `r_t = (delta_F1 / σ_F1) + β × (delta_homophily / σ_homophily)`
- 20-step warmup before normalization kicks in (raw values used during warmup)
- β=1 now genuinely means equal weighting — values are interpretable
- all pre-normalization runs renamed with `prenorm_` prefix, kept as directional reference

### wandb wired in and working
- installed wandb 0.27.0 on gpu2
- logged in as mlekhi-western-university
- project: graphhare at wandb.ai/mlekhi-western-university/graphhare
- run naming: `{dataset}-beta{beta}-seed{seed}`, tagged by dataset and beta
- logging: reward, macro_f1, homophily, delta_macro_f1, delta_homophily, valid_actions, n_edges, action add/remove ratio, loss/clip, loss/value, loss/entropy, grad_norm

### normalized beta sweep running on cora (seed=42, 500ep, 100 steps, entropy_coef=0.05)
- all 7 runs started with --wandb, visible live at wandb.ai/mlekhi-western-university/graphhare
- beta = {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} running in parallel tmux panes on gpu2
- early (~17 episodes): curves still in warmup, differentiation expected after episode 100

### normalized beta sweep completed — cora (seed=42, 500ep, 100 steps)

| β | best macro_f1 | delta | h_refined | Δ homophily |
|---|---|---|---|---|
| 0.00 | 0.8794 | +0.0131 | 0.8129 | +0.0030 |
| 0.25 | 0.8800 | +0.0137 | 0.8145 | +0.0046 |
| 0.50 | 0.8781 | +0.0118 | 0.8132 | +0.0032 |
| 0.75 | 0.8786 | +0.0123 | 0.8135 | +0.0036 |
| **1.00** | **0.8816** | **+0.0153** | **0.8140** | **+0.0041** |
| 1.50 | 0.8782 | +0.0119 | 0.8135 | +0.0035 |
| 2.00 | 0.8776 | +0.0113 | 0.8131 | +0.0031 |

- β=1.0 peaks — inverted-U confirmed (single seed)
- all refined graphs above original homophily (0.810) and well above kNN pool (0.517)
- beats graphsage baseline (0.866) at all β values

### experiment tracking system built
- compile_results.py — single source of truth CSV across all runs
- configs/ — yaml file per experiment for reproducible runs
- results.json per run: full config, 3 homophily values, git commit hash, wall time
- wandb: live curves at wandb.ai/mlekhi-western-university/graphhare
- β ablation curve plot generated (runs/beta_curve.png)

---

## 2026-06-05 — thursday meeting with fadi

### meeting outcomes
- results confirmed: β=1.0 peak, inverted-U curve shown live in wandb
- fadi confirmed: homophily-in-reward is novel, no prior work does this
- **venue upgrade: NeurIPS** — if learnable β + math formulation + multi-dataset ablation
- fadi's theory confirmed by baselines: more classes = bigger GNN advantage over MLP
  - cora (7 classes): 13pp gap
  - citeseer (6 classes): 3pp gap  
  - pubmed (3 classes): 1pp gap
- fadi suggested: run β = 1.1, 1.2, 1.3 to find true peak between 1.0 and 1.5
- fadi suggested: learnable β via NoisyNet-inspired mechanism (β as network weight)
- fadi: will discuss math formulation at next meeting

### completed after meeting
- pubmed frozen sage: val_acc=0.902, test_acc=0.887, test_macro_f1=0.887
- citeseer β runs (0, 0.5, 1.0): all give best_f1=0.7242 (+0.0112) — no F1 differentiation yet, homophily increases with β
- β = 1.1, 1.2, 1.3 on cora running
- pubmed β runs (0, 0.5, 1.0) running
- citeseer full sweep (0.25, 0.75, 1.5, 2.0) running

### cora fine-tuning results (β = 1.1, 1.2, 1.3)

| β | best macro_f1 | delta | h_refined |
|---|---|---|---|
| 1.1 | **0.8820** | **+0.0157** | 0.8123 |
| 1.2 | **0.8820** | **+0.0157** | 0.8131 |
| 1.3 | 0.8799 | +0.0136 | 0.8142 |

- true peak is β=1.1 (tied with 1.2 on single seed)
- β=1.1 beats β=1.0 by +0.0004 — argues for learnable β to find exact optimum
- β ablation curve regenerated: runs/beta_curve_cora_full.png

### repo cleanup
- scripts renamed by pipeline order: 01_baseline_planetoid.py through 06_plot_beta_curve.py
- RESEARCH_HANDOVER.md moved to archive/
- configs/ yaml files for all cora β values

### currently running on gpu2
- citeseer β: {0.25, 0.75, 1.5, 2.0}
- pubmed β: {0, 0.5, 1.0} + {0.25, 0.75, 1.5, 2.0} queued

### next steps
- [ ] scp citeseer + pubmed results when done
- [ ] plot β curves for citeseer and pubmed
- [ ] read NoisyNet paper (arxiv 1706.10295)
- [ ] implement learnable β
- [ ] 10-seed CI at best β on all 3 datasets
- [ ] thursday meeting: NeurIPS math formulation discussion with fadi
