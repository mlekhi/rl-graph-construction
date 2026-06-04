# GraphHARE — Full End-to-End Flowchart

## 1. Setup (run once per dataset)

```
Planetoid Dataset (Cora / PubMed / CiteSeer)
        │
        ├── 01_baseline_planetoid.py
        │       runs GraphSAGE + MLP baselines (10 splits, optuna tuning)
        │       saves: best_params.json, baseline_summary.csv
        │       result: GraphSAGE macro-F1, MLP macro-F1, gap between them
        │
        ├── 03_homophily_diagnostic.py
        │       computes edge homophily of original graph + kNN pool
        │       gate: if kNN pool < 1.5x random baseline → drop dataset
        │       result: all 3 datasets pass (cora 3.6x, citeseer 3.2x, pubmed 2.1x)
        │
        └── 02_freeze_sage.py
                trains GraphSAGE with best params on fixed 4-way split
                freezes weights (never updated again)
                saves: frozen_sage_seed42.pt, masks_seed42.pt
```

---

## 2. RL Training (GraphHARE)

```
frozen_sage_seed42.pt + original graph
        │
        ▼
┌──────────────────────────────────────────────┐
│  GraphEnv (graph_env.py)                     │
│                                              │
│  state per node:                             │
│  [sage_embedding || sage_logits ||           │
│   degree || homophily || entropy]            │
│                                              │
│  candidate edges per step:                   │
│  kNN pool (add) + existing edges (remove)    │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  EPISODE LOOP (500 episodes x 100 steps)     │
│                                              │
│  for each step:                              │
│  1. sample node weighted by entropy          │
│  2. get ~25 candidate edges for that node    │
│  3. PolicyNet (policy_net.py) scores them    │
│     input: [s_i || s_j || cos_sim ||        │
│             in_orig || op_is_add]  (151-dim) │
│     output: logits -> categorical sample     │
│  4. apply chosen edge edit to graph          │
│                                              │
│  every 5 steps:                              │
│  5. frozen SAGE forward pass on new graph    │
│  6. compute reward:                          │
│                                              │
│  FIXED beta:                                 │
│  r = (dF1/s_F1) + beta x (dhom/s_hom)       │
│                                              │
│  LEARNABLE beta (week 8):                    │
│  beta = exp(log_beta)  <- learned by network │
│  r = (dF1/s_F1) + beta x (dhom/s_hom)       │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  PPO UPDATE (after each episode)             │
│                                              │
│  - compute discounted returns                │
│  - normalize advantages                      │
│  - clip ratio update (eps=0.2)               │
│  - entropy bonus (0.05)                      │
│  - update policy weights                     │
│  - (learnable beta) update log_beta too      │
└──────────────────────────────────────────────┘
        │
        ▼
best_policy.pt saved when macro-F1 improves
```

---

## 3. Experiment Tracking (every run)

```
04_train_rl.py produces:
        │
        ├── results.json
        │       full config (dataset, beta, all hyperparams, seed)
        │       metrics (best_macro_f1, delta, h_original, h_knn, h_refined)
        │       git commit hash + wall time
        │
        ├── wandb run
        │       live curves: reward, macro_f1, homophily, grad_norm, action ratio
        │       project: graphhare @ wandb.ai/mlekhi-western-university/graphhare
        │
        └── best_policy.pt
                saved whenever macro-F1 improves during training
```

---

## 4. Analysis

```
05_compile_results.py
        reads all results.json files
        outputs runs/results_summary.csv  <- single source of truth for paper tables
        │
        ▼
06_plot_beta_curve.py
        plots beta vs best macro-F1 per dataset
        shows inverted-U shape with peak at optimal beta
        outputs runs/beta_curve_{dataset}.png
```

---

## 5. Results So Far

```
Cora beta ablation (seed=42, normalized reward):

beta: 0    0.25  0.5   0.75  1.0   1.1   1.2   1.3   1.5   2.0
F1: .879  .880  .878  .879  .882  .882  .882  .880  .878  .878
                              peak region: beta=1.0-1.2

All refined graphs:
  h_original = 0.810
  h_knn_pool = 0.517
  h_refined  = 0.813-0.815  <- agent builds more homophilic graph

GraphHARE (beta=1.1) macro-F1 = 0.8820  (+0.0157 over GraphSAGE)
GraphSAGE baseline             = 0.8663
MLP baseline                   = 0.7320
```

---

## 6. What Comes Next

```
Currently running on gpu2:
  full beta ablation on pubmed + citeseer

Week 8:
  implement learnable beta
  beta self-tunes during training instead of manual sweep
  expected: converges near 1.1 on cora, lower on pubmed

Week 8:
  10-seed CI on all 3 datasets at best beta
  comparison table vs GraphSAGE, MLP, GraphRARE (beta=0)

Writing:
  math formulation (with fadi)
  methodology, results, ablation section
  target: NeurIPS
```
