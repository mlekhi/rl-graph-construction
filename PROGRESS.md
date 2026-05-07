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
- changes required:
  - `01_preprocessing.ipynb`: add `POLICY_VAL_SIZE` knob, produce `X_policy_val.npy` + `y_policy_val.npy`
  - `02_graphsage.ipynb`: add `policy_val_mask`, extend graph builder for policy_val nodes, re-tune SAGE under new split
- **note:** new baseline numbers will differ slightly from fadi's 0.6823 due to smaller sage_val (10% vs 20%) — must re-report before writing RL code

---

## 2026-05-06

### environment setup
- notebook was pinned to fadi's kernel `sentiment-rl-graph` (didn't exist locally)
- registered new kernel: `python3.11 -m ipykernel install --user --name sentiment-rl-graph`
- installed missing deps on python3.11: scikit-learn, imbalanced-learn, torch-geometric, optuna, seaborn, matplotlib

### 01_preprocessing.ipynb — 4-way split complete
- edited notebook to produce 4-way split: train / sage_val / policy_val / test
- `POLICY_VAL_SIZE = 0.5` (splits existing val 50/50 into sage_val + policy_val)
- new split sizes: train=24,087 (balanced) | sage_val=2,481 | policy_val=2,482 | test=6,203
- new files: `preprocessed/X_sage_val.npy`, `preprocessed/X_policy_val.npy`, `preprocessed/y_sage_val.npy`, `preprocessed/y_policy_val.npy`
- new data splits: `data_splits/sage_val.csv`, `data_splits/policy_val.csv`
- set `REGENERATE_SPLITS = False` after first run

### 02_graphsage.ipynb — re-run under new split (in progress)
- notebook uses cached best params from fadi's run (optuna re-tune skipped for now — acceptable for baseline replication)
- **new baseline numbers (4-way split, 10 seeds):**
  - sage_val accuracy: **0.6995 ± 0.0014**
  - test accuracy: **0.6819 ± 0.0025**
  - test acc drop from 0.6823 → 0.6819 is within CI — baseline is stable
- k-sweep running (K=5,10,15,20,30,50,100) — results pending
- note: SAGE params not re-tuned on new split yet — should do a proper optuna run before finalizing RL baseline

---

## 2026-05-07

### 02_graphsage k-sweep — completed
- k-sweep finished for all 7 K values (5, 10, 15, 20, 30, 50, 100) under the new 4-way split
- results pending commit once script writes final CSVs
- run was slow due to mac lid-close pausing the process (caffeinate not wrapping the python script run)
- lesson learned: always wrap long runs with `caffeinate -s` from the start

### wandb setup
- installed wandb 0.26.1 on python3.11
- logged in to account: mlekhi-western-university
- project name: `rl-graph-construction`
- not yet wired into training loop — to do next session

### RL design doc (`RL_DESIGN.md`)
- wrote policy sketch covering all open design questions from fadi's plan
- key decisions made (with assumptions noted):

| decision | choice | assumption |
|---|---|---|
| episode structure | bandit (1 step, all edges at once) | sequential per-edge would be intractable |
| construction frequency | once per episode (full graph) | per-sample too slow on 35k nodes |
| candidate pool | kNN-50 | inherits from k-sweep peak |
| state | node features + neighbor mean + sage logits + degree | sage confidence is informative signal |
| action | binary keep/prune per candidate, factorized | independent per-node decisions are reasonable |
| reward | sparse delta macro-F1 on policy_val | dense reward too noisy on 2.5k val nodes |
| policy net | 2-layer graphsage + MLP head | could be overkill — MLP on node pairs may suffice |
| algorithm | REINFORCE → PPO if unstable | standard starting point |

- biggest open assumption: policy net architecture (graphsage vs simpler MLP)
- **fadi should review design doc before any code is written**

### week 1 retrospective
completed:
- [x] env setup + kernel registration
- [x] baseline reproduced (within ±0.005 of 0.6823)
- [x] 4-way split refactor (ahead of schedule — was week 2 in fadi's plan)
- [x] new baseline numbers under 4-way split
- [x] k-sweep under new split
- [x] policy sketch (RL_DESIGN.md)
- [x] wandb installed + logged in

not done from fadi's checklist:
- [ ] read must-read papers (graphsage, lds-gnn, idgl, neuralsparse)
- [ ] wandb wired into training loop
- [ ] toy PPO smoke test on small graph
