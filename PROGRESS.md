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

---

## 2026-05-08

### graphrare paper read (2312.09708v2)

fadi identified this as closest competing work. full paper read (14 pages).

**what they do:**
- target: heterophilic graphs where same-label nodes are multi-hop away
- node relative entropy = feature entropy + structural entropy, weights structural by lambda (default 1.0)
- DRL agent: per node, selects k nodes to add and d nodes to remove from ranked candidate set
- co-trains GNN + DRL jointly (not frozen backbone)
- PPO with MLP policy (not a GNN policy)
- reward: delta accuracy + lambda * delta loss on training set (dense, every episode)
- tested on 7 datasets: 5 heterophilic + cora/pubmed

**key results:**
- graphsage-RARE beats vanilla graphsage by 7.81% average accuracy across 7 datasets
- gains largest on heterophilic datasets, smaller on cora/pubmed (homophilic)
- convergence: DRL reward noisy for first ~15 episodes then stabilizes (fig 6c)
- runtime: ~3x slower than baseline, but entropy computed once before training

**ablation findings:**
- both adding AND removing edges matter -- doing only one is suboptimal
- relative entropy prior is critical -- random shuffling of candidates hurts significantly
- reward function matters (AUC vs accuracy produces different results)
- DRL module clearly helps vs fixed-k baseline

**how we differ:**
- frozen SAGE vs co-training: avoids reward hacking, but GNN can't adapt to optimized graph
- moderate homophily (~0.51) vs purely heterophilic target -- we're closer to their cora regime
- tweet/NLP domain vs citation/social networks
- no relative entropy prior (pure RL exploration) -- could add this as a feature later
- our reward: delta macro-F1 on policy_val (held-out split) vs their training-set reward
- policy architecture TBD (fadi wants to revisit after reading this paper)

**implications for RL_DESIGN.md:**
- fadi's push for dense reward is validated by graphrare results
- per-node add+remove (not just prune) should be in our action space
- MLP policy may be sufficient -- graphrare didn't need a GNN policy
- 50-100 edits per episode (fadi's revision) aligns with graphrare's sequential approach

---

## 2026-05-12

### dataset pivot: cora + pubmed (fadi's instruction)

fadi confirmed switch away from the sentiment dataset. reason: MLP is already competitive with graphsage on sentiment (MLP F1 ~0.689 vs graphsage ~0.6856), leaving no headroom to demonstrate RL graph construction adds value. on cora/pubmed the GNN-vs-MLP gap is 5-8pp, making the contribution demonstrable on well-cited benchmarks.

**new primary datasets:**
- cora (primary)
- pubmed (primary)
- citeseer (possible third, for breadth)

**what's deprecated:**
- sentiment pipeline (bert embeddings, smote, pca, custom knn graph)
- verify_k100.py (no longer needed)
- optuna re-tune on sentiment split
- 01_preprocessing.ipynb / 02_graphsage.ipynb as primary notebooks

**what carries over:**
- multi-seed CI methodology (exactly same)
- sage_val / policy_val split pattern (apply to planetoid splits)
- wandb setup (already installed/logged in)
- graphsage training code (adapt for planetoid)
- RL design (dataset-agnostic, still valid)

**new approach:**
- load via PyG Planetoid loader
- follow graphrare split: 60/20/20 per class, 10 random splits
- add policy_val by splitting val 50/50 into sage_val + policy_val
- establish graphsage + MLP baselines on cora and pubmed
- use original citation graph edges (not knn from features) as RL candidate pool

**completed today:**
- [x] wrote `03_baseline_planetoid.py` -- graphsage + MLP baselines, graphrare-style splits
- [x] set up gpu2.gaul.csd.uwo.ca (4x GTX 1060 6GB, miniconda, pytorch+pyg)
- [x] ran cora baseline (10 splits, optuna 50 trials)
- [x] ran pubmed baseline (10 splits, optuna 50 trials)
- [x] updated RL_DESIGN.md for new dataset context + fadi's revisions

### baseline results (cora + pubmed, 10 splits, 60/10/10/20 per-class split)

| dataset | model | test acc | test f1 |
|---|---|---|---|
| cora | graphsage | 0.879 ± 0.013 | 0.866 ± 0.015 |
| cora | MLP | 0.757 ± 0.012 | 0.732 ± 0.016 |
| pubmed | graphsage | 0.886 ± 0.003 | 0.884 ± 0.003 |
| pubmed | MLP | 0.876 ± 0.003 | 0.874 ± 0.004 |

**cora gap: ~12pp** -- strong signal that graph structure matters, RL has clear headroom.
**pubmed gap: ~1pp** -- small gap, consistent with graphrare's findings on pubmed. may be less compelling for the thesis contribution. worth flagging to fadi.

RL targets to beat:
- cora: graphsage 0.879, graphrare reported graphsage-RARE at ~0.890 (our stretch)
- pubmed: graphsage 0.886, gap to MLP is narrow so contribution story is harder here

**next steps:**
- [ ] scp results from gpu2 back to mac and commit
- [ ] write project plan with weekly milestones through aug 15
- [ ] reply to fadi with baseline results + updated design doc + project plan
- [ ] start GraphEnv class (RL environment wrapper)
