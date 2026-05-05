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
