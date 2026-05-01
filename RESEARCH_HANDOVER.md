# Research handover — GNN-based sentiment classification + RL graph construction

This document is for the student picking up this project. It explains what's already been done, what the research is trying to achieve, and where to start.

---

## 1. The research goal

**Long-term goal:** train a transferable RL policy for **graph construction** on node-classification tasks. The agent learns to add, remove, or weight edges based on node features and classifier confidence. Unlike dataset-specific learned adjacency methods (LDS-GNN, IDGL), the policy should generalize across datasets — that's the novelty claim.

**Short-term goal (what's already done):** build a strong GraphSAGE baseline on a tweet sentiment dataset that the RL agent will eventually try to beat by improving the graph structure.

**Why this dataset:** it's a realistic, imbalanced, text-based node-classification task with semantic similarity edges. Different from the classical Cora/Citeseer benchmarks, so improvements transfer to real applications more credibly.

---

## 2. Repository at a glance

```text
Sentiment-RL-Graph-Construction/
|-- data/                            <- raw Kaggle CSVs (input — not committed)
|   |-- train.csv
|   `-- test.csv
|
|-- data_splits/                     <- cached stratified splits
|   |-- train.csv
|   |-- val.csv
|   `-- test.csv
|
|-- cache/                           <- expensive intermediate artifacts
|   |-- emb_*.npy                       BERT embeddings per split
|   |-- pca_*.joblib                    fitted PCA estimator
|   |-- feat_*.npy                      post-PCA, post-scaling features
|   `-- edge_*.pt                       cached KNN edge_index per (K, mode, ...)
|
|-- preprocessed/                    <- HANDOVER OUTPUT from notebook 01
|   |-- X_train_bal.npy, y_train_bal.npy
|   |-- X_val.npy,        y_val.npy
|   |-- X_test.npy,       y_test.npy
|   `-- metadata.json
|
|-- runs/GETS_GNN_ONLY/              <- experiment artifacts
|   |-- best_params/                    cached Optuna best params per model
|   |-- metrics/                        result CSVs (incl. k_sweep.csv)
|   |-- model_outputs/                  per-seed model weights
|   `-- plots/                          training curves, confusion matrix, ROC, k-sweep
|
|-- 01_preprocessing.ipynb           <- run first
|-- 02_graphsage.ipynb               <- run second; contains the K-sweep
|-- gnn_only.py                      <- importable script version (SAGE-only)
|-- GETS_Framework.ipynb             <- original multi-model notebook (kept for reference)
`-- RESEARCH_HANDOVER.md             <- this document
```

---

## 3. Quick start

### Install dependencies

```bash
pip install -U numpy pandas scikit-learn scipy seaborn matplotlib optuna \
                imbalanced-learn transformers tqdm joblib

# PyTorch + PyG (pin to versions matching your CUDA setup)
pip install torch
pip install torch-geometric
```

### First run

```bash
# 1) Place Kaggle CSVs at data/train.csv and data/test.csv
# 2) Run the notebooks in order:
jupyter lab 01_preprocessing.ipynb     # ~10-15 min on first run (BERT embedding pass)
jupyter lab 02_graphsage.ipynb         # ~3 hours on M3 Max (Optuna 50 trials + 10-seed final + K-sweep)
```

Or, end-to-end in one shot:

```bash
python gnn_only.py
```

**Wall-clock breakdown for `02_graphsage.ipynb` on M3 Max (MPS):**

- Imports + load preprocessed: <1 min
- KNN graph build (K=15 cached): instant after first run
- Optuna 50 trials: ~1.5–2 hours
- Multi-seed (10 seeds × 300 epochs max with patience 30): ~30 min
- K-sweep (7 K values × 5 seeds): ~1–1.5 hours
- Plotting + CSV writing: <1 min

### Subsequent runs

Both notebooks aggressively cache. After the first run, re-running takes seconds for steps that haven't changed inputs. To force re-computation, delete the relevant cache files (or set `REGENERATE_SPLITS=True`, `USE_CACHED_EMBEDDINGS=False`, etc.).

---

## 4. The pipeline, explained

The two notebooks together implement a 7-stage pipeline. **Each stage is justified — don't change a stage without understanding why it exists.**

### Stage 1: Stratified splits (notebook 01, section 3)

We re-split the Kaggle data ourselves rather than using their train/test as-is, because we need:

1. A consistent val set for early stopping (Kaggle only gives train + test).
2. Stratification so each split has the same class distribution (the data is imbalanced).
3. Reproducibility — same seed, same rows.

> **Knob to think about:** `TEST_SIZE=0.20`, `VAL_SIZE=0.20`. Standard 60/20/20 split. Try 70/15/15 if you want more training data.

### Stage 2: Label encoding (notebook 01, section 4)

Map `{negative, neutral, positive}` to `{0, 1, 2}`. The fixed list ensures stable indices across runs (matters for confusion matrices, ROC).

### Stage 3: BERT embeddings (notebook 01, section 5)

Pass each tweet through frozen `bert-base-uncased`. **Mean pool** over real (non-pad) tokens — generally better than `[CLS]` for similarity-based downstream tasks when the encoder isn't fine-tuned.

> **Knob to think about:** `HF_MODEL_NAME`. Try `roberta-base` or a domain-specific model like `cardiffnlp/twitter-roberta-base-sentiment` to see if better embeddings translate to better SAGE results.

### Stage 4: PCA (notebook 01, section 6)

Compress 768-dim BERT vectors to `PCA_DIM=128`. Faster KNN, less noise, smaller GNN inputs.

> **Critical:** PCA is fit on **TRAIN ONLY**. Fitting on the union would let test information leak into the projection.

### Stage 5: Feature scaling (notebook 01, section 7)

Z-score standardize so each feature has mean 0 / std 1. Same train-only-fit principle.

### Stage 6: Class balancing (notebook 01, section 8)

SMOTE on the **TRAIN** split only (val/test keep their natural distribution). SMOTE synthesizes minority-class samples by interpolating between real ones in feature space.

> **Knob to think about:** `BALANCE_METHOD`. We default to SMOTE; try `none`, `smote_tomek`, `tomek` and see which the RL agent prefers (the agent might do better on un-balanced data because synthetic SMOTE samples create artificial edges).

### Stage 7: KNN graph induction (notebook 02, section 4)

Build a graph where each node connects to its `KNN_K` nearest feature-space neighbors (cosine distance). **Inductive directed mode**: train forms an undirected internal graph; val/test attach via directed edges train -> (val/test). This mirrors a deployment scenario.

> **Knob to think about:** `KNN_K` is the central knob the K-sensitivity sweep characterizes. **This is what the RL agent eventually replaces.**

### Stage 8: GraphSAGE training (notebook 02, section 7)

Two-phase training:

1. Optuna (50 trials) finds the best hyperparameters.
2. Multi-seed (10 seeds) final training for honest mean ± 95% CI.

The `train_gnn` function uses **eval-mode metrics** — after each optimizer step, it re-runs the forward pass with dropout off before measuring train/val accuracy. This avoids dropout noise contaminating the early-stopping signal.

---

## 5. Current results (and what they imply)

### The final baseline numbers (your starting point)

After multiple rounds of tuning (search space widened, K re-tuned), this is the frozen baseline:

| Metric | Value (mean ± 95% CI over 10 seeds) |
|---|---|
| **Test accuracy** | **0.6823 ± 0.0022** |
| Test macro-F1 | 0.6856 ± 0.0021 |
| Val accuracy | 0.6981 ± 0.0022 |
| Train accuracy | 0.797 ± 0.012 |
| ROC micro-AUC | 0.866 |
| Train–val gap | +9.9 pp (acceptable; not overfit) |
| Val–test gap | +1.6 pp (consistent across seeds) |

**Best hyperparameters** (saved at `runs/GETS_GNN_ONLY/best_params/GraphSAGE_best_params.json`):

- `hidden_dim = 256`, `num_layers = 4`, `dropout = 0.234`
- `lr = 1.71e-3`, `weight_decay = 6.0e-5`, `aggr = "mean"`
- KNN graph: `K = 15`, cosine, inductive_directed

### K-sensitivity sweep — already run, results saved

The sweep across `K ∈ {5, 10, 15, 20, 30, 50, 100}` shows graph structure clearly matters:

| K | Test accuracy | Comment |
|---|---|---|
| 5 | 0.6742 ± 0.0038 | sparsest — too few neighbors |
| 10 | 0.6788 ± 0.0034 | |
| **15** | **0.6829 ± 0.0029** | published default |
| 20 | 0.6835 ± 0.0027 | |
| 30 | 0.6842 ± 0.0042 | |
| **50** | **0.6853 ± 0.0028** | empirical peak |
| 100 | 0.6817 ± 0.0038 | over-smoothed |

The ~1.1 pp gap from K=5 to K=50 (CIs do not overlap) is the **go-signal for the RL project**: graph structure has measurable impact. Your RL has signal to chase.

The full table is at `runs/GETS_GNN_ONLY/metrics/k_sweep.csv`. The plot is at `runs/GETS_GNN_ONLY/plots/k_sweep.png`.

### Why GraphSAGE (and not GCN/GAT/GCNII/Transformer)

We initially ran the same pipeline with five GNN architectures: GCN, GAT, GraphSAGE, GCNII, GraphTransformer. **GraphSAGE was the strongest; GAT was the weakest.** This tells us about the *data*, not just the models:

1. The KNN graph is **highly homophilous** (most edges connect same-sentiment tweets). Simple aggregation (SAGE) outperforms attention (GAT) when neighbors are mostly relevant.
2. **BERT embeddings are already class-discriminative.** Attention has little signal to learn because the features themselves do most of the work.
3. **GraphSAGE's inductive design** matches the `inductive_directed` graph mode we use.

**Implication for the research:** the bottleneck is unlikely to be the GNN architecture. **The graph structure itself is the right thing to optimize.** That's why RL graph construction is the right next step.

The notebooks and `gnn_only.py` have been pruned to **GraphSAGE only** for the rest of the project.

### Confusion matrix takeaway

| Class | Recall |
|---|---|
| Negative | ~75% |
| **Neutral** | **~60%** ← weakest class, biggest opportunity |
| Positive | ~75% |

If your RL agent doesn't move the **neutral** class, it probably won't move the headline number. Track per-class metrics from day one.

### Data statistics (saved at `preprocessed/metadata.json`)

| Split | Total nodes | Negative | Neutral | Positive | Notes |
|---|---|---|---|---|---|
| Train (post-SMOTE) | 24,087 | 8,029 | 8,029 | 8,029 | balanced via SMOTE |
| Val | 4,963 | 1,405 | 2,008 | 1,550 | natural distribution |
| Test | 6,203 | 1,756 | 2,510 | 1,937 | natural distribution |
| **Graph total** | **35,253** | — | — | — | train + val + test stacked |

The PCA reduces BERT 768-dim embeddings to **128-dim** features. Z-score standardization is fit on train only. The graph (at K=15) has roughly 35k nodes and ~530k edges (train-internal undirected + train→val + train→test directed).

### Known soft warnings (audit findings, not blockers)

- **`hidden_dim=256` sits at the upper edge of the search space `{32, 64, 128, 256}`.** Widening to 512 is possible but probably not worth it — empirically, the gain from 128→256 was small.
- **`dropout=0.234` sits near the lower edge of `[0.2, 0.5]`.** The model is regularizing primarily via weight_decay (6e-5), not dropout. If you re-tune later, widen to `[0.05, 0.5]`.
- **Val−test gap is consistently +1.6 pp across all 10 seeds.** This is an artifact of `inductive_directed` graph mode (val nodes receive train→val messages during training, biasing the early-stopped checkpoint slightly toward val). It is **not** data leakage. Don't trust val accuracy as a proxy for test; report test.
- **Per-seed CI is unusually tight (~0.0022 on test acc).** Partly a property of MPS — its non-bit-determinism gives slightly inflated seed-to-seed correlation. Just note this when comparing RL against the baseline: tiny improvements may look "statistically significant" by Welch's t-test even when they're not practically meaningful.

---

## 6. Your first task: reproduce the baseline

The K-sensitivity sweep has already been run and gave a clear go-signal for the RL direction (see section 5). Your first task is to **reproduce the baseline numbers from a clean state** — that's how you'll know your environment matches before touching RL code.

### Reproduction steps

```bash
# 1. Clone the repo and install dependencies (see section 3)
# 2. Run notebook 01_preprocessing.ipynb end-to-end (~10-15 min)
# 3. Run notebook 02_graphsage.ipynb end-to-end (~3 hours on M3 Max GPU)
```

You should get test accuracy within ±0.005 of 0.6823. If you can't, **stop and fix the environment** before writing any RL code — drift in the baseline will make your RL evaluation noisy.

### Reproduction notes

The original baseline was run on:

- **Hardware:** Apple M3 Max, MPS backend
- **PyTorch:** 2.3+ recommended (MPS support has improved each minor version; below 2.3 may give ~20–40% slower runs and slightly different numbers due to scatter-op fallback differences)
- **Python:** 3.10 or 3.11

Before launching jupyter, set these environment variables in the shell:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1   # silently CPU-fallback unsupported MPS ops (PyG scatter etc.)
export OMP_NUM_THREADS=12               # cap BLAS threading on 14/16-core M3 Max — avoids oversubscription
export MKL_NUM_THREADS=12
```

These are also baked into cell 2 of the notebook for the env var, but setting them in the shell is belt-and-suspenders.

**Expected variance on re-run:** MPS is not bit-deterministic, so even with the same seed you'll see ±0.001–0.003 spread on test accuracy across reruns. CUDA users will see a slightly different CI shape (typically wider — CUDA respects `cudnn.deterministic` but kernel choice still varies).

### Common reasons the baseline won't reproduce

- Different PyTorch / PyG versions (especially MPS-related changes)
- Different sklearn version → slightly different KNN tie-breaking
- Different scikit-learn-extra / imbalanced-learn version → different SMOTE outputs
- A different GPU (CUDA results will differ slightly from MPS results due to non-determinism — that's expected, just larger CIs)

If the baseline diverges by more than 0.005 on test accuracy, pin versions to what `pip freeze` was at the time of the original run (ask the advisor).

### Suggested first-week checklist

| Day | Task | Why |
|---|---|---|
| 1–2 | Set up env, install deps, run 01_preprocessing | Sanity-check the env before the long run |
| 2–3 | Run 02_graphsage end-to-end (overnight is fine) | Confirm reproduction within ±0.005 of 0.6823 |
| 3–4 | Read this entire handover doc + audit comments in the notebook | Get the why behind every design choice |
| 4–5 | Read the must-read papers in section 10 (GraphSAGE, LDS-GNN, IDGL, NeuralSparse) | Build the mental model before writing RL code |
| 5–6 | Sketch the policy net + action space on paper, walk through it with the advisor | Validate the design before coding |
| 7 | Set up wandb / mlflow project + a minimal "PPO on a toy graph" smoke test | Have telemetry running before any real training |

If you finish faster, push to day 7's checklist item early. If you slip past day 5 without a clean baseline reproduction, that's a flag to escalate (see section 13).

### After reproducing — the go/no-go decision is already made

The K-sweep result (see section 5) tells us:

- Graph structure matters (K=5 to K=50 spans 1.1 pp with non-overlapping CIs)
- The peak is at K=50, but K=15 was chosen as the default for the published baseline
- RL has signal to chase between K=5 and K=50

You do **not** need to re-run the K-sweep. It's done. The result is at `runs/GETS_GNN_ONLY/metrics/k_sweep.csv`.

---

## 7. The phased research plan

### Phase 0 — Baseline (DONE)

- [x] BERT + PCA + SAGE pipeline
- [x] Optuna tuning + multi-seed CI (10 seeds)
- [x] K-sensitivity sweep (7 K values × 5 seeds, K=5 → K=100)
- [x] Search space audited: no best-params at edges of meaningful ranges
- [x] Train-val gap measured and acceptable (~10 pp)
- [x] Final test acc: **0.6823 ± 0.0022** (K=15 default), peak K-sweep value 0.6853 (K=50)

### Phase 1 — Within-dataset RL (months 1-3)

**Goal:** an RL agent that produces graphs better than any fixed-K KNN. Two concrete numbers to beat:

- **Easy target (the published baseline):** beat `0.6823` test acc (K=15 default) with non-overlapping 95% CIs. Roughly +0.5 pp would clear it.
- **Hard target (the K-sweep peak):** beat `0.6853` test acc (K=50, the strongest fixed-K KNN). Clearing this is what proves RL produces *meaningfully smarter* graphs and not just "a smarter K".

If you only beat the easy target but lose to the hard target, the contribution is real but the framing changes ("RL graph construction matches the best fixed-K, with the bonus of being learned"). If you clear both, you have a clean paper.

**Design (recommended):**

| Component | Choice | Why |
|---|---|---|
| Inner classifier | GraphSAGE, **frozen** after Phase 0 | Empirics say it's best; freezing makes the RL inner loop fast |
| Action space | **Discrete** edge prune-from-pool | KNN-50 candidate pool; agent picks ~10-15 to keep per node. Simpler than continuous weights and avoids SAGE's lack of native `edge_weight` support |
| Reward | Per-step val-accuracy delta, OR sparse final accuracy | Per-step is denser (easier to learn) but risks reward hacking — see gotchas |
| Algorithm | PPO with discrete actions | Standard, stable, well-documented |
| Policy net | Small SAGE encoder over current graph -> MLP head -> per-candidate logits | The policy itself is a GNN |
| State | Node features + current adjacency + classifier confidence | Don't over-engineer the state; let the policy learn |

**Milestones:**

- [ ] Reproduce Phase 0 numbers from scratch (within ±0.005 test acc)
- [ ] Set up experiment tracking (wandb / mlflow / tensorboard) — log every run from day one. RL is too noisy to debug from terminal output alone.
- [ ] Refactor splits to `60% train / 10% sage_val / 10% policy_val / 20% test` (see section 9 — this re-runs the whole pipeline)
- [ ] Implement candidate-pool builder (KNN-50)
- [ ] Implement PPO loop with frozen SAGE inner classifier
- [ ] Smoke test: PPO on a tiny graph (<1k nodes) end-to-end before scaling to 35k
- [ ] Beat **easy target** (0.6823) on val
- [ ] Beat **easy target** on test (the only number that counts)
- [ ] Beat **hard target** (0.6853) on test — stretch goal, makes the paper much stronger

**Recommended experiment tracking:**

- **wandb** is the standard choice — free for academic use, integrates cleanly with PyTorch/PyG, and gives you per-run hyperparameter sweeps + comparison plots.
- Alternatives: **mlflow** (self-hosted, no account needed) or **tensorboard** (built into PyTorch, no extra service).
- Whatever you pick, log: per-step reward, val accuracy of the modified graph, per-class accuracy, action histograms (which edges the agent is keeping/dropping), policy gradient norms.

### Phase 2 — Transfer setup (months 4-6)

**Goal:** show the policy works zero-shot or few-shot on a held-out dataset.

**Datasets to add:**
- Cora (citation network, homophilous)
- Citeseer (homophilous)
- Twitch (heterophilous, social network)
- Amazon-Photo (product co-purchase)

Train on N-1 datasets, evaluate zero-shot on the held-out. Mix homophilous + heterophilous so the policy can't memorize one regime.

**Critical for transfer:** the policy must condition on **intrinsic features** (node similarity, degree, neighborhood properties) — never on dataset identity. Otherwise it just memorizes per-dataset patterns and "transfer" is fake.

### Phase 3 — Meta-RL adaptation (months 7+)

**Goal:** few-shot adaptation on a new dataset using gradient steps at test time. MAML-style. This is the strongest scientific story but the riskiest to implement.

This is paper-2 territory. Don't start here.

---

## 8. Concrete things to change / experiment with

### In `01_preprocessing.ipynb`

| Change | Why try it | Expected impact |
|---|---|---|
| `HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"` | Domain-specific encoder usually beats generic BERT for tweets | Probably +1-3 accuracy points on the baseline |
| `BALANCE_METHOD = "none"` | SMOTE creates synthetic edges that may confuse the RL agent | Lower baseline accuracy but cleaner RL setup |
| `PCA_DIM = 256` | More feature capacity | Marginal improvement; bigger graph KNN cost |
| `EMBED_POOLING = "cls"` | Some encoders' CLS works better when fine-tuned | Probably worse with frozen BERT but worth checking |

### In `02_graphsage.ipynb`

The K-sweep is already done with `[5, 10, 15, 20, 30, 50, 100]` — no need to widen it further. Other knobs worth exploring:

| Change | Why try it | Expected impact |
|---|---|---|
| Set `OPTIMIZE_METRIC = "macro_f1"` | Optuna currently optimizes accuracy; F1 weights minority classes equally | Different best params; better neutral-class recall (currently 60%) |
| Set `GRAPH_MODE = "transductive"` | Counter-test: does the inductive constraint hurt? | If transductive is much better, the inductive constraint is costly |
| Widen `dropout` lower bound to `0.05` | Current best (0.234) is near the lower edge of [0.2, 0.5] | Possibly slightly better val acc; weight_decay does most of the regularizing already |
| Reduce `N_TRIALS` to 20 for quick iteration | 50-trial Optuna is slow during development | Use 50 only for final paper numbers |

### Beyond the existing code

- **Per-class accuracy:** SAGE might be lopsided across the 3 classes. Compute and report per-class precision/recall/F1.
- **Graph perturbation experiments:** randomly rewire 10/20/30% of edges; see how badly SAGE breaks. This gives an upper bound on how much the RL agent can hurt by exploring poorly.
- **Edge homophily by class:** is heterophily concentrated in one class (e.g., neutral)? If yes, the RL agent might get most of its win from fixing that one class.
- **Active learning baseline:** before going RL, try "label some uncertain val nodes and re-train" — if active learning beats fixed KNN by more than RL graph construction would, your novelty argument weakens.

---

## 9. Risks and gotchas

### Reward hacking the val split

If the RL agent is rewarded on val accuracy AND the SAGE classifier early-stops on val, you're double-dipping. The agent learns to game the same metric used for inner-loop convergence.

**Fix:** create a separate "policy-train val" split distinct from the SAGE early-stopping val. Concrete plan:

- Re-split: **60% train / 10% sage_val / 10% policy_val / 20% test**
- SAGE trains on train, early-stops on sage_val
- RL agent rewarded on policy_val
- Final eval on test (untouched)

**Important — this requires regenerating preprocessed data, not just relabeling.**

Concretely, the steps are:

1. Edit `01_preprocessing.ipynb` cell with `TEST_SIZE=0.20`, `VAL_SIZE=0.20`. Add a new `POLICY_VAL_SIZE=0.10` knob and produce a four-way stratified split.
2. Save `X_policy_val.npy`, `y_policy_val.npy` alongside the existing files.
3. Update `02_graphsage.ipynb` cell 6 to load the new array; update cell 8 (`build_pyg_data`) to add a `policy_val_mask` and extend the `inductive_directed` graph builder to attach policy_val nodes via directed train→policy_val edges (same pattern as val/test).
4. **You must re-tune the SAGE baseline** under the new split. The 60/10/10/20 changes the train size and therefore the SAGE checkpoint, so the published `0.6823` baseline number does not apply directly to the new split. Re-run the full pipeline and report new baseline numbers under the new split.

**Don't try to shortcut by sub-sampling the existing val set into sage_val / policy_val on the fly.** The early-stopping signal from SAGE was computed against the full val set; subsampling at RL-time gives you metrics that aren't comparable to the published baseline.

### SAGE doesn't natively use edge weights

`SAGEConv` aggregates with mean/max — there's no `edge_weight` parameter. If your RL agent outputs continuous edge weights, you must **either**:

1. Stay with discrete actions (recommended — simpler, faster).
2. Subclass `SAGEConv` and override `message()` to multiply by an edge_weight tensor (~30 lines of PyG).
3. Pre-multiply node features by neighbor weights before each forward pass (hacky but works).

### Compute budget

- Each RL step requires (re-)evaluating SAGE on the modified graph.
- With frozen SAGE: ~50ms per inference -> ~1000 steps/min. PPO with 1M steps -> ~17 hours per run.
- With re-training SAGE per RL step: forget it. **Always freeze.**
- Plan for ~10-50 RL runs during the project; budget 200-500 GPU-hours total.

### Differentiable baselines may beat RL

LDS-GNN (Franceschi et al., 2019) and IDGL (Chen et al., 2020) learn adjacency via bilevel optimization — **not RL**. They typically get 80-90% of the value with much simpler implementation.

**Your defense in the paper:** RL gives you **transferability** that bilevel optimization can't. Make sure your evaluation actually measures transfer (Phase 2). If you only show within-dataset wins, reviewers will rightly say "why not just use IDGL?"

### Reward variance on small val sets

Per-step delta in val accuracy is noisy when the val split is small (~6k tweets here). Smoothing options:
- Compute reward over fixed minibatches of the policy_val split, not the whole thing
- Use confidence-margin proxies (sum of max-softmax) as a smoother dense reward
- Use sparse final-accuracy reward instead of per-step (slower learning but more honest)

### Graph structure learning is often robustness-driven, not accuracy-driven

A lot of recent literature (ProGNN, GIB-GSL, Pro-GNN) frames graph structure learning as a **robustness** technique — defending against adversarial perturbations. They don't always show clean accuracy wins on clean data.

If your K-sweep is flat, consider pivoting toward **robustness**: train with adversarial perturbations, measure how much the RL policy recovers vs fixed KNN. Different framing, but still novel and publishable.

---

## 10. Related work to read (in priority order)

### Must-read (foundational)

1. **GraphSAGE** — Hamilton et al., NeurIPS 2017. The model.
2. **LDS-GNN** — Franceschi et al., ICML 2019. Differentiable graph learning, your direct competitor.
3. **IDGL** — Chen et al., NeurIPS 2020. Iterative joint graph + GNN learning.
4. **NeuralSparse** — Zheng et al., ICML 2020. Differentiable edge sparsification.

### RL + graphs

5. **Combinatorial optimization with RL on graphs** — Khalil et al., NeurIPS 2017 (S2V-DQN). The classic "RL learns to act on graphs" paper.
6. **GraphNAS** — Gao et al., IJCAI 2020. RL for GNN architecture search (different problem, but methodologically relevant).
7. **GCPN** — You et al., NeurIPS 2018. RL for molecular graph generation.

### Graph structure learning broadly

8. **ProGNN** — Jin et al., KDD 2020. Robustness-driven graph learning.
9. **Pro-GNN, GIB-GSL** — recent surveys of graph structure learning under information bottleneck. <!-- TODO: advisor to fill in specific citations -->

### Transfer learning on graphs

10. **GraphMAML** — recent meta-learning for GNNs. <!-- TODO: advisor to fill in specific citation -->
11. **Pre-train, Prompt** — newer paradigm for cross-dataset GNN transfer. <!-- TODO: advisor to fill in specific citation (e.g., GPF, GraphPrompt) -->

> **Note for the student:** items 9, 10, 11 above are placeholder framings — the specific papers haven't been pinned down yet. Ask the advisor for the current best entry points before reading.

---

## 11. FAQ / common pitfalls

**Q: Why don't we re-tune Optuna at each `KNN_K` in the K-sweep?**
A: It would 5x the compute and the question we're asking ("does graph structure matter?") doesn't require re-tuning. Use the cached best params from the default K=10 baseline.

**Q: The K-sweep is flat. Project dead?**
A: Not necessarily. Try (a) re-running with `BALANCE_METHOD="none"` (SMOTE may be hiding signal), (b) more aggressive K range like `[2, 3, 5, 100, 200]`, (c) different graph metric (`KNN_METRIC="euclidean"`). If still flat after these, consider the robustness-pivot framing in section 9.

**Q: Should I fine-tune BERT instead of using frozen embeddings?**
A: Probably not for this project. Fine-tuning BERT would likely give you ~3-5 more accuracy points but eats most of the GPU budget. The graph isn't the bottleneck, BERT is — but improving BERT is a different paper. Stay focused on graph construction.

**Q: SMOTE creates synthetic samples — do they get edges in the KNN graph?**
A: Yes. SMOTE-generated points are real points in feature space; they get KNN neighbors like any other node. **For the RL extension this is a real complication** — the agent might learn "bad" patterns from synthetic edges. Try `BALANCE_METHOD="none"` to compare.

**Q: How long should I spend on Phase 1 before judging it?**
A: Set a budget of ~6 weeks. If after 6 weeks of RL training the agent can't beat the K=15 baseline (test acc 0.6823) by at least 0.5 accuracy points with non-overlapping CI, the within-dataset story is unlikely to hold up. Pivot or refine.

**Q: Can I add more GNN architectures back?**
A: You *can*, but the project is more focused on the RL component. Adding architectures dilutes the contribution. Better to stick with SAGE and put the energy into the policy design.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Homophily** | Fraction of edges where both endpoints share the same label. High homophily = neighbors agree |
| **Heterophily** | Opposite — neighbors tend to differ. Harder for vanilla GCN/SAGE; favors GAT or specialized layers |
| **Inductive vs transductive** | Inductive = the model handles unseen nodes at test time. Transductive = test nodes were present during training (just unlabeled) |
| **Message passing** | The general framework: each layer aggregates messages from neighbors and updates the node's representation |
| **Over-smoothing** | After many GCN layers, all node embeddings converge — a real problem for stacking depth |
| **Soft voting (multi-seed)** | Average per-seed softmax probabilities, then argmax. Usually beats any single seed |
| **TPE (Optuna)** | Tree-structured Parzen Estimator, a Bayesian-style sampler for hyperparameter search |
| **PPO** | Proximal Policy Optimization. Default RL algorithm for continuous control and discrete action spaces |
| **MAML** | Model-Agnostic Meta-Learning. Few-shot adaptation via second-order gradient steps |
| **Bilevel optimization** | Inner loop trains the model; outer loop adjusts something (here, the graph). LDS-GNN does this |

---

## 13. Where to ask for help

- The advisor (sentiment analysis, NLP, GNNs)
- For RL specifically: stable-baselines3 docs, CleanRL implementations, the OpenAI Spinning Up tutorials
- For PyG-specific quirks: the PyG GitHub issues are searchable and active

### When to escalate to the advisor (don't spin alone)

Push through small issues yourself, but **escalate immediately** in any of these cases — they're either signals that something fundamental is wrong, or they're things the advisor can resolve in 5 minutes that would take you days:

| Situation | Why escalate |
|---|---|
| Test acc reproduction off by >0.005 after fixing PyTorch/PyG versions | Something deeper is wrong (data, splits, env). Don't write RL on top of a broken baseline. |
| Optuna best params land on different values by >50% (e.g., new run picks `hidden_dim=64` instead of 256) | TPE seed not honored, or upstream data changed. |
| KNN graph build takes >10x the documented time | sklearn / BLAS misconfigured; the advisor has hit this before. |
| RL training loss diverges or NaNs in the first 1k steps | Almost always reward scaling or a state-encoding bug; the advisor will spot it fast. |
| You can't decide between two reasonable design choices and have spent >2 days on it | Most "either is fine, but…" decisions need the advisor's prior context, not more thinking. |
| Phase 1 budget (6 weeks) is half-spent and the agent still can't beat 0.6823 on val | Time to discuss pivot vs. push. Don't burn the second half hoping. |
| New transfer dataset (Cora/Citeseer/etc.) won't load or has very different scale | Probably a known PyG issue; advisor likely has the workaround. |
| You find a result that contradicts something in this handover | Important — flag it. The doc may be wrong. |

**How to escalate efficiently:**

1. State the symptom (one sentence)
2. State what you've already tried (bullet list, max 5 items)
3. Attach the smallest reproducible snippet or log
4. Send via the channel the advisor prefers (probably email or repo issue)

A 4-line escalation gets a same-day reply. A meandering paragraph gets queued.

---

**Last updated:** 2026-04-30 (final baseline: K=15, test acc 0.6823 ± 0.0022; K-sweep complete; project ready for Phase 1)
**Maintained by:** Fadi Almahamid (advisor)
**For questions:** open an issue on this repo or email the advisor directly
