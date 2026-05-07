# RL graph construction - design doc

## problem framing

replace the static kNN graph construction step with an RL agent that learns which edges to keep from a kNN-50 candidate pool. the frozen graphsage classifier is the "environment" - the agent acts on the graph, sage scores it, agent gets rewarded.

---

## components

### 1. candidate edge pool (built once, reused)

pre-compute kNN-50 graph over training nodes. for each node `i`, this gives 50 candidate neighbors. the agent chooses a subset to keep (target: ~15, matching the K=15 baseline).

- built once before RL training starts
- stored as a (n_train, 50) index tensor
- never recomputed during RL loop

### 2. state

for each node `i`, the state is:

```
s_i = [x_i || mean(x_neighbors) || sage_logits_i || degree_i]
```

- `x_i`: node feature vector (128-dim, pca+zscore)
- `mean(x_neighbors)`: mean of current neighbors' features (128-dim)
- `sage_logits_i`: frozen sage output for node i (3-dim softmax)
- `degree_i`: current degree (scalar)

total state dim per node: ~260-dim. the policy sees the state for ALL nodes simultaneously (full-graph episode).

### 3. action space

**discrete, per-node:**

for each node `i`, the agent outputs a binary keep/prune decision over its 50 candidates:

```
a_i ∈ {0, 1}^50   (keep=1, prune=0)
```

total action space: n_train × 50 binary decisions per episode. this is large - we handle it by:
- treating each node's 50 decisions independently (factorized policy)
- policy net outputs 50 logits per node → sigmoid → bernoulli sample

**constraint:** enforce minimum degree (at least 5 edges kept per node) to prevent isolated nodes.

### 4. policy network

small GNN that reads the current graph state and outputs per-candidate keep probabilities:

```
input: node features x (n × 128)
→ sage layer 1: (128 → 64)
→ relu
→ sage layer 2: (64 → 32)
→ for each node i: MLP(h_i || h_candidate_j) → scalar logit
→ sigmoid → keep probability p_ij for each candidate j
```

the policy is itself a 2-layer graphsage. small on purpose - we want it fast.

### 5. reward

**sparse episode reward** (one reward per episode, not per step):

```
r = macro_f1(frozen_sage, policy_val) - macro_f1(frozen_sage, kNN_baseline)
```

- evaluated on `policy_val` split (completely separate from sage's `sage_val`)
- delta F1 relative to the K=15 kNN baseline at the start of training
- focused on macro-F1 (not accuracy) to penalize neutral class failures

**why sparse:** per-step rewards are noisy on 2.5k-node val sets. sparse final reward is slower to learn but more honest. start with sparse, switch to dense if convergence is too slow.

### 6. episode structure

one episode = one full graph construction attempt:

```
1. start from kNN-50 candidate pool (fixed)
2. policy net reads node states → outputs keep/prune for all edges
3. construct graph from kept edges
4. run frozen sage forward pass on new graph (inference only, no grad)
5. compute macro-F1 on policy_val
6. reward = delta F1 vs baseline
7. policy gradient update (REINFORCE or PPO)
```

episode length: 1 "step" (all decisions made simultaneously). this is a bandit-style setup, not sequential.

### 7. algorithm

**start with REINFORCE:**
- simple, no value function needed
- high variance but interpretable
- if unstable after 200 episodes → switch to PPO

**PPO setup (if needed):**
- clip ratio ε = 0.2
- value head: MLP(mean_node_embedding) → scalar
- entropy bonus: 0.01 (encourage exploration early)
- use stable-baselines3 or cleanrl PPO implementation

---

## open questions (resolved)

| question | decision |
|---|---|
| construction once per run or per sample? | once per episode (full graph, ~35k nodes) |
| freeze sage during RL? | yes - always frozen |
| episode length? | 1 step (bandit) |
| reward signal? | sparse delta macro-F1 on policy_val |
| SMOTE synthetic nodes in candidate pool? | include - filter later if signal is noisy |

---

## targets to beat

| target | value | what it means |
|---|---|---|
| easy (new baseline) | test acc 0.6819 ± 0.0025 | beat with non-overlapping CI |
| hard (k-sweep peak) | test acc ~0.685 (K=50, new split pending) | proves RL > best fixed-K |
| stretch (MLP) | test F1 0.689 | proves graph adds value at all |

neutral class recall is the leading indicator - if it doesn't move, headline number won't either.

---

## implementation order

1. `GraphEnv` class - wraps frozen sage + policy_val evaluation
2. `PolicyNet` class - 2-layer sage + per-candidate MLP head  
3. REINFORCE training loop - ~100 episodes smoke test on subset
4. logging - wandb: per-episode reward, val F1, action histograms, policy gradient norms
5. scale to full graph
6. evaluate vs baseline with 10-seed CI
