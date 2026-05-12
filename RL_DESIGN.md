# RL graph construction - design doc

_last updated: 2026-05-12. reflects fadi's revisions + graphrare findings + dataset pivot to cora/pubmed._

---

## problem framing

replace the static graph used by graphsage with one optimized by an RL agent. the frozen graphsage classifier is the "environment" -- the agent edits the graph, sage scores it, agent gets rewarded.

primary datasets: **cora** and **pubmed** (planetoid). baseline numbers confirmed:
- cora: graphsage 0.879 ± 0.013, MLP 0.757 ± 0.012 (12pp gap)
- pubmed: graphsage 0.886 ± 0.003, MLP 0.876 ± 0.003 (1pp gap -- cora is primary)

---

## contribution pillars (differentiating from graphrare)

three things that make this publishable beyond graphrare:

**1. novel reward design -- class-imbalance-aware**
```
r_t = delta_macro_f1_t + gamma * delta_f1_minority_t
```
graphrare uses delta accuracy + delta loss on the training set. we use macro-F1 + a minority class bonus on a held-out policy_val split. this directly targets class imbalance rather than optimizing accuracy which can ignore minority classes entirely. gamma is a tunable weight (start with 0.5).

**2. PPG over PPO**
we use phasic policy gradient (cobbe et al. 2021) via cleanrl rather than vanilla PPO. PPG separates policy and value function updates into alternating phases, giving better sample efficiency and stability. graphrare used PPO -- PPG is a direct upgrade on the same framework.

**3. head-to-head vs differentiable graph learning methods**
we compare against LDS-GNN and IDGL (differentiable graph structure learning), not just fixed-graph baselines. this answers "is RL worth the added complexity over differentiable alternatives?" -- a question graphrare never asks.

---

## components

### 1. candidate edge pool

two sources of candidate edges per node:

- **existing graph edges** -- original citation edges in cora/pubmed. starting topology.
- **kNN-k feature pool** -- built once from node features. gives candidate edges not in the original graph.

agent can add from the kNN pool or remove from existing edges. both operations matter (graphrare ablation: add-only and remove-only both underperform vs add+remove).

### 2. state

per-node state using structural features (not raw node features):

```
s_i = [h_i || sage_logits_i || degree_i || homophily_i || structural_entropy_i]
```

- `h_i`: graphsage embedding for node i from frozen model (hidden_dim)
- `sage_logits_i`: frozen sage softmax output (n_classes)
- `degree_i`: current degree (scalar)
- `homophily_i`: fraction of neighbors with same predicted class (scalar)
- `structural_entropy_i`: entropy of neighbor label distribution (scalar)

note: we use structural features rather than raw node features (1433-dim for cora) in the state. raw features are available to sage already -- the policy needs graph-level signals to decide which edges to change.

### 3. action space

**per-edge, sequential:**

at each step the agent picks one edge action:
```
(node_i, node_j, op)  where op in {add, remove}
```

- add: draw from kNN candidate pool for node i
- remove: remove an existing edge incident to node i

one episode = 50-100 sequential edge edits. the agent sees updated graph state between edits (not bandit).

**constraint:** min degree >= 2 per node (prevent isolated nodes).

### 4. policy network

**MLP (not a GNN policy):**

graphrare used an MLP policy and achieved strong results. GNN policy adds complexity without clear benefit.

```
input: [s_i || s_j || edge_features_ij]
  - edge_features_ij: cosine_sim(h_i, h_j), delta_degree, in_original_graph (bool)
-> MLP: (state_dim -> 256 -> 128 -> 2)
-> softmax over {add, remove}
```

### 5. reward

**dense, class-imbalance-aware, every ~5 edits, EMA-smoothed:**

```
r_t = delta_macro_f1_t + gamma * delta_f1_minority_t
```

- `delta_macro_f1_t`: change in macro-F1 on policy_val vs episode start
- `delta_f1_minority_t`: change in F1 of the minority class specifically
- `gamma`: minority class weight (default 0.5, tune in ablation)
- computed every 5 edits, smoothed with EMA (alpha=0.3)
- evaluated on `policy_val` split (held out from sage training entirely)

rationale: accuracy-based reward ignores minority class failures. macro-F1 + minority bonus directly penalizes ignoring hard classes. graphrare uses training-set accuracy which can overfit to majority class.

### 6. episode structure

```
1. start: original citation graph for cora/pubmed
2. for step in range(50-100):
   a. policy reads current graph state
   b. samples (node_i, node_j, op) from action distribution
   c. apply edit to graph
   d. every 5 steps: run frozen sage forward on policy_val, compute r_t
3. PPG update (policy phase + value phase alternating)
```

episode length: 50-100 steps. full graph state updated between steps.

### 7. algorithm

**PPG (phasic policy gradient) via cleanrl:**

skip REINFORCE (high variance) and go straight to PPG.

PPG config:
- clip ratio epsilon = 0.2
- policy phase: update policy network
- auxiliary phase: update value head with distillation loss
- entropy bonus: 0.01 (encourage exploration early)
- use cleanrl PPG implementation

**fallback:** if PPG is unstable early, start with PPO for first 200 episodes then switch to PPG.

---

## graphrare comparison

| dimension | graphrare | our approach |
|---|---|---|
| backbone | co-trained GNN + DRL | frozen graphsage (no co-training) |
| target graphs | heterophilic (chameleon, squirrel...) | homophilic (cora, pubmed) |
| reward | delta acc + loss on train set | delta macro-f1 + minority bonus on policy_val |
| algorithm | PPO | PPG (cleanrl) |
| comparison | vs fixed-graph GNNs only | vs fixed-graph GNNs + differentiable methods (LDS-GNN, IDGL) |
| action | per-node add k + remove d | per-edge add/remove sequential |
| state | node features + structural entropy | structural features only (degree, homophily, entropy, sage embeddings) |
| policy net | MLP | MLP |
| graph used | original + long-range edges | original citation + kNN candidates |

---

## targets to beat (cora)

| target | value | what it means |
|---|---|---|
| MLP baseline | 0.757 ± 0.012 | proves graph adds value at all |
| graphsage baseline | 0.879 ± 0.013 | beats standard fixed graph |
| graphsage-RARE (graphrare) | ~0.890 | matches competing RL method |

---

## open assumptions

| assumption | risk | mitigation |
|---|---|---|
| frozen sage is sufficient signal | agent may not get enough gradient to learn | if val f1 doesn't move in 50 eps, try co-training |
| 50-100 edits is right episode length | too few = can't reshape graph, too many = unstable | sweep [20, 50, 100] in ablation |
| kNN pool gives useful candidates | citation edges already high quality, kNN may add noise | start with remove-only from existing edges, add kNN later |
| MLP policy is sufficient | may need graph context to make good decisions | fallback: 1-layer sage as policy encoder |
| gamma=0.5 for minority bonus | wrong weighting hurts convergence | sweep gamma in [0.1, 0.5, 1.0] |

---

## implementation order

1. `GraphEnv` class -- wraps frozen sage + policy_val eval + episode step logic
2. `PolicyNet` class -- MLP over edge state pairs
3. PPG training loop -- cleanrl-based, ~100 episodes smoke test on cora
4. logging -- wandb: per-episode reward, val f1, minority class f1, action histograms, policy gradient norms
5. ablation -- episode length, gamma, reward frequency, add-only vs remove-only vs both
6. LDS-GNN + IDGL comparison runs
7. evaluate vs graphrare with 10-split CI on cora + pubmed
