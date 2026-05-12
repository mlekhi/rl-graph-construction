# RL graph construction - design doc

_last updated: 2026-05-12. reflects fadi's revisions + graphrare findings + dataset pivot to cora/pubmed._

---

## problem framing

replace the static graph used by graphsage with one optimized by an RL agent. the frozen graphsage classifier is the "environment" - the agent edits the graph, sage scores it, agent gets rewarded.

primary datasets: **cora** and **pubmed** (planetoid). these have a clear GNN-vs-MLP gap (5-8pp) so RL graph construction has headroom to demonstrate value. sentiment dataset dropped -- MLP was already competitive with graphsage there.

---

## components

### 1. candidate edge pool

two sources of candidate edges per node:

- **existing graph edges** -- original citation edges in cora/pubmed. starting topology.
- **kNN-k feature pool** -- built once from node features (bag-of-words for cora, tfidf for pubmed). gives candidate edges not in the original graph.

agent can add from the kNN pool or remove from existing edges. both operations matter (graphrare ablation: add-only and remove-only both underperform vs add+remove).

### 2. state

per-node state:

```
s_i = [x_i || h_i || sage_logits_i || degree_i || structural_entropy_i]
```

- `x_i`: node features (1433-dim cora, 500-dim pubmed)
- `h_i`: graphsage embedding for node i from frozen model (hidden_dim)
- `sage_logits_i`: frozen sage softmax output (n_classes)
- `degree_i`: current degree (scalar)
- `structural_entropy_i`: entropy of neighbor label distribution (scalar) -- signals heterophily locally

structural entropy is cheap to compute and was shown critical in graphrare ablation (table v, GCN-RA row).

### 3. action space

**per-edge, sequential:**

at each step the agent picks one edge action:
```
(node_i, node_j, op)  where op in {add, remove}
```

- add: draw from kNN candidate pool for node i
- remove: remove an existing edge incident to node i

one episode = 50-100 sequential edge edits (fadi's revision). this is not bandit -- the agent sees updated graph state between edits.

**constraint:** min degree >= 2 per node (prevent isolated nodes).

### 4. policy network

**MLP (not a GNN policy):**

graphrare used an MLP policy and achieved strong results. GNN policy adds complexity without clear benefit.

```
input: [s_i || s_j || edge_features_ij]
  - edge_features_ij: cosine_sim(x_i, x_j), delta_degree, in_original_graph (bool)
-> MLP: (state_dim -> 256 -> 128 -> 2)
-> softmax over {add, remove}
```

policy outputs action probabilities over candidate edge pairs. evaluated for each candidate at each step.

### 5. reward

**dense reward, every ~5 edits, EMA-smoothed (fadi's revision):**

```
r_t = macro_f1(frozen_sage, policy_val) - baseline_f1
```

- evaluated on `policy_val` split (held out from sage training)
- only computed every 5 edits (not every single step) to reduce noise
- smoothed with EMA (alpha=0.3) to stabilize gradient signal
- delta relative to frozen kNN baseline at episode start

rationale: sparse reward (one per episode) was too slow to converge. dense every-step reward was too noisy on small val sets. every-5 with EMA is the middle ground.

### 6. episode structure

```
1. start: original citation graph for cora/pubmed
2. for step in range(50-100):
   a. policy reads current graph state
   b. samples (node_i, node_j, op) from action distribution
   c. apply edit to graph
   d. every 5 steps: run frozen sage forward on policy_val, compute reward
3. policy gradient update (PPO)
```

episode length: 50-100 steps. full graph state updated between steps.

### 7. algorithm

**PPO directly (skip REINFORCE -- fadi's revision):**

REINFORCE has too high variance for this setup. go straight to PPO.

PPO config:
- clip ratio epsilon = 0.2
- value head: MLP(mean_node_embedding) -> scalar
- entropy bonus: 0.01 (encourage exploration early)
- update every episode
- use cleanrl PPO implementation (simpler than stable-baselines3 for custom envs)

**then PPG if PPO plateaus:**
- PPG (phasic policy gradient) separates policy and value updates
- better sample efficiency than PPO
- only move to PPG if PPO doesn't converge after ~200 episodes

---

## graphrare comparison

| dimension | graphrare | our approach |
|---|---|---|
| backbone | co-trained GNN + DRL | frozen graphsage (no co-training) |
| target graphs | heterophilic (chameleon, squirrel...) | homophilic (cora, pubmed) |
| edge prior | node relative entropy ranking | no prior (pure RL exploration) |
| action | per-node add k + remove d | per-edge add/remove sequential |
| reward | delta acc + loss on train set | delta macro-f1 on policy_val |
| policy net | MLP | MLP |
| algorithm | PPO | PPO -> PPG |
| graph used | original + long-range edges | original citation + kNN candidates |

key differentiator: we freeze the backbone (prevents reward hacking, cleaner evaluation), and we evaluate on a held-out policy_val split rather than the training set. the frozen backbone means our SAGE params don't shift during RL, which graphrare's co-training approach doesn't guarantee.

---

## targets to beat (cora, pending baseline run)

_baseline numbers from 03_baseline_planetoid.py -- to be filled in after run completes._

| target | value | what it means |
|---|---|---|
| easy | MLP test acc (TBD) | proves graph adds value at all |
| medium | graphsage baseline test acc (TBD) | beats standard fixed graph |
| hard | graphsage-RARE on cora (~89%) | matches competing RL method |

graphrare reported graphsage on cora at ~75.7% and graphsage-RARE at ~89.0% (table III, 60/20/20 splits). our MLP baseline will be ~74-75% based on graphrare's reported 74.61%.

neutral class recall (cora class imbalance is mild, pubmed has 3 classes) -- less of a concern than on sentiment. macro-f1 and accuracy are close here.

---

## open assumptions

| assumption | risk | mitigation |
|---|---|---|
| frozen sage is sufficient signal | agent may not get enough gradient to learn | if val f1 doesn't move in 50 eps, try co-training |
| 50-100 edits is right episode length | too few = can't reshape graph, too many = unstable | sweep [20, 50, 100] in ablation |
| kNN pool gives useful candidates | citation edges already high quality, kNN may add noise | start with remove-only from existing edges, add kNN later |
| MLP policy is sufficient | may need graph context to make good decisions | fallback: 1-layer sage as policy encoder |
| dense every-5 reward is informative | 2.5k policy_val nodes may still be noisy | if variance too high, try every-10 or full episode |

---

## implementation order

1. `GraphEnv` class -- wraps frozen sage + policy_val eval + episode step logic
2. `PolicyNet` class -- MLP over edge state pairs
3. PPO training loop -- cleanrl-based, ~100 episodes smoke test on cora
4. logging -- wandb: per-episode reward, val f1, action histograms (add vs remove ratio), policy gradient norms
5. ablation -- episode length, reward frequency, add-only vs remove-only vs both
6. evaluate vs graphrare baseline with 10-split CI on cora + pubmed
