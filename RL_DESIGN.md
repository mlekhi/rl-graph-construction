# GraphHARE — RL graph construction with homophily-aware reward

_last updated: 2026-05-18. reflects revised contribution framing (homophily-aware reward), homophily diagnostic as pre-training gate, and dataset pivot to cora/pubmed/citeseer._

---

## name

**GraphHARE** — Homophily-Aware Reward for Edges. Directly contrasts with GraphRARE (Relative Entropy Reward): GraphRARE rewards on entropy-ranked candidate edges with an accuracy/loss-based reward signal; GraphHARE rewards on the homophily of the resulting graph in addition to classification gain.

---

## research question and contribution

GraphRARE (ICDE 2024) established that RL can improve a base graph for downstream node classification. Their agent receives reward based on classification accuracy and loss, and they report homophily improvements as a _byproduct_ of training — not as a designed objective.

**GraphHARE's research question**: what if homophily is incorporated _directly_ into the reward, instead of left as a byproduct?

### Primary contribution (single, focused)

A **homophily-aware reward function** for RL-based graph structure learning. The reward explicitly combines:

1. Downstream classification improvement (the goal)
2. Edge homophily of the resulting graph (a structural prior)

allowing the agent to explicitly trade off accuracy gains against structural cleanliness. To our knowledge, **no prior RL graph construction method incorporates homophily directly into the reward signal**. Verified through literature search:

- GraphRARE (ICDE 2024): accuracy + loss reward; homophily is a byproduct, not in the signal
- HoLe (CIKM 2023): homophily-as-objective, but for unsupervised clustering and not RL
- "It Takes a Graph" (arXiv 2025): rewires for homophily via theoretical diffusion, not RL
- DHGR, TRIGON, ComFy: heuristic or supervised graph rewiring, not RL

### What this paper is NOT

- Not a new RL algorithm (we use standard policy optimization)
- Not a new GNN architecture (we use GraphSAGE)
- Not a head-to-head benchmark of RL vs differentiable methods (that's an experimental section, not a contribution)
- Not a "frozen classifier methodology" claim (frozen classifier with RL has been done — RELIEF, KDD 2025 — and is good practice rather than novelty)

### Target venue

Mid-tier IEEE conference (IEEE BigData, IJCNN) as a methodology paper.

---

## pre-training requirement: homophily diagnostic

**Mandatory step before any RL training begins.** Compute edge homophily of:

1. The starting citation graph (cora/pubmed/citeseer)
2. The candidate edge pool derived from feature-similarity KNN
3. (Later) the RL-refined graph at the end of training

For C-class balanced data, the random-graph baseline is 1/C. If the candidate edge pool's homophily is close to the random baseline, graph-based refinement is unlikely to recover meaningful improvement (the graph would be encoding signal already present in the features). The advisor has applied this diagnostic in recent work and confirmed it is a reliable pre-training screen — treat it as a gate, not as post-hoc reporting.

If a dataset's candidate pool fails the check, flag immediately and decide whether to drop the dataset before investing GPU time.

Standard formulation: Zhu et al. 2020, *Beyond Homophily in Graph Neural Networks*.

---

## datasets

**Primary**: Cora and PubMed.

**Required for breadth**: Citeseer.

**Optional stretch**: one heterophilic benchmark (Chameleon or Cornell) from GraphRARE's set, if buffer time allows.

Sentiment dataset dropped — MLP was already competitive with GraphSAGE, indicating the constructed graph contributes minimally beyond features.

---

## components

### 1. candidate edge pool

Two sources of candidate edges per node:

- **existing graph edges** — original citation edges in cora/pubmed/citeseer. Starting topology.
- **kNN-k feature pool** — built once from node features (bag-of-words for cora, tfidf for pubmed). Gives candidate edges not in the original graph.

Agent can add from the kNN pool or remove from existing edges. Pre-training homophily must be measured on both the starting graph and the kNN pool (see above).

### 2. state

Per-node state (compact, structural-leaning):

- node features (bag-of-words / tfidf, with optional projection)
- frozen graphsage hidden embedding
- frozen graphsage softmax output (class probabilities)
- current degree (scalar)
- structural entropy: entropy of the neighbor label distribution under graphsage's predictions
- local homophily proxy: fraction of neighbors with matching graphsage-predicted label

The last two are structural signals; they are the natural state features for an agent whose reward incorporates homophily.

### 3. action space

**Per-edge, sequential:**

At each step the agent picks one edge action `(node_i, node_j, op)` where `op ∈ {add, remove}`.

- add: draw from kNN candidate pool for node i
- remove: remove an existing edge incident to node i

One episode = 50–100 sequential edge edits. Not bandit — the agent sees the updated graph state between edits.

**Constraint**: minimum degree ≥ 2 per node (prevent isolated nodes).

### 4. policy network

**MLP over edge state pairs:**

Input is the concatenation of state_i, state_j, and edge-level features (cosine similarity of features, current degree delta, in-original-graph indicator). Output is a softmax over `{add, remove}` and a candidate-score head.

MLP over a GNN policy here is justified: the structural state features already encode local graph context, so message passing inside the policy is not strictly required for this task scope. GraphRARE also used an MLP policy.

### 5. reward (the contribution)

**Homophily-aware reward, computed every M=5 edits with EMA smoothing.**

Conceptually, the reward at evaluation step _t_ combines two terms:

1. **Classification term**: change in macro-F1 on the held-out policy_val between consecutive evaluation points, relative to a fixed baseline F1 computed once at episode start.
2. **Homophily term**: change in edge homophily of the graph between consecutive evaluation points.

A hyperparameter β ∈ [0, ∞) controls the weight of the homophily term. The accuracy-only baseline reduces to β = 0; the homophily-only baseline is the limit β → ∞; the proposed homophily-aware reward sits between them.

**Ablation structure** (this is the empirical core):

- β = 0 (accuracy-only): replicates GraphRARE's reward design
- β > 0, multiple values (homophily-aware): our proposed family of rewards
- β → ∞ proxy (homophily-only): an extreme to characterize the trade-off

Per-step F1 noise on the small policy_val (~270 nodes for cora) is real. We mitigate via:

- evaluating every M=5 edits, not every step
- exponential moving average over the last 5–10 evaluation deltas

Rationale: sparse (one reward per episode) was too slow; dense every-step was too noisy. Every-5 with EMA is the middle ground that matches the existing literature's reward density.

### 6. episode structure

Initial state: starting citation graph (cora/pubmed/citeseer).

Per step: agent reads current graph state, samples an edge action, applies the edit. Every 5 steps the graph is evaluated for macro-F1 and edge homophily on policy_val. Reward is the homophily-aware combination.

Episode terminates after the edit budget (50–100 edits) or when no remaining candidate has Q above a threshold.

### 7. algorithm

**PPO via stable-baselines3** to get the loop running. Replace with **PPG (Phasic Policy Gradient, Cobbe et al. 2021)** if PPO plateaus or shows instability after 200 episodes. The algorithm choice is not a contribution; the reward design is.

PPO hyperparameters:

- clip ratio ε = 0.2
- entropy bonus = 0.01
- update every episode
- standard advantage estimation (GAE)

Don't roll your own — use library implementations (stable-baselines3 for PPO, CleanRL for PPG).

---

## graphrare comparison

| dimension | GraphRARE | GraphHARE (us) |
|---|---|---|
| reward signal | Δaccuracy + λ · Δloss | **Δmacro-F1 + β · Δhomophily** (the contribution) |
| homophily | byproduct, reported as result | **direct training signal** |
| backbone | co-trained GNN + DRL | frozen graphsage (good practice; not the contribution) |
| target graphs | both homophilic (cora, pubmed) and heterophilic (chameleon, squirrel, ...) | homophilic primary (cora, pubmed, citeseer); optional heterophilic stretch |
| edge prior | node relative entropy ranking | feature kNN pool + existing edges |
| action | multi-discrete per-node (k add + d remove) | per-edge sequential add/remove |
| reward eval | training-set accuracy/loss | held-out policy_val (good practice; not the contribution) |
| policy net | MLP | MLP |
| algorithm | PPO | PPO (PPG if needed) |

The contribution row is the reward signal. Everything else is either good practice (frozen backbone, held-out eval) or a design choice with no novelty claim.

---

## targets and what success looks like

Baseline numbers on Cora (under your 60/10/10/20 split, 10 seeds):

- MLP: 0.757 ± 0.012
- GraphSAGE (default graph): 0.879 ± 0.013

What we want to see from GraphHARE on Cora:

- Beat GraphSAGE (default graph) with non-overlapping CIs at some β > 0
- Characterize the β trade-off: how does accuracy improvement scale with the homophily weight?
- Demonstrate distinct structural properties (final-graph homophily) compared to β = 0

If β = 0 and β > 0 produce indistinguishable results, the contribution weakens to "GraphRARE's accuracy gains are robust to reward design" — still publishable, weaker positioning. If they produce measurably different graphs and different generalization, the contribution is strong.

PubMed has a narrower MLP-vs-GraphSAGE gap (~1 pp). Treat as secondary; success here is "method works, gain is small."

Citeseer is required for breadth.

---

## open assumptions and risks

| assumption | risk | mitigation |
|---|---|---|
| Homophily-aware reward produces distinguishably different graphs from accuracy-only | If they converge to the same solution, contribution weakens | Run ablation early (week 5). If β = 0 ≈ β > 0, adjust β range or pivot framing toward "robustness of accuracy-only" |
| Frozen graphsage is sufficient signal | Agent may not get enough gradient to learn | If val f1 doesn't move in 50 episodes, try (a) co-training as a fallback, (b) richer state features |
| 50–100 edits is right episode length | Too few = can't reshape graph; too many = unstable | Sweep [20, 50, 100] in ablation |
| KNN pool gives useful candidates | Citation edges already high quality; KNN may add noise edges | The pre-training homophily check on the KNN pool catches this before training |
| Dense every-5 reward is informative | Small policy_val (~270 cora) may still be noisy | If variance too high, try every-10 or full-episode |

---

## implementation order

1. **Pre-training homophily check** on Cora, PubMed, Citeseer (starting graph + KNN pool). Flag any dataset where the pool is near random baseline. This is the gate.
2. `GraphEnv` class — wraps frozen graphsage, manages graph state, applies edge edits, computes the dual reward signal (classification + homophily).
3. `PolicyNet` class — MLP over edge state pairs.
4. **PPO training loop** — stable-baselines3-based, ~100 episodes smoke test on Cora with β = 0 (replicate GraphRARE-style accuracy-only baseline).
5. **β ablation** — train at β ∈ {0, 0.1, 0.5, 1.0, 2.0} and characterize the trade-off curve.
6. **Logging via wandb**: per-episode reward, val F1, edge homophily of current graph, action histograms (add vs remove ratio), policy gradient norms.
7. **Multi-seed evaluation** on Cora; then PubMed; then Citeseer.
8. **PPG** if PPO plateaus.
9. **Paper draft** with the contribution centered on the homophily-aware reward and the β ablation as the empirical core.

---

## what to focus on

In priority order:

1. The β = 0 vs β > 0 ablation. This is the empirical core. Plan it from day one, not as an afterthought.
2. The pre-training homophily check on each dataset. Mandatory gate.
3. Clean reproduction of the GraphRARE-style accuracy-only baseline (β = 0) so the contrast is honest.
4. Reporting edge homophily of the resulting graph at every β setting, alongside accuracy. The homophily of the output graph is part of what we are claiming the reward controls.
