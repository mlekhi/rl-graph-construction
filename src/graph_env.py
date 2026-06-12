"""
graph_env.py

GraphEnv: gym-style environment for RL graph construction.

The agent sequentially edits a citation graph (add/remove edges).
A frozen GraphSAGE scores the result on policy_val.
Reward (GraphHARE):
    r_t = (delta_macro_f1 / s_f1) + beta * (delta_homophily / s_hom)
where the deltas are between CONSECUTIVE evaluation points (every
`reward_every` edits), and s_f1 / s_hom are fixed scales calibrated once
from a random-policy pre-pass so that beta is interpretable and the
reward scale is stationary across training.

Label-usage protocol (no test leakage):
  - The F1 term uses policy_val labels only (policy_val is the reward set).
  - The homophily REWARD term uses ground-truth labels only on edges whose
    BOTH endpoints are non-test nodes (train + sage_val + policy_val).
    Test-node labels never enter any training signal.
  - Full-graph homophily (all edges, ground truth) is computed only for
    REPORTING, never for reward.
  - Test-set F1 is evaluated once, after training, on the selected graph.

Usage:
    env = GraphEnv(dataset="Cora", split_seed=42, beta=0.5, knn_k=10)
    state, candidates = env.reset()
    for step in range(env.max_steps):
        action = policy(state, candidates)   # (node_i, node_j, op)
        state, reward, done, info = env.step(action)
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected


ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "planetoid"


# ============================================================
# frozen GraphSAGE (same arch as baseline)
# ============================================================
class GraphSAGENet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, aggr, num_classes):
        super().__init__()
        self.dropout = float(dropout)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim, aggr=aggr))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        self.convs.append(SAGEConv(hidden_dim, num_classes, aggr=aggr))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)

    def embed(self, x, edge_index):
        """Penultimate layer embeddings."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


# ============================================================
# GraphEnv
# ============================================================
class GraphEnv:
    def __init__(
        self,
        dataset: str = "Cora",
        split_seed: int = 42,
        beta: float = 0.5,         # homophily reward weight (beta=0 -> accuracy-only)
        knn_k: int = 10,           # kNN candidate pool size
        max_steps: int = 50,       # edits per episode
        reward_every: int = 5,     # compute reward every N steps
        min_degree: int = 2,       # min edges per node (constraint)
        calib_episodes: int = 5,   # random-policy episodes used to calibrate reward scales
        device: torch.device = None,
    ):
        self.dataset    = dataset
        self.split_seed = split_seed
        self.beta       = beta
        self.knn_k      = knn_k
        self.max_steps  = max_steps
        self.reward_every = reward_every
        self.min_degree = min_degree
        self.calib_episodes = calib_episodes
        self.device     = device or (
            torch.device("mps")  if torch.backends.mps.is_available()  else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )

        self._load_data()
        self._load_frozen_sage()
        self._build_knn_pool()
        self._compute_baseline()
        self._calibrate_reward_scales()

        print(f"GraphEnv ready | {dataset} | {self.num_nodes} nodes | "
              f"{self.original_edge_index.size(1)} edges | "
              f"kNN pool: {sum(len(v) for v in self.knn_pool.values())} candidates | "
              f"baseline macro_f1={self.baseline_macro_f1:.4f} | "
              f"baseline homophily(full)={self.baseline_homophily:.4f} | "
              f"baseline homophily(reward, non-test)={self.baseline_reward_homophily:.4f} | "
              f"reward scales: s_f1={self.f1_scale:.5f} s_hom={self.hom_scale:.5f}")

    # ----------------------------------------------------------
    # setup
    # ----------------------------------------------------------
    def _load_data(self):
        raw = Planetoid(root=str(DATA_DIR), name=self.dataset)[0]
        self.raw         = raw
        self.num_nodes   = raw.num_nodes
        self.num_classes = int(raw.y.max()) + 1
        self.x           = raw.x.to(self.device)
        self.y           = raw.y.to(self.device)

        # original graph edges (undirected, stored as set of frozensets for fast lookup)
        self.original_edge_index = raw.edge_index.clone()
        ei = raw.edge_index.numpy()
        self.original_edges_set = set(
            map(frozenset, zip(ei[0].tolist(), ei[1].tolist()))
        )

        # load masks
        masks_path = ROOT / "runs" / f"rl_{self.dataset.lower()}" / f"masks_seed{self.split_seed}.pt"
        if not masks_path.exists():
            raise FileNotFoundError(
                f"masks not found at {masks_path}. run freeze_sage.py first."
            )
        masks = torch.load(masks_path, weights_only=False)
        self.train_mask      = masks["train_mask"].to(self.device)
        self.sage_val_mask   = masks["sage_val_mask"].to(self.device)
        self.policy_val_mask = masks["policy_val_mask"].to(self.device)
        self.test_mask       = masks["test_mask"].to(self.device)
        # nodes whose ground-truth labels may appear in the TRAINING signal.
        # test labels must never reach the reward, hence the reward-homophily
        # term is restricted to edges between non-test nodes.
        self.non_test_mask   = ~self.test_mask


    def _load_frozen_sage(self):
        ckpt_path = ROOT / "runs" / f"rl_{self.dataset.lower()}" / f"frozen_sage_seed{self.split_seed}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"frozen sage not found at {ckpt_path}. run freeze_sage.py first."
            )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        p = ckpt["params"]
        self.sage = GraphSAGENet(
            in_dim=ckpt["in_dim"],
            hidden_dim=int(p["hidden_dim"]),
            num_layers=int(p["num_layers"]),
            dropout=float(p["dropout"]),
            aggr=str(p["aggr"]),
            num_classes=ckpt["num_classes"],
        ).to(self.device)
        self.sage.load_state_dict(ckpt["model_state"])
        self.sage.eval()
        for param in self.sage.parameters():
            param.requires_grad_(False)
        self.hidden_dim = int(p["hidden_dim"])
        print(f"frozen sage loaded (val_acc={ckpt['val_acc']:.4f})")

    def _build_knn_pool(self):
        """Build kNN candidate edges from node features (CPU, done once)."""
        print(f"building kNN-{self.knn_k} candidate pool...")
        x_np = self.x.cpu().numpy()
        nn_model = NearestNeighbors(
            n_neighbors=self.knn_k + 1, metric="cosine", algorithm="brute"
        ).fit(x_np)
        _, idx = nn_model.kneighbors(x_np)

        # knn_pool[i] = list of candidate neighbor indices (not already in original graph)
        self._knn_pool_original = {}
        for i in range(self.num_nodes):
            neighbors = idx[i, 1:]  # skip self
            candidates = [
                int(j) for j in neighbors
                if frozenset([i, j]) not in self.original_edges_set and i != j
            ]
            self._knn_pool_original[i] = candidates
        self.knn_pool = {k: list(v) for k, v in self._knn_pool_original.items()}

    def _compute_baseline(self):
        """Reference metrics on the original graph (episode starting point)."""
        mf1, _ = self._eval_f1(self.original_edge_index.to(self.device))
        self.baseline_macro_f1 = mf1
        # full-graph homophily: REPORTING ONLY (uses all ground-truth labels)
        self.baseline_homophily = self._eval_homophily(self.original_edge_index)
        # reward homophily: restricted to non-test edges, safe for the reward
        self.baseline_reward_homophily = self._eval_reward_homophily(self.original_edge_index)

    def _calibrate_reward_scales(self):
        """
        Estimate fixed scales for the two reward terms from a random-policy
        pre-pass, so that r = delta_f1/s_f1 + beta * delta_hom/s_hom puts both
        terms on comparable scale (beta=1 ~ equal weight) WITHOUT the
        non-stationarity of normalizing by running statistics of the learning
        policy's own rewards. Scales are frozen before training starts.
        """
        self.f1_scale, self.hom_scale = 1.0, 1.0   # raw deltas during calibration
        f1_deltas, hom_deltas = [], []

        for _ in range(self.calib_episodes):
            self.reset()
            attempts = 0
            while not self.done and attempts < self.max_steps * 10:
                attempts += 1
                node_i = self.sample_node_by_entropy()
                candidates = self.get_node_candidates(node_i)
                if not candidates:
                    continue
                action = candidates[np.random.randint(len(candidates))]
                _, _, _, info = self.step(action)
                if info["is_reward_step"]:
                    f1_deltas.append(info["raw_delta_f1"])
                    hom_deltas.append(info["raw_delta_hom"])

        def _scale(deltas, fallback):
            arr = np.asarray(deltas, dtype=np.float64)
            if arr.size == 0:
                return fallback
            std = float(arr.std())
            if std > 1e-8:
                return std
            mean_abs = float(np.abs(arr).mean())
            return mean_abs if mean_abs > 1e-8 else fallback

        # fallbacks: one prediction flip on policy_val / one edge flip in the graph
        n_pv    = int(self.policy_val_mask.sum().item())
        n_edges = max(self.original_edge_index.size(1) // 2, 1)
        self.f1_scale  = _scale(f1_deltas,  1.0 / max(n_pv, 1))
        self.hom_scale = _scale(hom_deltas, 1.0 / n_edges)
        self._calib_samples = len(f1_deltas)

    # ----------------------------------------------------------
    # gym interface
    # ----------------------------------------------------------
    def reset(self):
        """Start new episode from original graph."""
        self.current_edge_index = self.original_edge_index.clone().to(self.device)
        self.knn_pool = {k: list(v) for k, v in self._knn_pool_original.items()}
        self.step_count  = 0
        self.done        = False
        # consecutive-delta reward: track metrics at the previous evaluation point
        self._last_eval_f1  = self.baseline_macro_f1
        self._last_eval_hom = self.baseline_reward_homophily
        self._refresh_sage_cache()
        return self._get_node_states(), self._get_all_candidates()

    def step(self, action):
        """
        action: (node_i, node_j, op) where op in {'add', 'remove'}
        returns: (states, reward, done, info)
        """
        assert not self.done, "episode is done, call reset()"
        node_i, node_j, op = action

        # apply edit; refresh frozen-sage cache so the agent always sees the
        # state of the CURRENT graph (design doc: state updates between edits)
        valid = self._apply_edit(int(node_i), int(node_j), op)
        self.step_count += 1
        if valid:
            self._refresh_sage_cache()

        # compute reward every reward_every steps:
        # consecutive deltas, scaled by frozen calibration constants
        reward = 0.0
        scaled_f1_out, scaled_hom_out = 0.0, 0.0
        raw_f1_out, raw_hom_out = 0.0, 0.0
        is_reward_step = self.step_count % self.reward_every == 0
        if is_reward_step:
            mf1, _    = self._eval_f1(self.current_edge_index)
            homophily = self._eval_reward_homophily(self.current_edge_index)
            raw_f1_out  = mf1       - self._last_eval_f1
            raw_hom_out = homophily - self._last_eval_hom
            self._last_eval_f1, self._last_eval_hom = mf1, homophily
            scaled_f1_out  = raw_f1_out  / self.f1_scale
            scaled_hom_out = raw_hom_out / self.hom_scale
            reward = scaled_f1_out + self.beta * scaled_hom_out

        self.done = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "valid_action": valid,
            "num_edges": self.current_edge_index.size(1),
            "is_reward_step": is_reward_step,
            "raw_delta_f1": raw_f1_out,
            "raw_delta_hom": raw_hom_out,
            "scaled_delta_f1": scaled_f1_out,
            "scaled_delta_hom": scaled_hom_out,
        }
        return self._get_node_states(), reward, self.done, info

    # ----------------------------------------------------------
    # state + candidates
    # ----------------------------------------------------------
    def _refresh_sage_cache(self):
        """Run frozen sage forward pass, cache embeddings + logits."""
        with torch.no_grad():
            self.cached_embeddings = self.sage.embed(self.x, self.current_edge_index)
            self.cached_logits     = self.sage(self.x, self.current_edge_index)
            self.cached_probs      = F.softmax(self.cached_logits, dim=1)
            self.cached_preds      = self.cached_probs.argmax(dim=1)

        # precompute structural features per node
        self._compute_structural_features()

    def _compute_structural_features(self):
        """
        Degree, homophily, structural entropy per node (predicted labels).
        Vectorized -- this runs after every edit now that the state cache is
        refreshed per step, so the old per-node python loop would be too slow.
        """
        ei = self.current_edge_index.cpu().numpy()
        n  = self.num_nodes
        preds = self.cached_preds.cpu().numpy()
        src, dst = ei[0], ei[1]

        degree = np.bincount(src, minlength=n).astype(np.float32)
        safe_deg = np.maximum(degree, 1.0)

        # homophily: fraction of neighbors with same predicted class
        same = (preds[src] == preds[dst]).astype(np.float64)
        homophily = (np.bincount(src, weights=same, minlength=n) / safe_deg).astype(np.float32)

        # structural entropy: entropy of the neighbor predicted-label distribution
        counts = np.zeros((n, self.num_classes), dtype=np.float64)
        np.add.at(counts, (src, preds[dst]), 1.0)
        row_sums = np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        probs = counts / row_sums
        entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
        entropy = entropy.astype(np.float32)

        zero_deg = degree == 0
        homophily[zero_deg] = 0.0
        entropy[zero_deg]   = 0.0

        self.struct_degree    = torch.tensor(degree,    device=self.device)
        self.struct_homophily = torch.tensor(homophily, device=self.device)
        self.struct_entropy   = torch.tensor(entropy,   device=self.device)

    def get_node_state(self, node_i):
        """
        State for a single node:
        [h_i (hidden_dim) || logits_i (num_classes) || degree || homophily || entropy]
        """
        return torch.cat([
            self.cached_embeddings[node_i],
            self.cached_probs[node_i],
            self.struct_degree[node_i].unsqueeze(0),
            self.struct_homophily[node_i].unsqueeze(0),
            self.struct_entropy[node_i].unsqueeze(0),
        ])  # shape: (hidden_dim + num_classes + 3,)

    def _get_node_states(self):
        """State matrix for all nodes (n_nodes, state_dim)."""
        return torch.cat([
            self.cached_embeddings,
            self.cached_probs,
            self.struct_degree.unsqueeze(1),
            self.struct_homophily.unsqueeze(1),
            self.struct_entropy.unsqueeze(1),
        ], dim=1)

    @property
    def state_dim(self):
        return self.hidden_dim + self.num_classes + 3

    def get_edge_states_batch(self, candidates):
        """
        Vectorized edge state computation for a list of (node_i, node_j, op) tuples.
        Returns (N, edge_state_dim) tensor. Much faster than calling get_edge_state in a loop.
        """
        if not candidates:
            return torch.zeros(0, self.edge_state_dim, device=self.device)

        node_states = self._get_node_states()  # (n_nodes, state_dim)
        h_norm = F.normalize(self.cached_embeddings, dim=1)  # (n_nodes, hidden_dim)

        idx_i  = torch.tensor([c[0] for c in candidates], dtype=torch.long, device=self.device)
        idx_j  = torch.tensor([c[1] for c in candidates], dtype=torch.long, device=self.device)
        ops    = torch.tensor([1.0 if c[2] == "add" else 0.0 for c in candidates],
                              device=self.device).unsqueeze(1)

        s_i      = node_states[idx_i]                          # (N, state_dim)
        s_j      = node_states[idx_j]                          # (N, state_dim)
        cos_sim  = (h_norm[idx_i] * h_norm[idx_j]).sum(dim=1, keepdim=True)  # (N, 1)
        in_orig  = torch.tensor(
            [float(frozenset([c[0], c[1]]) in self.original_edges_set) for c in candidates],
            device=self.device
        ).unsqueeze(1)                                          # (N, 1)

        return torch.cat([s_i, s_j, cos_sim, in_orig, ops], dim=1)  # (N, edge_state_dim)

    def get_edge_state(self, node_i, node_j, op):
        """
        Edge-level state for policy input:
        [s_i || s_j || cosine_sim || in_original_graph || op_is_add]
        """
        s_i = self.get_node_state(node_i)
        s_j = self.get_node_state(node_j)
        h_i = F.normalize(self.cached_embeddings[node_i], dim=0)
        h_j = F.normalize(self.cached_embeddings[node_j], dim=0)
        cos_sim = (h_i * h_j).sum().unsqueeze(0)
        in_orig = torch.tensor(
            [float(frozenset([node_i, node_j]) in self.original_edges_set)],
            device=self.device
        )
        op_flag = torch.tensor([1.0 if op == "add" else 0.0], device=self.device)
        return torch.cat([s_i, s_j, cos_sim, in_orig, op_flag])

    @property
    def edge_state_dim(self):
        return 2 * self.state_dim + 3

    def get_node_candidates(self, node_i):
        """
        Returns candidates for a single node: add from kNN pool + remove from current edges.
        ~10-25 candidates per node -- tractable for policy evaluation.
        """
        ei  = self.current_edge_index.cpu().numpy()
        deg = int((ei[0] == node_i).sum())

        candidates = []
        # add: from kNN pool
        for j in self.knn_pool[node_i]:
            candidates.append((node_i, j, "add"))
        # remove: current edges (only if degree > min_degree)
        if deg > self.min_degree:
            nbrs = ei[1][ei[0] == node_i].tolist()
            for j in nbrs:
                candidates.append((node_i, j, "remove"))
        return candidates

    def sample_node_by_entropy(self):
        """
        Sample a node weighted by structural entropy -- high entropy nodes
        are most uncertain and most worth editing.
        """
        entropy = self.struct_entropy.cpu().numpy()
        entropy = entropy + 1e-6  # avoid all-zero
        probs   = entropy / entropy.sum()
        return int(np.random.choice(self.num_nodes, p=probs))

    def _get_all_candidates(self):
        """
        Returns list of (node_i, node_j, op) candidate actions.
        add: from kNN pool
        remove: from current edges (respecting min_degree)
        """
        candidates = []
        ei = self.current_edge_index.cpu().numpy()
        degree = np.bincount(ei[0], minlength=self.num_nodes)

        for i in range(self.num_nodes):
            # add candidates
            for j in self.knn_pool[i]:
                candidates.append((i, j, "add"))
            # remove candidates (only if degree > min_degree)
            if degree[i] > self.min_degree:
                nbrs = ei[1][ei[0] == i].tolist()
                for j in nbrs:
                    candidates.append((i, j, "remove"))

        return candidates

    # ----------------------------------------------------------
    # graph editing
    # ----------------------------------------------------------
    def _apply_edit(self, node_i, node_j, op):
        """Apply add/remove to current_edge_index. Returns True if valid."""
        ei = self.current_edge_index
        if op == "add":
            # add undirected edge
            new_edges = torch.tensor(
                [[node_i, node_j], [node_j, node_i]], dtype=torch.long, device=self.device
            ).T
            self.current_edge_index = torch.cat([ei, new_edges], dim=1)
            # remove from kNN pool to avoid duplicates
            if node_j in self.knn_pool.get(node_i, []):
                self.knn_pool[node_i].remove(node_j)
            if node_i in self.knn_pool.get(node_j, []):
                self.knn_pool[node_j].remove(node_i)
            return True

        elif op == "remove":
            src, dst = ei[0], ei[1]
            # check degree constraint
            degree_i = (src == node_i).sum().item()
            degree_j = (src == node_j).sum().item()
            if degree_i <= self.min_degree or degree_j <= self.min_degree:
                return False  # would violate min degree
            # remove both directions
            keep = ~(
                ((src == node_i) & (dst == node_j)) |
                ((src == node_j) & (dst == node_i))
            )
            self.current_edge_index = ei[:, keep]
            return True

        return False

    # ----------------------------------------------------------
    # reward computation
    # ----------------------------------------------------------
    def _eval_f1(self, edge_index):
        """Macro-F1 on policy_val (the designated reward set)."""
        self.sage.eval()
        with torch.no_grad():
            logits = self.sage(self.x, edge_index)
            pred = logits[self.policy_val_mask].argmax(1).cpu().numpy()
            true = self.y[self.policy_val_mask].cpu().numpy()

        _, _, f1_per_class, _ = precision_recall_fscore_support(
            true, pred, average=None, labels=list(range(self.num_classes)), zero_division=0
        )
        macro_f1 = float(f1_per_class.mean())
        return macro_f1, f1_per_class

    def _edge_homophily(self, edge_index, node_mask=None):
        """
        Edge homophily (Zhu et al. 2020): h = |{(i,j) in E : y_i == y_j}| / |E|.
        Counts each undirected edge once (src < dst). If node_mask is given,
        only edges with BOTH endpoints inside the mask are counted.
        """
        src = edge_index[0].cpu()
        dst = edge_index[1].cpu()
        keep = src < dst
        if node_mask is not None:
            nm = node_mask.cpu()
            keep = keep & nm[src] & nm[dst]
        src, dst = src[keep], dst[keep]
        if len(src) == 0:
            return 0.0
        labels = self.y.cpu()
        return float((labels[src] == labels[dst]).float().mean().item())

    def _eval_homophily(self, edge_index):
        """Full-graph homophily, all ground-truth labels. REPORTING ONLY."""
        return self._edge_homophily(edge_index, node_mask=None)

    def _eval_reward_homophily(self, edge_index):
        """
        Homophily restricted to edges between non-test nodes. This is the
        version that may enter the reward: test labels never touch it.
        """
        return self._edge_homophily(edge_index, node_mask=self.non_test_mask)

    # ----------------------------------------------------------
    # post-training evaluation (reporting; test used ONLY here)
    # ----------------------------------------------------------
    def evaluate_graph(self, edge_index, split="test"):
        """
        Accuracy + macro-F1 of the frozen sage on a given graph and split.
        `split` in {"test", "policy_val", "sage_val", "train"}.
        Call with split="test" only AFTER training, on the selected graph.
        """
        mask = {
            "test": self.test_mask,
            "policy_val": self.policy_val_mask,
            "sage_val": self.sage_val_mask,
            "train": self.train_mask,
        }[split]
        self.sage.eval()
        with torch.no_grad():
            logits = self.sage(self.x, edge_index.to(self.device))
            pred = logits[mask].argmax(1).cpu().numpy()
            true = self.y[mask].cpu().numpy()
        acc = float((pred == true).mean())
        _, _, f1, _ = precision_recall_fscore_support(
            true, pred, average="macro", zero_division=0
        )
        return {"accuracy": acc, "macro_f1": float(f1)}

    def get_graph_copy(self):
        """CPU copy of the current edge_index (for saving the refined graph)."""
        return self.current_edge_index.detach().cpu().clone()

    # ----------------------------------------------------------
    # utils
    # ----------------------------------------------------------
    def get_current_metrics(self):
        """Current policy_val macro-F1 and full-graph homophily (for logging)."""
        macro_f1, _ = self._eval_f1(self.current_edge_index)
        homophily   = self._eval_homophily(self.current_edge_index)
        return macro_f1, homophily

    def get_current_reward_homophily(self):
        """Current non-test-edge homophily (the quantity the reward sees)."""
        return self._eval_reward_homophily(self.current_edge_index)

    def _eval_homophily_from_pool(self):
        """
        Edge homophily of the kNN candidate pool using ground-truth labels.
        Computed once after pool is built.
        """
        srcs, dsts = [], []
        for i, neighbors in self.knn_pool.items():
            for j in neighbors:
                if i < j:
                    srcs.append(i)
                    dsts.append(j)
        if not srcs:
            return 0.0
        src = torch.tensor(srcs)
        dst = torch.tensor(dsts)
        labels = self.y.cpu()
        return float((labels[src] == labels[dst]).float().mean().item())
