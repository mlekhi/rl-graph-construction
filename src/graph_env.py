"""
graph_env.py

GraphEnv: gym-style environment for RL graph construction.

The agent sequentially edits a citation graph (add/remove edges).
A frozen GraphSAGE scores the result on policy_val.
Reward: r_t = delta_macro_f1 + beta * delta_homophily  (GraphHARE)

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
        ema_alpha: float = 0.3,    # EMA smoothing for reward
        min_degree: int = 2,       # min edges per node (constraint)
        norm_warmup: int = 20,     # reward steps before normalization kicks in
        device: torch.device = None,
    ):
        self.dataset    = dataset
        self.split_seed = split_seed
        self.beta       = beta
        self.knn_k      = knn_k
        self.norm_warmup = norm_warmup
        self.max_steps  = max_steps
        self.reward_every = reward_every
        self.ema_alpha  = ema_alpha
        self.min_degree = min_degree
        self.device     = device or (
            torch.device("mps")  if torch.backends.mps.is_available()  else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )

        self._load_data()
        self._load_frozen_sage()
        self._build_knn_pool()
        self._compute_baseline()
        self._init_norm_stats()

        print(f"GraphEnv ready | {dataset} | {self.num_nodes} nodes | "
              f"{self.original_edge_index.size(1)} edges | "
              f"kNN pool: {sum(len(v) for v in self.knn_pool.values())} candidates | "
              f"baseline macro_f1={self.baseline_macro_f1:.4f} | "
              f"baseline homophily={self.baseline_homophily:.4f}")

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
        """F1 and homophily on original graph -- reward deltas are relative to these."""
        mf1, _ = self._eval_f1(self.original_edge_index.to(self.device))
        self.baseline_macro_f1  = mf1
        self.baseline_homophily = self._eval_homophily(self.original_edge_index)

    def _init_norm_stats(self):
        """Running variance estimates for reward normalization (Welford online algorithm)."""
        self._norm_n      = 0
        self._norm_f1_m2  = 0.0   # sum of squared deviations for delta_f1
        self._norm_hom_m2 = 0.0   # sum of squared deviations for delta_homophily
        self._norm_f1_mean  = 0.0
        self._norm_hom_mean = 0.0

    def _update_norm_stats(self, delta_f1, delta_hom):
        """Welford online update for running mean and variance."""
        self._norm_n += 1
        n = self._norm_n
        # delta_f1
        d1 = delta_f1 - self._norm_f1_mean
        self._norm_f1_mean += d1 / n
        self._norm_f1_m2   += d1 * (delta_f1 - self._norm_f1_mean)
        # delta_homophily
        d2 = delta_hom - self._norm_hom_mean
        self._norm_hom_mean += d2 / n
        self._norm_hom_m2   += d2 * (delta_hom - self._norm_hom_mean)

    def _normalize(self, delta_f1, delta_hom):
        """
        Normalize delta_f1 and delta_hom by their running std.
        During warmup (first norm_warmup steps) returns raw values.
        """
        if self._norm_n < self.norm_warmup:
            return delta_f1, delta_hom
        std_f1  = max((self._norm_f1_m2  / self._norm_n) ** 0.5, 1e-8)
        std_hom = max((self._norm_hom_m2 / self._norm_n) ** 0.5, 1e-8)
        return delta_f1 / std_f1, delta_hom / std_hom

    # ----------------------------------------------------------
    # gym interface
    # ----------------------------------------------------------
    def reset(self):
        """Start new episode from original graph."""
        self.current_edge_index = self.original_edge_index.clone().to(self.device)
        self.knn_pool = {k: list(v) for k, v in self._knn_pool_original.items()}
        self.step_count  = 0
        self.done        = False
        self.ema_reward  = 0.0
        self.episode_start_macro_f1  = self.baseline_macro_f1
        self.episode_start_homophily = self.baseline_homophily
        self._refresh_sage_cache()
        return self._get_node_states(), self._get_all_candidates()

    def step(self, action):
        """
        action: (node_i, node_j, op) where op in {'add', 'remove'}
        returns: (states, reward, done, info)
        """
        assert not self.done, "episode is done, call reset()"
        node_i, node_j, op = action

        # apply edit
        valid = self._apply_edit(int(node_i), int(node_j), op)
        self.step_count += 1

        # compute reward every reward_every steps
        reward = 0.0
        norm_f1_out = 0.0
        norm_hom_out = 0.0
        if self.step_count % self.reward_every == 0:
            mf1, _    = self._eval_f1(self.current_edge_index)
            homophily = self._eval_homophily(self.current_edge_index)
            delta_f1  = mf1      - self.baseline_macro_f1
            delta_hom = homophily - self.baseline_homophily
            # update running stats then normalize
            self._update_norm_stats(delta_f1, delta_hom)
            norm_f1, norm_hom = self._normalize(delta_f1, delta_hom)
            norm_f1_out, norm_hom_out = norm_f1, norm_hom
            raw_reward = norm_f1 + self.beta * norm_hom
            # EMA smoothing
            self.ema_reward = (
                self.ema_alpha * raw_reward + (1 - self.ema_alpha) * self.ema_reward
            )
            reward = self.ema_reward
            self._refresh_sage_cache()

        self.done = self.step_count >= self.max_steps

        info = {
            "step": self.step_count,
            "valid_action": valid,
            "num_edges": self.current_edge_index.size(1),
            "norm_delta_f1": norm_f1_out,
            "norm_delta_hom": norm_hom_out,
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
        """Degree, homophily, structural entropy per node."""
        ei = self.current_edge_index.cpu().numpy()
        n  = self.num_nodes
        preds = self.cached_preds.cpu().numpy()

        degree      = np.zeros(n, dtype=np.float32)
        homophily   = np.zeros(n, dtype=np.float32)
        entropy     = np.zeros(n, dtype=np.float32)

        # build adjacency list
        adj = [[] for _ in range(n)]
        for s, t in zip(ei[0], ei[1]):
            adj[s].append(t)

        for i in range(n):
            nbrs = adj[i]
            deg  = len(nbrs)
            degree[i] = deg
            if deg == 0:
                continue
            nbr_labels = preds[nbrs]
            # homophily: fraction of neighbors with same predicted class
            homophily[i] = float((nbr_labels == preds[i]).mean())
            # structural entropy: entropy of neighbor label distribution
            counts = np.bincount(nbr_labels, minlength=self.num_classes).astype(float)
            probs  = counts / counts.sum()
            probs  = probs[probs > 0]
            entropy[i] = float(-np.sum(probs * np.log(probs + 1e-12)))

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
        """Macro-F1 on policy_val."""
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

    def _eval_homophily(self, edge_index):
        """
        Edge homophily (Zhu et al. 2020) using ground-truth labels:
            h = |{(i,j) in E : y_i == y_j}| / |E|
        Counts each undirected edge once (src < dst).
        """
        src = edge_index[0].cpu()
        dst = edge_index[1].cpu()
        mask = src < dst
        src, dst = src[mask], dst[mask]
        if len(src) == 0:
            return 0.0
        labels = self.y.cpu()
        return float((labels[src] == labels[dst]).float().mean().item())

    # ----------------------------------------------------------
    # utils
    # ----------------------------------------------------------
    def get_current_metrics(self):
        """Current macro-F1 and edge homophily (for logging)."""
        macro_f1, _ = self._eval_f1(self.current_edge_index)
        homophily   = self._eval_homophily(self.current_edge_index)
        return macro_f1, homophily

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
