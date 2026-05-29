"""
homophily_diagnostic.py

Pre-training gate: compute edge homophily of the starting citation graph
and the kNN candidate pool for each dataset.

If the kNN pool's homophily is close to the random baseline (1/C),
graph-based refinement is unlikely to help -- flag and decide before
investing GPU time.

Run:
    python3 src/homophily_diagnostic.py
    python3 src/homophily_diagnostic.py --datasets Cora PubMed CiteSeer --knn_k 10
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.datasets import Planetoid

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "planetoid"

parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=["Cora", "PubMed", "CiteSeer"])
parser.add_argument("--knn_k",    type=int, default=10)
args = parser.parse_args()


def edge_homophily(edge_index, labels):
    """
    Edge homophily (Zhu et al. 2020):
        h = |{(i,j) in E : y_i == y_j}| / |E|

    Uses undirected edges (each counted once via src < dst).
    """
    src, dst = edge_index[0], edge_index[1]
    # keep one direction only
    mask = src < dst
    src, dst = src[mask], dst[mask]
    if len(src) == 0:
        return 0.0
    same = (labels[src] == labels[dst]).float().mean().item()
    return same


def build_knn_edges(x_np, k, existing_edge_set):
    """
    Build kNN-k edges from node features.
    Returns edge_index (2, E) and a set of frozensets for the pool.
    Only includes edges NOT already in the original graph.
    """
    n = x_np.shape[0]
    nn_model = NearestNeighbors(
        n_neighbors=k + 1, metric="cosine", algorithm="brute"
    ).fit(x_np)
    _, idx = nn_model.kneighbors(x_np)

    srcs, dsts = [], []
    for i in range(n):
        for j in idx[i, 1:]:  # skip self
            if frozenset([i, int(j)]) not in existing_edge_set:
                srcs.append(i)
                dsts.append(int(j))

    if not srcs:
        return torch.zeros(2, 0, dtype=torch.long), set()

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    edge_set   = set(frozenset([s, d]) for s, d in zip(srcs, dsts))
    return edge_index, edge_set


def run_diagnostic(dataset_name, knn_k):
    print(f"\n{'='*55}")
    print(f"  {dataset_name}")
    print(f"{'='*55}")

    data = Planetoid(root=str(DATA_DIR), name=dataset_name)[0]
    labels     = data.y
    edge_index = data.edge_index
    num_nodes  = data.num_nodes
    num_classes = int(labels.max()) + 1
    x_np       = data.x.numpy()

    # class distribution
    counts = np.bincount(labels.numpy(), minlength=num_classes)
    class_fracs = counts / counts.sum()
    random_baseline = float((class_fracs ** 2).sum())  # expected same-class under random
    uniform_baseline = 1.0 / num_classes

    print(f"  nodes: {num_nodes}  |  edges: {edge_index.size(1) // 2}  |  classes: {num_classes}")
    print(f"  class counts: {counts.tolist()}")
    print(f"  random baseline (uniform):   {uniform_baseline:.4f}  (= 1/{num_classes})")
    print(f"  random baseline (empirical): {random_baseline:.4f}  (accounts for class imbalance)")

    # ---- original graph homophily ----
    ei  = edge_index
    src = ei[0].numpy()
    existing_edge_set = set(
        frozenset([int(s), int(d)]) for s, d in zip(ei[0].tolist(), ei[1].tolist())
    )

    h_orig = edge_homophily(ei, labels)
    print(f"\n  original graph:")
    print(f"    edges:      {ei.size(1) // 2}")
    print(f"    homophily:  {h_orig:.4f}  "
          f"({'HIGH' if h_orig > 0.5 else 'LOW'} -- "
          f"{h_orig / uniform_baseline:.1f}x random baseline)")

    # ---- kNN pool homophily ----
    print(f"\n  building kNN-{knn_k} candidate pool...")
    knn_ei, knn_set = build_knn_edges(x_np, knn_k, existing_edge_set)

    if knn_ei.size(1) == 0:
        print(f"    kNN pool: empty (all candidates already in graph)")
        return

    h_knn = edge_homophily(knn_ei, labels)
    print(f"  kNN-{knn_k} pool:")
    print(f"    candidate edges: {knn_ei.size(1)}")
    print(f"    homophily:       {h_knn:.4f}  "
          f"({'HIGH' if h_knn > 0.5 else 'LOW'} -- "
          f"{h_knn / uniform_baseline:.1f}x random baseline)")

    # ---- gate verdict ----
    print(f"\n  gate verdict:")
    if h_knn < uniform_baseline * 1.5:
        print(f"  ⚠  FAIL -- kNN pool homophily ({h_knn:.4f}) is near random "
              f"({uniform_baseline:.4f}). adding these edges is unlikely to help.")
    else:
        print(f"  ✓  PASS -- kNN pool is {h_knn / uniform_baseline:.1f}x above random. "
              f"graph refinement should provide signal.")

    if h_orig > 0.5:
        print(f"  ✓  original graph is highly homophilic ({h_orig:.4f}). "
              f"graphsage aggregation should be effective.")
    else:
        print(f"  ⚠  original graph has low homophily ({h_orig:.4f}). "
              f"graphsage may struggle.")

    return {
        "dataset": dataset_name,
        "num_nodes": num_nodes,
        "num_classes": num_classes,
        "h_original": h_orig,
        "h_knn_pool": h_knn,
        "random_baseline_uniform": uniform_baseline,
        "random_baseline_empirical": random_baseline,
        "knn_pool_edges": knn_ei.size(1),
    }


# ============================================================
results = []
for ds in args.datasets:
    r = run_diagnostic(ds, args.knn_k)
    if r:
        results.append(r)

print(f"\n\n{'='*55}")
print(f"  SUMMARY (kNN-{args.knn_k})")
print(f"{'='*55}")
print(f"  {'dataset':<12} {'h_original':>12} {'h_knn_pool':>12} {'random_base':>12} {'ratio':>8}")
print(f"  {'-'*58}")
for r in results:
    ratio = r["h_knn_pool"] / r["random_baseline_uniform"]
    print(f"  {r['dataset']:<12} {r['h_original']:>12.4f} {r['h_knn_pool']:>12.4f} "
          f"{r['random_baseline_uniform']:>12.4f} {ratio:>7.1f}x")
