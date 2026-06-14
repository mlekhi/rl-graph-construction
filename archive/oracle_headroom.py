"""
headroom check: what's the UPPER BOUND on test F1 from graph editing?
if even a label-cheating oracle can't beat baseline, there's no point fixing the
RL policy on this dataset -- the graph is already near-optimal for frozen SAGE.

two oracles:
  full   : remove every heterophilic edge + add every homophilic kNN candidate,
           using ALL true labels (incl. test) -> absolute ceiling (leaky)
  fair   : only touch edges where BOTH endpoints are non-test (train+val labels
           the method legitimately has) -> reachable headroom, measured on test
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from graph_env import GraphEnv
from sklearn.metrics import precision_recall_fscore_support

DATASET = sys.argv[1] if len(sys.argv) > 1 else "Cora"
SEED = 42
env = GraphEnv(dataset=DATASET, split_seed=SEED, beta=0.0, knn_k=10)
dev = env.device
y = env.y.cpu()
test_mask = env.test_mask.cpu()

def f1_test(edge_index):
    env.sage.eval()
    with torch.no_grad():
        logits = env.sage(env.x, edge_index)
        pred = logits[env.test_mask].argmax(1).cpu().numpy()
        true = env.y[env.test_mask].cpu().numpy()
    _, _, f1, _ = precision_recall_fscore_support(true, pred, average="macro", zero_division=0)
    return float(f1)

base = f1_test(env.original_edge_index.to(dev))
print(f"\n{DATASET} baseline test F1: {base:.4f}\n")

def build(edits_allowed):
    # start from original undirected edges
    ei = env.original_edge_index.cpu().numpy()
    edges = set(map(frozenset, zip(ei[0].tolist(), ei[1].tolist())))
    # remove heterophilic
    kept = set()
    for e in edges:
        i, j = tuple(e) if len(e) == 2 else (list(e)[0], list(e)[0])
        if not edits_allowed(i, j):
            kept.add(e); continue
        if y[i] == y[j]:
            kept.add(e)            # homophilic -> keep
        # else drop
    # add homophilic kNN candidates
    for i, neigh in env._knn_pool_original.items():
        for j in neigh:
            if not edits_allowed(i, j):
                continue
            if y[i] == y[j]:
                kept.add(frozenset([i, j]))
    src, dst = [], []
    for e in kept:
        t = tuple(e)
        a, b = (t[0], t[1]) if len(t) == 2 else (t[0], t[0])
        src += [a, b]; dst += [b, a]
    return torch.tensor([src, dst], dtype=torch.long, device=dev)

full_ei = build(lambda i, j: True)
fair_ei = build(lambda i, j: (not test_mask[i]) and (not test_mask[j]))
print(f"full oracle (uses test labels, leaky ceiling): test F1 {f1_test(full_ei):.4f}  ({f1_test(full_ei)-base:+.4f})")
print(f"fair oracle (train+val labels only):           test F1 {f1_test(fair_ei):.4f}  ({f1_test(fair_ei)-base:+.4f})")

# prediction-guided oracle: edit ALL edges using frozen SAGE's predictions (no true
# labels at all) -- this is exactly the information the RL policy has access to.
with torch.no_grad():
    preds = env.sage(env.x, env.original_edge_index.to(dev)).argmax(1).cpu()
yp = preds
def build_pred():
    ei = env.original_edge_index.cpu().numpy()
    kept = set()
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        if yp[a] == yp[b]:
            kept.add(frozenset([a, b]))
    for i, neigh in env._knn_pool_original.items():
        for j in neigh:
            if yp[i] == yp[j]:
                kept.add(frozenset([i, j]))
    src, dst = [], []
    for e in kept:
        t = tuple(e); a, b = (t[0], t[1]) if len(t) == 2 else (t[0], t[0])
        src += [a, b]; dst += [b, a]
    return torch.tensor([src, dst], dtype=torch.long, device=dev)
pred_ei = build_pred()
print(f"pred-guided oracle (SAGE preds, what policy can use): test F1 {f1_test(pred_ei):.4f}  ({f1_test(pred_ei)-base:+.4f})")
print(f"\n=> pred-guided gain is the realistic ceiling a (working) RL policy could target")
