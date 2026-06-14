"""
stress-test the 'no prediction-reachable headroom' claim.
instead of blunt all-or-nothing editing, use confidence-thresholded selective
edits restricted to non-test edges (the set that gave the fair oracle +1pp on cora).
if some threshold recovers the gain with PREDICTIONS, the method is viable.
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from graph_env import GraphEnv
from sklearn.metrics import precision_recall_fscore_support

DATASET = sys.argv[1] if len(sys.argv) > 1 else "Cora"
env = GraphEnv(dataset=DATASET, split_seed=42, beta=0.0, knn_k=10)
dev = env.device
test_mask = env.test_mask.cpu()
y = env.y.cpu()

def f1_test(ei):
    with torch.no_grad():
        pred = env.sage(env.x, ei)[env.test_mask].argmax(1).cpu().numpy()
    true = env.y[env.test_mask].cpu().numpy()
    return float(precision_recall_fscore_support(true, pred, average="macro", zero_division=0)[2])

base = f1_test(env.original_edge_index.to(dev))
with torch.no_grad():
    logits = env.sage(env.x, env.original_edge_index.to(dev))
    prob = torch.softmax(logits, 1).cpu()
    pred = prob.argmax(1)
    conf = prob.max(1).values
print(f"\n{DATASET} baseline test F1 {base:.4f}")

def edit(use_true, thresh, nontest_only):
    lab = y if use_true else pred
    ei = env.original_edge_index.cpu().numpy()
    kept = set()
    def ok(i, j):
        if nontest_only and (test_mask[i] or test_mask[j]): return False
        if use_true: return True
        return (conf[i] > thresh) and (conf[j] > thresh)
    for a, b in zip(ei[0].tolist(), ei[1].tolist()):
        if not ok(a, b): kept.add(frozenset([a, b]))
        elif lab[a] == lab[b]: kept.add(frozenset([a, b]))
    for i, neigh in env._knn_pool_original.items():
        for j in neigh:
            if ok(i, j) and lab[i] == lab[j]: kept.add(frozenset([i, j]))
    s, d = [], []
    for e in kept:
        t = tuple(e); a, b = (t[0], t[1]) if len(t) == 2 else (t[0], t[0])
        s += [a, b]; d += [b, a]
    return torch.tensor([s, d], dtype=torch.long, device=dev)

print(f"fair oracle (TRUE labels, non-test): {f1_test(edit(True,0,True))-base:+.4f}")
print("pred-guided, non-test edges, confidence-thresholded:")
for th in [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]:
    g = f1_test(edit(False, th, True)) - base
    print(f"   conf>{th:.2f}: {g:+.4f}")
