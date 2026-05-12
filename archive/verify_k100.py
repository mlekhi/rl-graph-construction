"""
Quick verification script: re-run K=100 k-sweep only.
Uses cached best params. Updates k_sweep.csv with confirmed K=100 row.
Run with: caffeinate -s python3 verify_k100.py
"""
import os, json, math, random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected
from tqdm import tqdm

# === paths ===
ROOT         = Path(__file__).parent
PREPROC_DIR  = ROOT / "preprocessed"
CACHE_DIR    = ROOT / "cache"
METRICS_DIR  = ROOT / "runs/GETS_GNN_ONLY/metrics"
BEST_PARAMS  = ROOT / "runs/GETS_GNN_ONLY/best_params/GraphSAGE_best_params.json"

# === config (must match notebook) ===
SEED          = 42
N_SEEDS_CHECK = 5
KNN_K_CHECK   = 100
GRAPH_MODE    = "inductive_directed"
KNN_METRIC    = "cosine"
MAX_EPOCHS    = 300
PATIENCE      = 30
OPTIMIZE_METRIC = "accuracy"
LABELS        = ["negative", "neutral", "positive"]

def set_seed(s):
    random.seed(s); np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)

set_seed(SEED)

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"device: {device}")

# === load data ===
X_tr = np.load(PREPROC_DIR / "X_train_bal.npy")
y_tr = np.load(PREPROC_DIR / "y_train_bal.npy")
X_sv = np.load(PREPROC_DIR / "X_sage_val.npy")
y_sv = np.load(PREPROC_DIR / "y_sage_val.npy")
X_pv = np.load(PREPROC_DIR / "X_policy_val.npy")
y_pv = np.load(PREPROC_DIR / "y_policy_val.npy")
X_te = np.load(PREPROC_DIR / "X_test.npy")
y_te = np.load(PREPROC_DIR / "y_test.npy")
meta = json.loads((PREPROC_DIR / "metadata.json").read_text())
print(f"shapes: train={X_tr.shape}, sage_val={X_sv.shape}, policy_val={X_pv.shape}, test={X_te.shape}")

best_p = json.loads(BEST_PARAMS.read_text())
print(f"best params: {best_p}")

# === graph builder ===
def build_graph(k):
    n_tr, n_sv, n_pv, n_te = len(X_tr), len(X_sv), len(X_pv), len(X_te)
    n_total = n_tr + n_sv + n_pv + n_te
    X_all = np.vstack([X_tr, X_sv, X_pv, X_te]).astype(np.float32)
    y_all = np.concatenate([y_tr, y_sv, y_pv, y_te]).astype(int)

    cache_path = CACHE_DIR / (
        f"edge_{GRAPH_MODE}_k{k}_{KNN_METRIC}_"
        f"n{n_total}_tr{n_tr}_sv{n_sv}_pv{n_pv}_te{n_te}_"
        f"pca{meta['pca_dim']}_{meta['feature_scaling']}_"
        f"bal{meta['balance_scope']}_{meta['balance_method']}.pt"
    )

    if cache_path.exists():
        print(f"[cache] loading edge_index for K={k}")
        edge_index = torch.load(cache_path, weights_only=False)
    else:
        print(f"[compute] building kNN graph K={k} ...")
        nn_tr = NearestNeighbors(n_neighbors=k+1, metric=KNN_METRIC, algorithm="brute").fit(X_tr)
        _, idx_tr = nn_tr.kneighbors(X_tr)
        src = np.repeat(np.arange(n_tr), k)
        dst = idx_tr[:, 1:].reshape(-1)
        edge_tr = to_undirected(torch.tensor(np.vstack([src, dst]), dtype=torch.long))

        def directed(X_tgt, offset):
            nn_ = NearestNeighbors(n_neighbors=k, metric=KNN_METRIC, algorithm="brute").fit(X_tr)
            _, idx = nn_.kneighbors(X_tgt)
            return torch.tensor(np.vstack([
                idx.reshape(-1),
                np.repeat(np.arange(X_tgt.shape[0]), k) + offset,
            ]), dtype=torch.long)

        edges = torch.cat([edge_tr,
                           directed(X_sv, n_tr),
                           directed(X_pv, n_tr + n_sv),
                           directed(X_te, n_tr + n_sv + n_pv)], dim=1)
        torch.save(edges, cache_path)
        edge_index = edges

    masks = {
        "train":      slice(0, n_tr),
        "sage_val":   slice(n_tr, n_tr+n_sv),
        "policy_val": slice(n_tr+n_sv, n_tr+n_sv+n_pv),
        "test":       slice(n_tr+n_sv+n_pv, None),
    }
    def make_mask(s):
        m = torch.zeros(n_total, dtype=torch.bool)
        m[s] = True
        return m

    return Data(
        x=torch.tensor(X_all, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y_all, dtype=torch.long),
        train_mask=make_mask(masks["train"]),
        sage_val_mask=make_mask(masks["sage_val"]),
        policy_val_mask=make_mask(masks["policy_val"]),
        test_mask=make_mask(masks["test"]),
    ).to(device)

# === model ===
class GraphSAGENet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, aggr, num_classes=3):
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

def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_f1": float(f1)}

def ci95(vals):
    vals = np.array(vals, dtype=float)
    if len(vals) <= 1: return 0.0
    from scipy.stats import t as tdist
    return float(tdist.ppf(0.975, df=len(vals)-1) * vals.std(ddof=1) / math.sqrt(len(vals)))

# === run K=100 ===
print(f"\n=== running K={KNN_K_CHECK} with {N_SEEDS_CHECK} seeds ===")
data_k = build_graph(KNN_K_CHECK)
in_dim = int(data_k.num_features)

seed_val_acc, seed_test_acc = [], []
seed_val_f1, seed_test_f1 = [], []

for i, s in enumerate([SEED + j for j in range(N_SEEDS_CHECK)]):
    print(f"  seed {i+1}/{N_SEEDS_CHECK} (seed={s})")
    set_seed(s)
    model = GraphSAGENet(in_dim,
                         hidden_dim=int(best_p["hidden_dim"]),
                         num_layers=int(best_p["num_layers"]),
                         dropout=float(best_p["dropout"]),
                         aggr=str(best_p["aggr"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(best_p["lr"]),
                                  weight_decay=float(best_p["weight_decay"]))
    ce = nn.CrossEntropyLoss()

    best_score, best_state, wait = -1e9, None, 0
    for epoch in tqdm(range(1, MAX_EPOCHS+1), desc=f"  K={KNN_K_CHECK} s={s}", leave=False):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = ce(model(data_k.x, data_k.edge_index)[data_k.train_mask], data_k.y[data_k.train_mask])
        loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data_k.x, data_k.edge_index)
            va_pred = logits[data_k.sage_val_mask].argmax(1).cpu().numpy()
            va_true = data_k.y[data_k.sage_val_mask].cpu().numpy()
            score = float(accuracy_score(va_true, va_pred))

        if score > best_score + 1e-12:
            best_score, best_state, wait = score, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE: break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data_k.x, data_k.edge_index)
        va = metrics(data_k.y[data_k.sage_val_mask].cpu().numpy(),
                     logits[data_k.sage_val_mask].argmax(1).cpu().numpy())
        te = metrics(data_k.y[data_k.test_mask].cpu().numpy(),
                     logits[data_k.test_mask].argmax(1).cpu().numpy())

    seed_val_acc.append(va["accuracy"]); seed_val_f1.append(va["macro_f1"])
    seed_test_acc.append(te["accuracy"]); seed_test_f1.append(te["macro_f1"])
    print(f"    val_acc={va['accuracy']:.4f}  test_acc={te['accuracy']:.4f}  test_f1={te['macro_f1']:.4f}")

    if device.type == "mps": torch.mps.empty_cache()

print(f"\n=== K=100 FINAL RESULT ===")
print(f"val_acc:  {np.mean(seed_val_acc):.4f} +/- {ci95(seed_val_acc):.4f}")
print(f"test_acc: {np.mean(seed_test_acc):.4f} +/- {ci95(seed_test_acc):.4f}")
print(f"test_f1:  {np.mean(seed_test_f1):.4f} +/- {ci95(seed_test_f1):.4f}")

# update k_sweep.csv
csv_path = METRICS_DIR / "k_sweep.csv"
df = pd.read_csv(csv_path)
df = df[df["knn_k"] != KNN_K_CHECK]  # remove old K=100 row if exists
new_row = {
    "knn_k": KNN_K_CHECK,
    "val_accuracy_mean": np.mean(seed_val_acc),  "val_accuracy_ci": ci95(seed_val_acc),
    "val_macro_f1_mean": np.mean(seed_val_f1),   "val_macro_f1_ci": ci95(seed_val_f1),
    "test_accuracy_mean": np.mean(seed_test_acc), "test_accuracy_ci": ci95(seed_test_acc),
    "test_macro_f1_mean": np.mean(seed_test_f1),  "test_macro_f1_ci": ci95(seed_test_f1),
}
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).sort_values("knn_k")
df.to_csv(csv_path, index=False)
print(f"\nupdated {csv_path}")
print(df[["knn_k","test_accuracy_mean","test_accuracy_ci"]].to_string(index=False))
