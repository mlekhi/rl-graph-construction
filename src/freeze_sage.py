"""
freeze_sage.py

Trains GraphSAGE with best optuna params on a fixed split and saves the
checkpoint. This frozen model is loaded by GraphEnv during RL training.

Run:
    python3 freeze_sage.py --dataset Cora --split_seed 42
"""

import argparse
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv

# ============================================================
# config
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    default="Cora", choices=["Cora", "PubMed", "CiteSeer"])
parser.add_argument("--split_seed", type=int, default=42)
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data" / "planetoid"
PARAMS_DIR = ROOT / "runs" / f"baseline_{args.dataset.lower()}" / "best_params"
CKPT_DIR  = ROOT / "runs" / f"rl_{args.dataset.lower()}"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MAX_EPOCHS = 300
PATIENCE   = 30
TRAIN_FRAC = 0.6
VAL_FRAC   = 0.2

device = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"dataset={args.dataset}  split_seed={args.split_seed}  device={device}")


def set_seed(s):
    random.seed(s); np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(args.seed)


# ============================================================
# data + split
# ============================================================
raw = Planetoid(root=str(DATA_DIR), name=args.dataset)[0]
num_classes = int(raw.y.max()) + 1
y_np = raw.y.numpy()


def make_split(seed):
    rng = np.random.default_rng(seed)
    n = raw.num_nodes
    train_idx, sage_val_idx, policy_val_idx, test_idx = [], [], [], []
    for c in range(num_classes):
        idx = np.where(y_np == c)[0]
        rng.shuffle(idx)
        n_c  = len(idx)
        n_tr = max(1, int(round(n_c * TRAIN_FRAC)))
        n_te = max(1, int(round(n_c * VAL_FRAC)))
        n_val = n_c - n_tr - n_te
        n_sv  = max(1, n_val // 2)
        n_pv  = n_val - n_sv
        train_idx.extend(idx[:n_tr])
        sage_val_idx.extend(idx[n_tr:n_tr+n_sv])
        policy_val_idx.extend(idx[n_tr+n_sv:n_tr+n_sv+n_pv])
        test_idx.extend(idx[n_tr+n_sv+n_pv:])
    def mask(idxs):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.tensor(idxs, dtype=torch.long)] = True
        return m
    return mask(train_idx), mask(sage_val_idx), mask(policy_val_idx), mask(test_idx)


train_mask, sage_val_mask, policy_val_mask, test_mask = make_split(args.split_seed)
print(f"split: train={train_mask.sum()} sage_val={sage_val_mask.sum()} "
      f"policy_val={policy_val_mask.sum()} test={test_mask.sum()}")


# ============================================================
# model
# ============================================================
class GraphSAGENet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, aggr):
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
        """Return penultimate layer embeddings (before final conv)."""
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# ============================================================
# train
# ============================================================
params = json.loads((PARAMS_DIR / "GraphSAGE_best_params.json").read_text())
print(f"best params: {params}")

model = GraphSAGENet(
    in_dim=raw.num_node_features,
    hidden_dim=int(params["hidden_dim"]),
    num_layers=int(params["num_layers"]),
    dropout=float(params["dropout"]),
    aggr=str(params["aggr"]),
).to(device)

data = raw.to(device)
train_mask_d = train_mask.to(device)
sage_val_mask_d = sage_val_mask.to(device)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=float(params["lr"]),
                              weight_decay=float(params["weight_decay"]))
ce = nn.CrossEntropyLoss()

best_score, best_state, wait = -1e9, None, 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = ce(model(data.x, data.edge_index)[train_mask_d], data.y[train_mask_d])
    loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        va_pred = logits[sage_val_mask_d].argmax(1).cpu().numpy()
        va_true = data.y[sage_val_mask_d].cpu().numpy()
        score = float(accuracy_score(va_true, va_pred))

    if score > best_score + 1e-12:
        best_score, best_state, wait = score, deepcopy(model.state_dict()), 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print(f"early stop at epoch {epoch}")
            break

    if epoch % 50 == 0:
        print(f"epoch {epoch:03d} | val_acc={score:.4f} | best={best_score:.4f}")

model.load_state_dict(best_state)

# eval on all splits
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    for name, mask in [("sage_val", sage_val_mask_d),
                        ("policy_val", policy_val_mask.to(device)),
                        ("test", test_mask.to(device))]:
        pred = logits[mask].argmax(1).cpu().numpy()
        true = data.y[mask].cpu().numpy()
        acc = accuracy_score(true, pred)
        _, _, f1, _ = precision_recall_fscore_support(true, pred, average="macro", zero_division=0)
        print(f"{name}: acc={acc:.4f}  macro_f1={f1:.4f}")

# save checkpoint + metadata
ckpt_path = CKPT_DIR / f"frozen_sage_seed{args.split_seed}.pt"
torch.save({
    "model_state": best_state,
    "params": params,
    "split_seed": args.split_seed,
    "dataset": args.dataset,
    "num_classes": num_classes,
    "in_dim": raw.num_node_features,
    "val_acc": best_score,
}, ckpt_path)
print(f"\nfrozen sage saved to {ckpt_path}")

# save masks too
masks_path = CKPT_DIR / f"masks_seed{args.split_seed}.pt"
torch.save({
    "train_mask": train_mask,
    "sage_val_mask": sage_val_mask,
    "policy_val_mask": policy_val_mask,
    "test_mask": test_mask,
}, masks_path)
print(f"masks saved to {masks_path}")
