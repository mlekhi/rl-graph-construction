"""
03_baseline_planetoid.py

GraphSAGE + MLP baselines on Cora and PubMed.
Follows GraphRARE split: 60/20/20 per class, 10 random splits.
Adds policy_val by halving the val split -> sage_val + policy_val.

Run:
    caffeinate -s python3 03_baseline_planetoid.py --dataset Cora
    caffeinate -s python3 03_baseline_planetoid.py --dataset PubMed
    caffeinate -s python3 03_baseline_planetoid.py --dataset Cora --skip_optuna
"""

import argparse
import json
import math
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================
# config
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",      default="Cora", choices=["Cora", "PubMed", "CiteSeer"])
parser.add_argument("--n_splits",     type=int, default=10,  help="number of random splits")
parser.add_argument("--val_frac",     type=float, default=0.2, help="val+test fraction each")
parser.add_argument("--n_trials",     type=int, default=50,  help="optuna trials")
parser.add_argument("--skip_optuna",  action="store_true",   help="use cached best params if available")
parser.add_argument("--seed",         type=int, default=42)
args = parser.parse_args()

DATASET      = args.dataset
N_SPLITS     = args.n_splits
TRAIN_FRAC   = 0.6
VAL_FRAC     = args.val_frac    # split 50/50 into sage_val + policy_val
TEST_FRAC    = args.val_frac
N_TRIALS     = args.n_trials
SKIP_OPTUNA  = args.skip_optuna
BASE_SEED    = args.seed

MAX_EPOCHS_TUNE  = 100
PATIENCE_TUNE    = 15
MAX_EPOCHS_FINAL = 300
PATIENCE_FINAL   = 30
OPTIMIZE_METRIC  = "accuracy"

ROOT = Path(__file__).parent
DATA_DIR    = ROOT / "data" / "planetoid"
RUN_DIR     = ROOT / "runs" / f"baseline_{DATASET.lower()}"
METRICS_DIR = RUN_DIR / "metrics"
PARAMS_DIR  = RUN_DIR / "best_params"
for d in [DATA_DIR, METRICS_DIR, PARAMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

device = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)
print(f"dataset={DATASET}  device={device}  splits={N_SPLITS}  optuna_trials={N_TRIALS}")


# ============================================================
# reproducibility
# ============================================================
def set_seed(s):
    random.seed(s); np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(BASE_SEED)


# ============================================================
# data loading + split generation
# ============================================================
raw = Planetoid(root=str(DATA_DIR), name=DATASET)[0]
print(f"\n{DATASET}: {raw.num_nodes} nodes, {raw.edge_index.size(1)} edges, "
      f"{raw.num_node_features} features, {int(raw.y.max())+1} classes")

num_classes = int(raw.y.max()) + 1
y_np = raw.y.numpy()


def make_split(seed):
    """
    Per-class stratified split: 60% train / 10% sage_val / 10% policy_val / 20% test.
    Returns boolean masks of length n_nodes.
    """
    rng = np.random.default_rng(seed)
    n = raw.num_nodes
    train_idx, sage_val_idx, policy_val_idx, test_idx = [], [], [], []

    for c in range(num_classes):
        idx = np.where(y_np == c)[0]
        rng.shuffle(idx)
        n_c = len(idx)
        n_tr  = max(1, int(round(n_c * TRAIN_FRAC)))
        n_te  = max(1, int(round(n_c * TEST_FRAC)))
        n_val = n_c - n_tr - n_te          # remaining -> split 50/50
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


# quick sanity check on split sizes
tr, sv, pv, te = make_split(BASE_SEED)
print(f"split sizes (seed={BASE_SEED}): train={tr.sum()}, sage_val={sv.sum()}, "
      f"policy_val={pv.sum()}, test={te.sum()}")


# ============================================================
# metrics
# ============================================================
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": float(acc), "macro_f1": float(f1)}

def ci95(vals):
    vals = np.array(vals, dtype=float)
    if len(vals) <= 1: return 0.0
    try:
        from scipy.stats import t as tdist
        tcrit = float(tdist.ppf(0.975, df=len(vals)-1))
    except Exception:
        tcrit = 1.96
    return tcrit * float(vals.std(ddof=1)) / math.sqrt(len(vals))

def maybe_clear():
    try:
        if device.type == "mps":  torch.mps.empty_cache()
        elif device.type == "cuda": torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================
# models
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


class MLPNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.dropout = float(dropout)
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x, _edge_index=None):
        return self.net(x)


# ============================================================
# train / eval
# ============================================================
def train_model(model, data, sage_val_mask, lr, weight_decay, max_epochs, patience):
    model = model.to(device)
    data  = data.to(device)
    sage_val_mask = sage_val_mask.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce  = nn.CrossEntropyLoss()

    best_score, best_state, wait = -1e9, None, 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(data.x, data.edge_index)
        loss = ce(logits[data.train_mask], data.y[data.train_mask])
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            va_pred = logits[sage_val_mask].argmax(1).cpu().numpy()
            va_true = data.y[sage_val_mask].cpu().numpy()
            score = float(accuracy_score(va_true, va_pred))

        if score > best_score + 1e-12:
            best_score, best_state, wait = score, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def eval_model(model, data, mask):
    model.eval()
    mask = mask.to(device)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        pred = logits[mask].argmax(1).cpu().numpy()
        true = data.y[mask].cpu().numpy()
    return compute_metrics(true, pred)


# ============================================================
# optuna
# ============================================================
SAGE_SPACE = {
    "hidden_dim":   {"type": "categorical", "choices": [64, 128, 256]},
    "num_layers":   {"type": "int",         "low": 2, "high": 5},
    "dropout":      {"type": "float",       "low": 0.2, "high": 0.6},
    "lr":           {"type": "float",       "low": 1e-4, "high": 1e-2, "log": True},
    "weight_decay": {"type": "float",       "low": 1e-6, "high": 1e-2, "log": True},
    "aggr":         {"type": "categorical", "choices": ["mean", "max"]},
}

MLP_SPACE = {
    "hidden_dim":   {"type": "categorical", "choices": [64, 128, 256]},
    "num_layers":   {"type": "int",         "low": 2, "high": 4},
    "dropout":      {"type": "float",       "low": 0.2, "high": 0.6},
    "lr":           {"type": "float",       "low": 1e-4, "high": 1e-2, "log": True},
    "weight_decay": {"type": "float",       "low": 1e-6, "high": 1e-2, "log": True},
}


def suggest(trial, space):
    p = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "categorical": p[name] = trial.suggest_categorical(name, spec["choices"])
        elif t == "int":       p[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "float":     p[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return p


def run_optuna(model_name, build_fn, space, data, sage_val_mask):
    params_path = PARAMS_DIR / f"{model_name}_best_params.json"
    if SKIP_OPTUNA and params_path.exists():
        print(f"[cache] loaded best params for {model_name}")
        return json.loads(params_path.read_text())

    print(f"\noptuna tuning {model_name} ({N_TRIALS} trials) ...")

    def objective(trial):
        set_seed(BASE_SEED)
        p = suggest(trial, space)
        m = build_fn(p).to(device)
        m = train_model(m, data, sage_val_mask, p["lr"], p["weight_decay"],
                        MAX_EPOCHS_TUNE, PATIENCE_TUNE)
        return eval_model(m, data, sage_val_mask)["accuracy"]

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=BASE_SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    best = study.best_params
    print(f"  best val acc: {study.best_value:.4f}  params: {best}")
    params_path.write_text(json.dumps(best, indent=2))
    return best


# ============================================================
# multi-split evaluation
# ============================================================
def build_sage(p):
    return GraphSAGENet(raw.num_node_features, int(p["hidden_dim"]),
                        int(p["num_layers"]), float(p["dropout"]), str(p["aggr"]))

def build_mlp(p):
    return MLPNet(raw.num_node_features, int(p["hidden_dim"]),
                  int(p["num_layers"]), float(p["dropout"]))


def run_baseline(model_name, build_fn, space):
    print(f"\n{'='*55}")
    print(f"  {model_name} on {DATASET}")
    print(f"{'='*55}")

    # tune on split 0
    set_seed(BASE_SEED)
    tr0, sv0, pv0, te0 = make_split(BASE_SEED)
    tune_data = raw.clone().to(device)
    tune_data.train_mask = tr0.to(device)
    best_p = run_optuna(model_name, build_fn, space, tune_data, sv0)

    # multi-split eval
    val_accs, test_accs, val_f1s, test_f1s = [], [], [], []

    for i in range(N_SPLITS):
        split_seed = BASE_SEED + i
        set_seed(split_seed)
        tr, sv, pv, te = make_split(split_seed)

        split_data = raw.clone().to(device)
        split_data.train_mask = tr.to(device)

        model = build_fn(best_p)
        model = train_model(model, split_data, sv,
                            best_p["lr"], best_p["weight_decay"],
                            MAX_EPOCHS_FINAL, PATIENCE_FINAL)

        mv = eval_model(model, split_data, sv)
        mt = eval_model(model, split_data, te)

        val_accs.append(mv["accuracy"]); val_f1s.append(mv["macro_f1"])
        test_accs.append(mt["accuracy"]); test_f1s.append(mt["macro_f1"])
        maybe_clear()

        print(f"  split {i+1:02d}/{N_SPLITS} | val_acc={mv['accuracy']:.4f}  "
              f"test_acc={mt['accuracy']:.4f}  test_f1={mt['macro_f1']:.4f}")

    print(f"\n  {model_name} SUMMARY ({N_SPLITS} splits):")
    print(f"    val_acc:  {np.mean(val_accs):.4f} +/- {ci95(val_accs):.4f}")
    print(f"    test_acc: {np.mean(test_accs):.4f} +/- {ci95(test_accs):.4f}")
    print(f"    test_f1:  {np.mean(test_f1s):.4f} +/- {ci95(test_f1s):.4f}")

    return {
        "model": model_name, "dataset": DATASET, "n_splits": N_SPLITS,
        "val_acc_mean":  np.mean(val_accs),  "val_acc_ci":  ci95(val_accs),
        "test_acc_mean": np.mean(test_accs), "test_acc_ci": ci95(test_accs),
        "test_f1_mean":  np.mean(test_f1s),  "test_f1_ci":  ci95(test_f1s),
        "best_params": best_p,
    }


# ============================================================
# main
# ============================================================
results = []
results.append(run_baseline("GraphSAGE", build_sage, SAGE_SPACE))
results.append(run_baseline("MLP",       build_mlp,  MLP_SPACE))

# save
rows = []
for r in results:
    row = {k: v for k, v in r.items() if k != "best_params"}
    rows.append(row)
df = pd.DataFrame(rows)
out = METRICS_DIR / "baseline_summary.csv"
df.to_csv(out, index=False)

print(f"\n{'='*55}")
print(f"  FINAL RESULTS -- {DATASET}")
print(f"{'='*55}")
print(df[["model","test_acc_mean","test_acc_ci","test_f1_mean","test_f1_ci"]].to_string(index=False))
print(f"\nsaved to {out}")

# also save best params per model
for r in results:
    p = PARAMS_DIR / f"{r['model']}_best_params.json"
    p.write_text(json.dumps(r["best_params"], indent=2))
    print(f"best params saved: {p}")
