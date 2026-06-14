"""
09_multiseed_eval.py

Multi-seed (default 10) evaluation of GraphHARE against the MLP and
GraphSAGE baselines, on identical splits, with mean +/- 95% CI on TEST.
This produces the headline table for the paper.

Per seed s (default 42..51, matching 01_baseline_planetoid.py's splits):
  1. 02_freeze_sage.py  --split_seed s   (if masks/checkpoint missing)
     -> frozen GraphSAGE on the original graph = the GraphSAGE baseline
  2. MLP baseline trained on the same split (params from
     runs/baseline_<ds>/best_params/MLP_best_params.json)
  3. 04_train_rl.py for each beta, using the Optuna-tuned PPO params
     (runs/rl_<ds>/ppo_optuna_best.json, produced by 08_tune_rl_optuna.py)
     -> test F1 of the best-selected refined graph

The comparison metric is TEST macro-F1 / accuracy, evaluated once per run
on the graph selected by policy_val. policy_val numbers are never reported
as results.

Run:
    python3 src/09_multiseed_eval.py --dataset Cora
    python3 src/09_multiseed_eval.py --dataset Cora --betas 0.0,0.5,1.0 --n_seeds 10
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.datasets import Planetoid

ROOT = Path(__file__).parent.parent
SRC  = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    default="Cora", choices=["Cora", "PubMed", "CiteSeer"])
parser.add_argument("--betas",      type=str, default="0.0,0.5,1.0",
                    help="comma-separated beta values to evaluate")
parser.add_argument("--n_seeds",    type=int, default=10)
parser.add_argument("--base_seed",  type=int, default=42,
                    help="seeds are base_seed..base_seed+n_seeds-1 (matches 01_baseline splits)")
parser.add_argument("--n_episodes", type=int, default=500)
parser.add_argument("--params_json", type=str, default=None,
                    help="tuned PPO params (default: runs/rl_<ds>/ppo_optuna_best.json)")
parser.add_argument("--rerun",      action="store_true", help="rerun even if results.json exists")
parser.add_argument("--run_tag",    type=str, default="ms", help="tag for the multiseed runs")
args = parser.parse_args()

BETAS = [float(b) for b in args.betas.split(",")]
SEEDS = list(range(args.base_seed, args.base_seed + args.n_seeds))

DS_L     = args.dataset.lower()
RL_DIR   = ROOT / "runs" / f"rl_{DS_L}"
OUT_DIR  = ROOT / "runs" / f"multiseed_{DS_L}"
DATA_DIR = ROOT / "data" / "planetoid"
MLP_PARAMS_PATH = ROOT / "runs" / f"baseline_{DS_L}" / "best_params" / "MLP_best_params.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

# ------------------------------------------------------------
# tuned PPO params (08_tune_rl_optuna.py output). Optuna BEFORE multi-seed.
# ------------------------------------------------------------
params_path = Path(args.params_json) if args.params_json else RL_DIR / "ppo_optuna_best.json"
if params_path.exists():
    tuned = json.loads(params_path.read_text())["best_params"]
    print(f"using Optuna-tuned PPO params from {params_path}: {tuned}")
else:
    tuned = {}
    print(f"WARNING: {params_path} not found -- falling back to 04_train_rl.py "
          f"defaults. Run 08_tune_rl_optuna.py first; hyperparameters should be "
          f"tuned before the multi-seed evaluation.")


def set_seed(s):
    random.seed(s); np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def ci95(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) <= 1:
        return 0.0
    try:
        from scipy.stats import t as tdist
        tcrit = float(tdist.ppf(0.975, df=len(vals) - 1))
    except Exception:
        tcrit = 1.96
    return tcrit * float(vals.std(ddof=1)) / math.sqrt(len(vals))


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        print(proc.stdout[-3000:])
        print(proc.stderr[-3000:])
        raise RuntimeError(f"command failed (rc={proc.returncode})")
    return proc


# ------------------------------------------------------------
# MLP baseline (same split, same early-stopping protocol as the sage)
# ------------------------------------------------------------
class MLPNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, num_classes):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp_baseline(raw, masks, params, seed, max_epochs=300, patience=30):
    """Train MLP on train_mask, early-stop on sage_val, evaluate on test."""
    set_seed(seed)
    num_classes = int(raw.y.max()) + 1
    model = MLPNet(raw.num_node_features, int(params["hidden_dim"]),
                   int(params["num_layers"]), float(params["dropout"]),
                   num_classes).to(device)
    x = raw.x.to(device); y = raw.y.to(device)
    tr = masks["train_mask"].to(device)
    sv = masks["sage_val_mask"].to(device)
    te = masks["test_mask"].to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]),
                            weight_decay=float(params["weight_decay"]))
    ce = nn.CrossEntropyLoss()
    best_score, best_state, wait = -1e9, None, 0

    for _ in range(max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = ce(model(x)[tr], y[tr])
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            score = float((model(x)[sv].argmax(1) == y[sv]).float().mean())
        if score > best_score + 1e-12:
            best_score, best_state, wait = score, deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x)[te].argmax(1).cpu().numpy()
    true = y[te].cpu().numpy()
    acc = accuracy_score(true, pred)
    _, _, f1, _ = precision_recall_fscore_support(true, pred, average="macro",
                                                  zero_division=0)
    return {"accuracy": float(acc), "macro_f1": float(f1)}


# ------------------------------------------------------------
# per-seed pipeline
# ------------------------------------------------------------
if not MLP_PARAMS_PATH.exists():
    raise FileNotFoundError(
        f"{MLP_PARAMS_PATH} not found. run 01_baseline_planetoid.py first "
        f"(it Optuna-tunes both baselines)."
    )
mlp_params = json.loads(MLP_PARAMS_PATH.read_text())
raw = Planetoid(root=str(DATA_DIR), name=args.dataset)[0]

per_seed = {}          # seed -> {"mlp": {...}, "sage": {...}, "hare": {beta: {...}}}

for seed in SEEDS:
    print(f"\n{'='*60}\n  SEED {seed}\n{'='*60}")
    per_seed[seed] = {"hare": {}}

    # 1. frozen sage + masks for this split
    masks_path = RL_DIR / f"masks_seed{seed}.pt"
    ckpt_path  = RL_DIR / f"frozen_sage_seed{seed}.pt"
    if not (masks_path.exists() and ckpt_path.exists()):
        run([sys.executable, SRC / "02_freeze_sage.py",
             "--dataset", args.dataset, "--split_seed", seed, "--seed", seed])
    masks = torch.load(masks_path, weights_only=False)

    # 2. MLP baseline on the identical split
    per_seed[seed]["mlp"] = train_mlp_baseline(raw, masks, mlp_params, seed)
    print(f"  MLP test: f1={per_seed[seed]['mlp']['macro_f1']:.4f} "
          f"acc={per_seed[seed]['mlp']['accuracy']:.4f}")

    # 3. GraphHARE per beta (tuned PPO params, fixed across betas)
    for beta in BETAS:
        run_dir = RL_DIR / f"ppo_seed{seed}_beta{beta}_{args.run_tag}"
        results_path = run_dir / "results.json"
        if args.rerun or not results_path.exists():
            cmd = [sys.executable, SRC / "04_train_rl.py",
                   "--dataset", args.dataset,
                   "--split_seed", seed, "--seed", seed,
                   "--beta", beta,
                   "--n_episodes", args.n_episodes,
                   "--run_tag", args.run_tag]
            for k in ["lr", "entropy_coef", "clip_eps", "n_epochs",
                      "gamma", "vf_coef", "max_steps"]:
                if k in tuned:
                    cmd += [f"--{k}", tuned[k]]
            run(cmd)
        d = json.loads(results_path.read_text())
        per_seed[seed]["hare"][beta] = d["test"]
        # the frozen sage's test score on the original graph IS the
        # GraphSAGE baseline for this split (same model, same protocol)
        per_seed[seed].setdefault("sage", d["test"]["original_graph"])
        print(f"  beta={beta}: test_f1 best_graph="
              f"{d['test']['best_graph']['macro_f1']:.4f} "
              f"(orig={d['test']['original_graph']['macro_f1']:.4f})")

# ------------------------------------------------------------
# aggregate
# ------------------------------------------------------------
def agg(vals):
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ci95": ci95(vals), "n": len(vals), "values": [float(v) for v in vals]}

summary = {"dataset": args.dataset, "seeds": SEEDS, "n_episodes": args.n_episodes,
           "tuned_ppo_params": tuned, "models": {}}

summary["models"]["MLP"] = {
    "test_f1":  agg([per_seed[s]["mlp"]["macro_f1"] for s in SEEDS]),
    "test_acc": agg([per_seed[s]["mlp"]["accuracy"] for s in SEEDS]),
}
summary["models"]["GraphSAGE"] = {
    "test_f1":  agg([per_seed[s]["sage"]["macro_f1"] for s in SEEDS]),
    "test_acc": agg([per_seed[s]["sage"]["accuracy"] for s in SEEDS]),
}
for beta in BETAS:
    f1s  = [per_seed[s]["hare"][beta]["best_graph"]["macro_f1"] for s in SEEDS]
    accs = [per_seed[s]["hare"][beta]["best_graph"]["accuracy"] for s in SEEDS]
    # paired per-seed delta vs GraphSAGE on the same split (stronger test
    # than comparing the two means, since split difficulty is shared)
    deltas = [per_seed[s]["hare"][beta]["best_graph"]["macro_f1"]
              - per_seed[s]["sage"]["macro_f1"] for s in SEEDS]
    summary["models"][f"GraphHARE(beta={beta})"] = {
        "test_f1":  agg(f1s),
        "test_acc": agg(accs),
        "paired_delta_f1_vs_sage": agg(deltas),
    }

(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

# csv + console table
import csv
with open(OUT_DIR / "summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "test_f1_mean", "test_f1_ci95", "test_acc_mean",
                "test_acc_ci95", "paired_delta_f1_vs_sage", "delta_ci95", "n_seeds"])
    for name, m in summary["models"].items():
        pd_ = m.get("paired_delta_f1_vs_sage", {})
        w.writerow([name,
                    f"{m['test_f1']['mean']:.4f}",  f"{m['test_f1']['ci95']:.4f}",
                    f"{m['test_acc']['mean']:.4f}", f"{m['test_acc']['ci95']:.4f}",
                    f"{pd_.get('mean', ''):.4f}" if pd_ else "",
                    f"{pd_.get('ci95', ''):.4f}" if pd_ else "",
                    m["test_f1"]["n"]])

print(f"\n{'='*70}")
print(f"  MULTI-SEED RESULTS | {args.dataset} | {len(SEEDS)} seeds | TEST set")
print(f"{'='*70}")
print(f"  {'model':<24} {'test_f1':>16} {'test_acc':>16} {'d_f1 vs SAGE':>14}")
print("  " + "-" * 68)
for name, m in summary["models"].items():
    pd_ = m.get("paired_delta_f1_vs_sage")
    d_str = f"{pd_['mean']:+.4f}±{pd_['ci95']:.4f}" if pd_ else "--"
    print(f"  {name:<24} {m['test_f1']['mean']:.4f}±{m['test_f1']['ci95']:.4f}"
          f"{'':>4} {m['test_acc']['mean']:.4f}±{m['test_acc']['ci95']:.4f}"
          f"{'':>4} {d_str:>14}")
print(f"\n  saved: {OUT_DIR / 'summary.json'}  /  {OUT_DIR / 'summary.csv'}")
print("  note: GraphHARE beats GraphSAGE only if the paired delta CI excludes 0.")
