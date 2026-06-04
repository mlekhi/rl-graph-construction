"""
plot_beta_curve.py

Plots the beta ablation curve: x=beta, y=best macro-F1.
Shows the inverted-U shape peaking at optimal beta.

Run:
    python3 src/plot_beta_curve.py --runs_dir runs/rl_cora_normalized2
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--runs_dir", default="runs/rl_cora_normalized2")
parser.add_argument("--dataset", default="Cora", help="dataset name for plot title and filename")
parser.add_argument("--out", default=None, help="output path (default: runs/beta_curve_{dataset}.png)")
args = parser.parse_args()

# collect results
runs_dir = ROOT / args.runs_dir
rows = []
for path in sorted(runs_dir.rglob("results.json")):
    if "prenorm" in str(path):
        continue
    d = json.loads(path.read_text())
    if "beta" not in d or "best_macro_f1" not in d:
        continue
    rows.append({
        "beta": float(d["beta"]),
        "best_macro_f1": float(d["best_macro_f1"]),
        "homophily_refined": float(d.get("homophily_refined", 0)),
    })

rows.sort(key=lambda r: r["beta"])
betas   = [r["beta"] for r in rows]
f1s     = [r["best_macro_f1"] for r in rows]
homs    = [r["homophily_refined"] for r in rows]
baseline = 0.8663

# ---- plot ----
fig, ax1 = plt.subplots(figsize=(8, 5))

# macro-F1 curve
ax1.plot(betas, f1s, "o-", color="#2196F3", linewidth=2.5, markersize=8, label="Best macro-F1")
ax1.axhline(baseline, color="gray", linestyle="--", linewidth=1.5, label=f"GraphSAGE baseline ({baseline})")

# mark peak
peak_idx = f1s.index(max(f1s))
ax1.scatter([betas[peak_idx]], [f1s[peak_idx]], color="#F44336", zorder=5, s=120,
            label=f"Peak: β={betas[peak_idx]}, F1={f1s[peak_idx]:.4f}")

ax1.set_xlabel("β (homophily reward weight)", fontsize=13)
ax1.set_ylabel("Best macro-F1", fontsize=13, color="#2196F3")
ax1.tick_params(axis="y", labelcolor="#2196F3")
ax1.set_xticks(betas)
ax1.set_ylim(0.875, 0.886)

# homophily on second axis
ax2 = ax1.twinx()
ax2.plot(betas, homs, "s--", color="#4CAF50", linewidth=1.5, markersize=6, alpha=0.7, label="h_refined")
ax2.set_ylabel("Refined graph homophily", fontsize=13, color="#4CAF50")
ax2.tick_params(axis="y", labelcolor="#4CAF50")
ax2.set_ylim(0.811, 0.817)

# legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=10)

plt.title(f"GraphHARE β Ablation — {args.dataset} (seed=42, normalized reward)", fontsize=13, fontweight="bold")
plt.tight_layout()

out_path = ROOT / (args.out or f"runs/beta_curve_{args.dataset.lower()}.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved to {out_path}")
plt.show()
