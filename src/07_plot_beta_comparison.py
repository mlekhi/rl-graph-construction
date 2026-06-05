"""
07_plot_beta_comparison.py

Plots beta ablation curves for all 3 datasets side by side.

Run:
    python3 src/07_plot_beta_comparison.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'sans-serif'

ROOT = Path(__file__).parent.parent

datasets = [
    ("Cora",     "runs/rl_cora_normalized2"),
    ("CiteSeer", "runs/rl_citeseer_full"),
    ("PubMed",   "runs/rl_pubmed_results"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("GraphHARE β Ablation — All Datasets (seed=42, normalized reward)",
             fontsize=14, fontweight="bold")

for ax, (dataset, runs_dir) in zip(axes, datasets):
    runs_path = ROOT / runs_dir
    rows = []
    for path in sorted(runs_path.rglob("results.json")):
        if "prenorm" in str(path):
            continue
        d = json.loads(path.read_text())
        if "beta" not in d or "best_macro_f1" not in d:
            continue
        rows.append({
            "beta":             float(d["beta"]),
            "best_macro_f1":    float(d["best_macro_f1"]),
            "homophily_refined": float(d.get("homophily_refined", 0)),
            "baseline_macro_f1": float(d.get("baseline_macro_f1", 0)),
        })

    rows.sort(key=lambda r: r["beta"])
    betas    = [r["beta"] for r in rows]
    f1s      = [r["best_macro_f1"] for r in rows]
    homs     = [r["homophily_refined"] for r in rows]
    baseline = rows[0]["baseline_macro_f1"] if rows else 0

    # F1 curve
    ax.plot(betas, f1s, "o-", color="#2196F3", linewidth=2.5, markersize=7, label="Best macro-F1")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.5,
               label=f"Baseline ({baseline:.4f})")

    # mark peak
    peak_idx = f1s.index(max(f1s))
    ax.scatter([betas[peak_idx]], [f1s[peak_idx]], color="#F44336", zorder=5, s=100,
               label=f"Peak β={betas[peak_idx]}, F1={f1s[peak_idx]:.4f}")

    # auto-scale y
    f1_range = max(f1s) - min(f1s)
    ax.set_ylim(min(f1s) - max(f1_range * 0.5, 0.002),
                max(f1s) + max(f1_range * 0.5, 0.002))

    ax.set_title(dataset, fontsize=13, fontweight="bold")
    ax.set_xlabel("β", fontsize=12)
    ax.set_ylabel("Best macro-F1", fontsize=11, color="#2196F3")
    ax.tick_params(axis="y", labelcolor="#2196F3")
    ax.set_xticks(betas)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9, loc="lower center")

    # homophily on twin axis
    ax2 = ax.twinx()
    ax2.plot(betas, homs, "s--", color="#4CAF50", linewidth=1.5, markersize=5, alpha=0.7)
    ax2.set_ylabel("h_refined", fontsize=10, color="#4CAF50")
    ax2.tick_params(axis="y", labelcolor="#4CAF50")
    hom_range = max(homs) - min(homs)
    ax2.set_ylim(min(homs) - max(hom_range * 0.5, 0.001),
                 max(homs) + max(hom_range * 0.5, 0.001))

plt.tight_layout()
out_path = ROOT / "runs/beta_curve_all_datasets.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved to {out_path}")
plt.show()
