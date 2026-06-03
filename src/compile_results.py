"""
compile_results.py

Reads all results.json files and compiles into a single CSV.
This is the single source of truth for paper tables.

Run:
    python3 src/compile_results.py
    python3 src/compile_results.py --runs_dir runs/rl_cora_normalized2
"""

import argparse
import json
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent

parser = argparse.ArgumentParser()
parser.add_argument("--runs_dir", default=None, help="specific runs dir to scan (default: all)")
parser.add_argument("--out", default=None, help="output csv path")
args = parser.parse_args()

# find all results.json files
if args.runs_dir:
    search_dirs = [ROOT / args.runs_dir]
else:
    search_dirs = [ROOT / "runs"]

results_files = []
for d in search_dirs:
    results_files.extend(sorted(d.rglob("results.json")))

print(f"found {len(results_files)} results.json files")

rows = []
for path in results_files:
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        print(f"  skip {path}: {e}")
        continue

    # skip prenorm runs
    if "prenorm" in str(path):
        print(f"  skip (prenorm): {path}")
        continue

    row = {
        "path":               str(path.relative_to(ROOT)),
        "dataset":            d.get("dataset", ""),
        "split_seed":         d.get("split_seed", ""),
        "beta":               d.get("beta", ""),
        "n_episodes":         d.get("n_episodes", ""),
        "max_steps":          d.get("args", {}).get("max_steps", ""),
        "entropy_coef":       d.get("args", {}).get("entropy_coef", ""),
        "baseline_macro_f1":  round(d.get("baseline_macro_f1", 0), 4),
        "best_macro_f1":      round(d.get("best_macro_f1", 0), 4),
        "delta_macro_f1":     round(d.get("delta_macro_f1", 0), 4),
        "final_macro_f1":     round(d.get("final_macro_f1", 0), 4) if d.get("final_macro_f1") else "",
        "homophily_original": round(d["homophily_original"], 4) if d.get("homophily_original") else "",
        "homophily_knn_pool": round(d["homophily_knn_pool"], 4) if d.get("homophily_knn_pool") else "",
        "homophily_refined":  round(d["homophily_refined"], 4) if d.get("homophily_refined") else "",
        "delta_homophily":    round(d["homophily_refined"] - d["homophily_original"], 4) if d.get("homophily_refined") and d.get("homophily_original") else "",
        "wall_time_sec":      d.get("wall_time_sec", ""),
        "git_commit":         d.get("git_commit", ""),
    }
    rows.append(row)

if not rows:
    print("no results found")
    exit()

# sort by dataset, beta
rows.sort(key=lambda r: (r["dataset"], float(r["beta"]) if r["beta"] != "" else 0))

# write csv
out_path = ROOT / (args.out or "runs/results_summary.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(rows[0].keys())
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nsaved to {out_path}")
print(f"\n{'beta':>6} {'best_f1':>10} {'delta':>8} {'h_orig':>8} {'h_knn':>8} {'h_refined':>10} {'d_hom':>8}")
print("-" * 65)
for r in rows:
    print(f"{str(r['beta']):>6} {str(r['best_macro_f1']):>10} {str(r['delta_macro_f1']):>8} "
          f"{str(r['homophily_original']):>8} {str(r['homophily_knn_pool']):>8} "
          f"{str(r['homophily_refined']):>10} {str(r['delta_homophily']):>8}")
