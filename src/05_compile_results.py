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

    def _round(val):
        # 0.0 is a legitimate value -- only treat None/missing as absent
        return round(val, 4) if val is not None else ""

    test = d.get("test", {}) or {}
    row = {
        "path":               str(path.relative_to(ROOT)),
        "dataset":            d.get("dataset", ""),
        "split_seed":         d.get("split_seed", ""),
        "seed":               d.get("seed", ""),
        "beta":               d.get("beta", ""),
        "n_episodes":         d.get("n_episodes", ""),
        "max_steps":          d.get("args", {}).get("max_steps", ""),
        "entropy_coef":       d.get("args", {}).get("entropy_coef", ""),
        "baseline_macro_f1":  _round(d.get("baseline_macro_f1")),
        "mean_last50_pv_f1":  _round(d.get("mean_last50_policy_val_f1")),
        "best_macro_f1":      _round(d.get("best_macro_f1")),
        "delta_macro_f1":     _round(d.get("delta_macro_f1")),
        "final_macro_f1":     _round(d.get("final_macro_f1")),
        # test columns: the headline numbers (empty for legacy runs that
        # predate the leak/eval fixes -- those should not be reported anyway)
        "test_f1_original":   _round((test.get("original_graph") or {}).get("macro_f1")),
        "test_f1_best":       _round((test.get("best_graph") or {}).get("macro_f1")),
        "test_f1_final":      _round((test.get("final_graph") or {}).get("macro_f1")),
        "test_delta_f1":      _round(test.get("delta_f1_best_vs_original")),
        "homophily_original": _round(d.get("homophily_original")),
        "homophily_knn_pool": _round(d.get("homophily_knn_pool")),
        "homophily_refined":  _round(d.get("homophily_refined")),
        "delta_homophily":    round(d["homophily_refined"] - d["homophily_original"], 4)
                              if d.get("homophily_refined") is not None and d.get("homophily_original") is not None else "",
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
print(f"\n{'beta':>6} {'mean50_pv':>10} {'test_orig':>10} {'test_best':>10} {'test_d':>8} {'h_orig':>8} {'h_refined':>10}")
print("-" * 70)
for r in rows:
    print(f"{str(r['beta']):>6} {str(r['mean_last50_pv_f1']):>10} {str(r['test_f1_original']):>10} "
          f"{str(r['test_f1_best']):>10} {str(r['test_delta_f1']):>8} "
          f"{str(r['homophily_original']):>8} {str(r['homophily_refined']):>10}")
