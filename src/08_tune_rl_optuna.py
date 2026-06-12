"""
08_tune_rl_optuna.py

Optuna search over PPO hyperparameters for the RL stage, run BEFORE the
multi-seed evaluation (09_multiseed_eval.py). Mirrors the protocol already
used for the GraphSAGE / MLP baselines in 01_baseline_planetoid.py:
tune once on the primary split, then evaluate the chosen configuration
across 10 seeds.

Design decisions (see CODE_REVIEW_EDITS.md):
  - Tuning happens at beta=0 by default. PPO hyperparameters (lr, clip,
    entropy, ...) must not be tuned at a beta>0 setting, otherwise the
    hyperparameter search itself could favor the proposed method over the
    accuracy-only baseline and the beta ablation would not be a fair
    comparison. The tuned PPO config is reused unchanged for every beta.
  - The objective is mean_last50_policy_val_f1 minus the baseline -- the
    honest learning signal. It is NOT best_macro_f1 (a max over episodes),
    which rewards noisy configurations rather than learning ones.
  - Each trial is a subprocess call to 04_train_rl.py with a reduced
    episode budget, so a trial sees exactly the code path of a real run.

Run:
    python3 src/08_tune_rl_optuna.py --dataset Cora --n_trials 30
    python3 src/08_tune_rl_optuna.py --dataset Cora --n_trials 30 --n_episodes 150
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.INFO)

ROOT = Path(__file__).parent.parent
SRC  = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    default="Cora", choices=["Cora", "PubMed", "CiteSeer"])
parser.add_argument("--split_seed", type=int, default=42, help="tuning split (multi-seed eval uses others)")
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--beta",       type=float, default=0.0,
                    help="beta used during tuning. keep 0 so PPO params are not tuned in favor of the contribution")
parser.add_argument("--n_trials",   type=int, default=30)
parser.add_argument("--n_episodes", type=int, default=150, help="reduced budget per trial")
parser.add_argument("--timeout_per_trial", type=int, default=3600, help="seconds before a trial is killed")
args = parser.parse_args()

RL_DIR = ROOT / "runs" / f"rl_{args.dataset.lower()}"
OUT_PARAMS = RL_DIR / "ppo_optuna_best.json"
TRIALS_CSV = RL_DIR / "ppo_optuna_trials.csv"
RL_DIR.mkdir(parents=True, exist_ok=True)

masks = RL_DIR / f"masks_seed{args.split_seed}.pt"
if not masks.exists():
    raise FileNotFoundError(
        f"{masks} not found. run 02_freeze_sage.py --dataset {args.dataset} "
        f"--split_seed {args.split_seed} first."
    )


def run_trial(trial_number: int, params: dict) -> float:
    """One trial = one short 04_train_rl.py run; objective read from results.json."""
    tag = f"optuna_t{trial_number:03d}"
    cmd = [
        sys.executable, str(SRC / "04_train_rl.py"),
        "--dataset", args.dataset,
        "--split_seed", str(args.split_seed),
        "--seed", str(args.seed),
        "--beta", str(args.beta),
        "--n_episodes", str(args.n_episodes),
        "--run_tag", tag,
        "--lr", str(params["lr"]),
        "--entropy_coef", str(params["entropy_coef"]),
        "--clip_eps", str(params["clip_eps"]),
        "--n_epochs", str(params["n_epochs"]),
        "--gamma", str(params["gamma"]),
        "--vf_coef", str(params["vf_coef"]),
        "--max_steps", str(params["max_steps"]),
    ]
    print(f"\n[trial {trial_number}] {params}")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=args.timeout_per_trial)
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"trial {trial_number} failed (rc={proc.returncode})")

    results_path = (RL_DIR / f"ppo_seed{args.split_seed}_beta{args.beta}_{tag}"
                    / "results.json")
    d = json.loads(results_path.read_text())
    objective = d["mean_last50_policy_val_f1"] - d["baseline_macro_f1"]
    print(f"[trial {trial_number}] mean_last50 - baseline = {objective:+.4f} "
          f"({time.time()-t0:.0f}s)")
    return objective


def objective(trial: optuna.Trial) -> float:
    params = {
        "lr":           trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "entropy_coef": trial.suggest_float("entropy_coef", 1e-3, 5e-2, log=True),
        "clip_eps":     trial.suggest_categorical("clip_eps", [0.1, 0.2, 0.3]),
        "n_epochs":     trial.suggest_int("n_epochs", 2, 8),
        "gamma":        trial.suggest_float("gamma", 0.95, 0.999),
        "vf_coef":      trial.suggest_float("vf_coef", 0.25, 1.0),
        "max_steps":    trial.suggest_categorical("max_steps", [50, 100]),
    }
    return run_trial(trial.number, params)


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=args.seed),
    study_name=f"graphhare_ppo_{args.dataset.lower()}",
)
study.optimize(objective, n_trials=args.n_trials, catch=(RuntimeError,
               subprocess.TimeoutExpired))

best = dict(study.best_params)
best_record = {
    "dataset": args.dataset,
    "tuned_at_beta": args.beta,
    "tuning_split_seed": args.split_seed,
    "n_trials": args.n_trials,
    "n_episodes_per_trial": args.n_episodes,
    "objective": "mean_last50_policy_val_f1 - baseline_macro_f1",
    "best_objective": study.best_value,
    "best_params": best,
}
OUT_PARAMS.write_text(json.dumps(best_record, indent=2))
study.trials_dataframe().to_csv(TRIALS_CSV, index=False)

print(f"\n{'='*55}")
print(f"  OPTUNA DONE | {args.dataset} | {args.n_trials} trials")
print(f"{'='*55}")
print(f"  best objective (mean_last50 - baseline): {study.best_value:+.4f}")
print(f"  best params: {best}")
print(f"  saved to {OUT_PARAMS}")
print(f"  trials log: {TRIALS_CSV}")
if study.best_value <= 0:
    print("\n  NOTE: best objective <= 0 means no configuration produced a "
          "durable policy_val improvement. Do not proceed to the multi-seed "
          "stage expecting positive results -- investigate learning first "
          "(reward signal strength, episode length, candidate quality).")
