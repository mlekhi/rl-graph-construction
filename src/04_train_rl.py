"""
train_rl.py

PPO training loop for RL graph construction.

Evaluation protocol (no selection on noise, no test leakage):
  - During training, only policy_val is ever evaluated.
  - The refined graph with the best episode-end policy_val F1 is SAVED
    (selection happens on policy_val, the designated model-selection set).
  - After training, the frozen sage is evaluated on TEST exactly once per
    graph of interest: original graph, best-selected graph, final graph.
  - The honest learning signal is mean_last50_policy_val_f1 (not the
    max over episodes, which inflates with episode count under noise).

Run:
    python3 src/04_train_rl.py --dataset Cora --split_seed 42
    python3 src/04_train_rl.py --dataset Cora --split_seed 42 --smoke_test  # 20 episodes
"""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import sys
sys.path.insert(0, str(Path(__file__).parent))
from graph_env import GraphEnv
from policy_net import PolicyNet

# ============================================================
# config
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config",       default=None,   help="path to yaml config file (overrides defaults, CLI overrides config)")
parser.add_argument("--dataset",      default="Cora", choices=["Cora", "PubMed", "CiteSeer"])
parser.add_argument("--split_seed",   type=int, default=42)
parser.add_argument("--n_episodes",   type=int, default=500)
parser.add_argument("--max_steps",    type=int, default=50)
parser.add_argument("--reward_every", type=int, default=5)
parser.add_argument("--beta",         type=float, default=0.5,  help="homophily reward weight (0=accuracy-only, ignored if --learnable_beta)")
parser.add_argument("--learnable_beta", action="store_true",   help="make beta a learnable network parameter")
parser.add_argument("--run_tag",      type=str, default="",    help="optional tag appended to run name and output dir (e.g. knnfix)")
parser.add_argument("--knn_k",        type=int, default=10)
parser.add_argument("--lr",           type=float, default=3e-4)
parser.add_argument("--gamma",        type=float, default=0.99, help="discount factor")
parser.add_argument("--clip_eps",     type=float, default=0.2,  help="PPO clip ratio")
parser.add_argument("--vf_coef",      type=float, default=0.5)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--n_epochs",     type=int, default=4,      help="PPO update epochs per episode")
parser.add_argument("--dropout",      type=float, default=0.0,  help="policy dropout; keep 0 for PPO (train/eval mismatch corrupts the importance ratio)")
parser.add_argument("--calib_episodes", type=int, default=5,    help="random-policy episodes for reward scale calibration")
parser.add_argument("--smoke_test",   action="store_true",      help="run 20 episodes only")
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--wandb",        action="store_true")
args = parser.parse_args()

# load yaml config and apply as defaults (CLI args take precedence)
if args.config:
    if not HAS_YAML:
        raise ImportError("pyyaml not installed. run: pip install pyyaml")
    cfg = yaml.safe_load(Path(args.config).read_text())
    # only set values not explicitly passed via CLI
    cli_set = {a.dest for a in parser._actions if a.option_strings}
    import sys
    cli_explicitly_set = set()
    for i, a in enumerate(sys.argv[1:]):
        if a.startswith("--"):
            cli_explicitly_set.add(a.lstrip("--").split("=")[0].replace("-", "_"))
    for k, v in cfg.items():
        if k not in cli_explicitly_set and hasattr(args, k):
            setattr(args, k, v)

if args.smoke_test:
    args.n_episodes = 20

ROOT    = Path(__file__).parent.parent
tag_suffix = f"_{args.run_tag}" if args.run_tag else ""
if args.learnable_beta:
    OUT_DIR = ROOT / "runs" / f"rl_{args.dataset.lower()}" / f"ppo_seed{args.split_seed}_learnable_beta{tag_suffix}"
else:
    OUT_DIR = ROOT / "runs" / f"rl_{args.dataset.lower()}" / f"ppo_seed{args.split_seed}_beta{args.beta}{tag_suffix}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)


def set_seed(s):
    random.seed(s); np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(args.seed)

# git commit hash for reproducibility
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    git_commit = "unknown"

wall_start = time.time()

# ============================================================
# wandb (optional)
# ============================================================
if args.wandb:
    import wandb
    run_name = (f"{args.dataset}-learnablebeta-seed{args.split_seed}{tag_suffix}"
                if args.learnable_beta else
                f"{args.dataset}-beta{args.beta}-seed{args.split_seed}{tag_suffix}")
    wandb.init(
        project="graphhare",
        name=run_name,
        config=vars(args),
        tags=[args.dataset, "learnable_beta" if args.learnable_beta else f"beta{args.beta}"],
    )
    log = wandb.log
else:
    log = lambda d, **kw: None

# ============================================================
# setup
# ============================================================
env = GraphEnv(
    dataset=args.dataset,
    split_seed=args.split_seed,
    beta=args.beta,
    knn_k=args.knn_k,
    max_steps=args.max_steps,
    reward_every=args.reward_every,
    calib_episodes=args.calib_episodes,
    device=device,
)

policy = PolicyNet(
    edge_state_dim=env.edge_state_dim,
    hidden_dims=[256, 128],
    dropout=args.dropout,
    learnable_beta=args.learnable_beta,
    beta_init=args.beta,
).to(device)

if args.learnable_beta:
    print("WARNING: --learnable_beta is experimental. Beta receives gradients "
          "only through the value loss (advantages are detached), so it tends "
          "to shrink reward variance rather than optimize performance. Do not "
          "report learnable-beta results without validating against the fixed-"
          "beta sweep.")

# kNN pool homophily (computed once from pool edges + ground truth labels)
knn_pool_homophily = env._eval_homophily_from_pool()

optimizer = AdamW(policy.parameters(), lr=args.lr)

print(f"\nPPO training | {args.dataset} | {args.n_episodes} episodes | "
      f"max_steps={args.max_steps} | device={device}")
print(f"policy params: {sum(p.numel() for p in policy.parameters()):,}")
if args.learnable_beta:
    print(f"learnable beta: enabled (init={args.beta})")

# ============================================================
# rollout buffer
# ============================================================
class RolloutBuffer:
    """
    Stores one episode. Per step we keep the full candidate set so we can
    recompute log_probs from the same categorical distribution during update.

    For learnable beta: stores scaled (delta_f1, delta_hom) separately so
    rewards can be recomputed with the current beta during the PPO update,
    allowing beta to receive gradients through the value loss.
    """
    def __init__(self):
        self.all_states    = []   # list of (N_cands, edge_state_dim) tensors
        self.action_idxs   = []   # int index into all_states[t]
        self.log_probs     = []   # scalar, log_prob of chosen action
        self.rewards       = []   # float (combined reward, for fixed beta)
        self.scaled_delta_f1s  = []  # scaled delta_f1 per reward step
        self.scaled_delta_homs = []  # scaled delta_hom per reward step
        self.values        = []   # scalar critic estimate

    def clear(self):
        self.__init__()

    def add(self, all_states, action_idx, log_prob, reward, value,
            scaled_delta_f1=0.0, scaled_delta_hom=0.0):
        self.all_states.append(all_states)
        self.action_idxs.append(action_idx)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.scaled_delta_f1s.append(scaled_delta_f1)
        self.scaled_delta_homs.append(scaled_delta_hom)
        self.values.append(value)

    def compute_returns(self, beta=None):
        """
        Discounted returns. If beta is a tensor (learnable), recomputes rewards
        from scaled deltas so beta gets gradients. Otherwise uses stored rewards.
        """
        if beta is not None and len(self.scaled_delta_f1s) > 0:
            # recompute rewards with current learnable beta (differentiable)
            rewards = [df1 + beta * dhom
                       for df1, dhom in zip(self.scaled_delta_f1s, self.scaled_delta_homs)]
        else:
            rewards = self.rewards

        if beta is not None:
            # differentiable path — keep as tensor list
            result = []
            R = torch.tensor(0.0, device=device)
            for r in reversed(rewards):
                R = r + args.gamma * R
                result.insert(0, R)
            return torch.stack(result)
        else:
            result, R = [], 0.0
            for r in reversed(rewards):
                R = r + args.gamma * R
                result.insert(0, R)
            return torch.tensor(result, dtype=torch.float32, device=device)


buffer = RolloutBuffer()

# ============================================================
# training loop
# ============================================================
best_macro_f1 = env.baseline_macro_f1   # selection metric: episode-end policy_val F1
best_episode  = 0
best_graph    = env.original_edge_index.cpu().clone()  # fall back to original graph
episode_rewards = []
episode_macro_f1s = []
episode_homophilys = []

print(f"\nbaseline: policy_val macro_f1={env.baseline_macro_f1:.4f}  "
      f"homophily(full)={env.baseline_homophily:.4f}  "
      f"homophily(reward)={env.baseline_reward_homophily:.4f}  beta={args.beta}\n")

for episode in range(1, args.n_episodes + 1):
    buffer.clear()
    env.reset()

    ep_reward  = 0.0
    ep_valid   = 0
    ep_adds    = 0
    ep_removes = 0

    # ---- collect one episode ----
    policy.eval()
    for step in range(args.max_steps):
        node_i     = env.sample_node_by_entropy()
        candidates = env.get_node_candidates(node_i)

        if not candidates:
            continue

        all_states = env.get_edge_states_batch(candidates).cpu()  # (N_cands, D)

        with torch.no_grad():
            logits, value = policy(all_states.to(device))
            dist          = torch.distributions.Categorical(logits=logits)
            action_idx    = dist.sample()
            log_prob      = dist.log_prob(action_idx)

        action = candidates[action_idx.item()]
        _, reward, done, info = env.step(action)
        ep_reward += reward
        ep_valid  += int(info["valid_action"])
        if action[2] == "add":
            ep_adds += 1
        else:
            ep_removes += 1

        buffer.add(
            all_states=all_states,
            action_idx=action_idx.item(),
            log_prob=log_prob.item(),
            reward=reward,
            value=value.item(),
            scaled_delta_f1=info.get("scaled_delta_f1", 0.0),
            scaled_delta_hom=info.get("scaled_delta_hom", 0.0),
        )

        if done:
            break

    T = len(buffer.rewards)
    if T == 0:
        continue

    old_lp_t = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    values_t = torch.tensor(buffer.values,    dtype=torch.float32, device=device)

    # ---- PPO update ----
    policy.train()
    for _ in range(args.n_epochs):
        # recompute log_probs + values from same candidate sets
        new_lp_list, new_val_list, ent_list = [], [], []
        for t in range(T):
            states_t = buffer.all_states[t].to(device)               # (N_cands, D)
            logits_t, val_t = policy(states_t)
            dist_t   = torch.distributions.Categorical(logits=logits_t)
            idx_t    = torch.tensor(buffer.action_idxs[t], device=device)
            new_lp_list.append(dist_t.log_prob(idx_t))
            new_val_list.append(val_t)
            ent_list.append(dist_t.entropy())

        new_lp  = torch.stack(new_lp_list)
        new_val = torch.stack(new_val_list)
        entropy = torch.stack(ent_list).mean()

        # recompute returns with current beta (differentiable if learnable)
        beta_val = policy.beta if args.learnable_beta else None
        returns  = buffer.compute_returns(beta=beta_val)
        advantages = (returns - values_t).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio     = (new_lp - old_lp_t).exp()
        loss_clip = -torch.min(
            ratio * advantages,
            ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * advantages,
        ).mean()
        loss_val  = F.mse_loss(new_val, returns)
        loss_ent  = -args.entropy_coef * entropy
        loss      = loss_clip + args.vf_coef * loss_val + loss_ent

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

    # ---- eval + logging (policy_val only during training) ----
    macro_f1, homophily = env.get_current_metrics()
    reward_homophily = env.get_current_reward_homophily()
    episode_rewards.append(ep_reward)
    episode_macro_f1s.append(macro_f1)
    episode_homophilys.append(homophily)

    delta_macro     = macro_f1  - env.baseline_macro_f1
    delta_homophily = homophily - env.baseline_homophily

    log_dict = {
        "episode": episode,
        "reward": ep_reward,
        "macro_f1": macro_f1,
        "homophily": homophily,
        "reward_homophily": reward_homophily,
        "delta_macro_f1": delta_macro,
        "delta_homophily": delta_homophily,
        "valid_actions": ep_valid,
        "n_edges": env.current_edge_index.size(1),
        "actions/adds": ep_adds,
        "actions/removes": ep_removes,
        "actions/add_ratio": ep_adds / max(ep_adds + ep_removes, 1),
        "loss/clip": loss_clip.item(),
        "loss/value": loss_val.item(),
        "loss/entropy": entropy.item(),
        "grad_norm": grad_norm.item(),
    }
    if args.learnable_beta:
        log_dict["beta"] = policy.beta.item()
    log(log_dict)

    # selection on policy_val: keep the GRAPH (the actual artifact) + policy
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_episode  = episode
        best_graph    = env.get_graph_copy()
        torch.save(policy.state_dict(), OUT_DIR / "best_policy.pt")
        torch.save({"edge_index": best_graph, "episode": episode,
                    "policy_val_macro_f1": macro_f1},
                   OUT_DIR / "best_graph.pt")

    if episode % 10 == 0 or episode <= 5:
        print(f"ep {episode:04d} | reward={ep_reward:+.4f} | "
              f"macro_f1={macro_f1:.4f} (d={delta_macro:+.4f}) | "
              f"homophily={homophily:.4f} (d={delta_homophily:+.4f}) | "
              f"edges={env.current_edge_index.size(1)}")

# ============================================================
# final summary + ONE-TIME test evaluation
# ============================================================
final_macro_f1, final_homophily = env.get_current_metrics()
final_graph = env.get_graph_copy()
torch.save({"edge_index": final_graph, "episode": args.n_episodes,
            "policy_val_macro_f1": final_macro_f1},
           OUT_DIR / "final_graph.pt")

# honest learning signal: average of the last 50 episode-end policy_val F1s.
# (the max over episodes inflates with episode count under noise -- a random
# policy "improves" by ~3.5 sigma over 500 episodes. never report the max alone.)
last50 = episode_macro_f1s[-50:] if len(episode_macro_f1s) >= 50 else episode_macro_f1s
mean_last50 = float(np.mean(last50)) if last50 else env.baseline_macro_f1
std_last50  = float(np.std(last50))  if last50 else 0.0

# test is touched exactly here, once per graph, after all training decisions
test_original = env.evaluate_graph(env.original_edge_index, split="test")
test_best     = env.evaluate_graph(best_graph,  split="test")
test_final    = env.evaluate_graph(final_graph, split="test")
hom_best_full   = env._eval_homophily(best_graph)
hom_best_reward = env._eval_reward_homophily(best_graph)

print(f"\n{'='*55}")
print(f"  DONE | {args.dataset} | {args.n_episodes} episodes")
print(f"{'='*55}")
print(f"  policy_val baseline:        {env.baseline_macro_f1:.4f}")
print(f"  policy_val mean (last 50):  {mean_last50:.4f} +/- {std_last50:.4f}  "
      f"(delta={mean_last50 - env.baseline_macro_f1:+.4f})")
print(f"  policy_val best (ep {best_episode:04d}):   {best_macro_f1:.4f}  "
      f"(selection metric -- inflated by max over episodes, do not headline)")
print(f"  avg reward (last 50):       {np.mean(episode_rewards[-50:]):.4f}")
print(f"  --- test (evaluated once, after training) ---")
print(f"  test F1 original graph:     {test_original['macro_f1']:.4f}  (GraphSAGE baseline)")
print(f"  test F1 best graph:         {test_best['macro_f1']:.4f}  "
      f"(delta={test_best['macro_f1'] - test_original['macro_f1']:+.4f})")
print(f"  test F1 final graph:        {test_final['macro_f1']:.4f}")

results = {
    "dataset": args.dataset,
    "split_seed": args.split_seed,
    "seed": args.seed,
    "n_episodes": args.n_episodes,
    "beta": args.beta,
    "learnable_beta": args.learnable_beta,
    "learned_beta_final": policy.beta.item() if args.learnable_beta else None,
    "git_commit": git_commit,
    "config_file": args.config,
    "wall_time_sec": round(time.time() - wall_start, 1),
    # reward calibration (frozen scales from the random-policy pre-pass)
    "reward_f1_scale": env.f1_scale,
    "reward_hom_scale": env.hom_scale,
    # homophily reporting: full graph (all labels) and reward view (non-test)
    "homophily_original": env.baseline_homophily,
    "homophily_original_reward": env.baseline_reward_homophily,
    "homophily_knn_pool": knn_pool_homophily,
    "homophily_refined": final_homophily,
    "homophily_best_graph": hom_best_full,
    "homophily_best_graph_reward": hom_best_reward,
    # policy_val (training-time signal + selection)
    "baseline_macro_f1": env.baseline_macro_f1,
    "mean_last50_policy_val_f1": mean_last50,
    "std_last50_policy_val_f1": std_last50,
    "best_macro_f1": best_macro_f1,            # selection metric (max over episodes)
    "best_episode": best_episode,
    "delta_macro_f1": best_macro_f1 - env.baseline_macro_f1,
    "final_macro_f1": final_macro_f1,
    # test (the headline numbers -- evaluated once, post-training)
    "test": {
        "original_graph": test_original,
        "best_graph": test_best,
        "final_graph": test_final,
        "delta_f1_best_vs_original": test_best["macro_f1"] - test_original["macro_f1"],
    },
    "episode_rewards": episode_rewards,
    "episode_macro_f1s": episode_macro_f1s,
    "episode_homophilys": episode_homophilys,
    "args": vars(args),
}
(OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
print(f"\nresults saved to {OUT_DIR / 'results.json'}")
print(f"graphs saved to {OUT_DIR / 'best_graph.pt'} / {OUT_DIR / 'final_graph.pt'}")

log({
    "final/macro_f1": final_macro_f1,
    "final/mean_last50_policy_val_f1": mean_last50,
    "final/best_macro_f1": best_macro_f1,
    "final/test_f1_original_graph": test_original["macro_f1"],
    "final/test_f1_best_graph": test_best["macro_f1"],
    "final/test_f1_final_graph": test_final["macro_f1"],
    "final/homophily_original": env.baseline_homophily,
    "final/homophily_knn_pool": knn_pool_homophily,
    "final/homophily_refined": final_homophily,
    "final/delta_homophily": final_homophily - env.baseline_homophily,
    "final/wall_time_sec": round(time.time() - wall_start, 1),
})

if args.wandb:
    wandb.finish()
