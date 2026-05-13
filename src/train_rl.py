"""
train_rl.py

PPO training loop for RL graph construction.

Run:
    python3 src/train_rl.py --dataset Cora --split_seed 42
    python3 src/train_rl.py --dataset Cora --split_seed 42 --smoke_test  # 20 episodes
"""

import argparse
import json
import os
import random
from pathlib import Path

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
parser.add_argument("--dataset",      default="Cora", choices=["Cora", "PubMed", "CiteSeer"])
parser.add_argument("--split_seed",   type=int, default=42)
parser.add_argument("--n_episodes",   type=int, default=500)
parser.add_argument("--max_steps",    type=int, default=50)
parser.add_argument("--reward_every", type=int, default=5)
parser.add_argument("--gamma_reward", type=float, default=0.5,  help="minority class reward weight")
parser.add_argument("--knn_k",        type=int, default=10)
parser.add_argument("--lr",           type=float, default=3e-4)
parser.add_argument("--gamma",        type=float, default=0.99, help="discount factor")
parser.add_argument("--clip_eps",     type=float, default=0.2,  help="PPO clip ratio")
parser.add_argument("--vf_coef",      type=float, default=0.5)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--n_epochs",     type=int, default=4,      help="PPO update epochs per episode")
parser.add_argument("--smoke_test",   action="store_true",      help="run 20 episodes only")
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--wandb",        action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_episodes = 20

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "runs" / f"rl_{args.dataset.lower()}" / f"ppo_seed{args.split_seed}"
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

# ============================================================
# wandb (optional)
# ============================================================
if args.wandb:
    import wandb
    wandb.init(
        project="rl-graph-construction",
        name=f"PPO-{args.dataset}-seed{args.split_seed}",
        config=vars(args),
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
    gamma=args.gamma_reward,
    knn_k=args.knn_k,
    max_steps=args.max_steps,
    reward_every=args.reward_every,
    device=device,
)

policy = PolicyNet(
    edge_state_dim=env.edge_state_dim,
    hidden_dims=[256, 128],
    dropout=0.1,
).to(device)

optimizer = AdamW(policy.parameters(), lr=args.lr)

print(f"\nPPO training | {args.dataset} | {args.n_episodes} episodes | "
      f"max_steps={args.max_steps} | device={device}")
print(f"policy params: {sum(p.numel() for p in policy.parameters()):,}")

# ============================================================
# rollout buffer
# ============================================================
class RolloutBuffer:
    """
    Stores one episode. Per step we keep the full candidate set so we can
    recompute log_probs from the same categorical distribution during update.
    With node-first selection each step has ~10-25 candidates × 151 dims — tiny.
    """
    def __init__(self):
        self.all_states  = []   # list of (N_cands, edge_state_dim) tensors
        self.action_idxs = []   # int index into all_states[t]
        self.log_probs   = []   # scalar, log_prob of chosen action
        self.rewards     = []   # float
        self.values      = []   # scalar critic estimate

    def clear(self):
        self.__init__()

    def add(self, all_states, action_idx, log_prob, reward, value):
        self.all_states.append(all_states)   # keep on cpu to save GPU mem
        self.action_idxs.append(action_idx)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def compute_returns(self):
        """Discounted returns (no bootstrapping — episode ends naturally)."""
        returns, R = [], 0.0
        for r in reversed(self.rewards):
            R = r + args.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=device)


buffer = RolloutBuffer()

# ============================================================
# training loop
# ============================================================
best_macro_f1 = env.baseline_macro_f1
episode_rewards = []
episode_macro_f1s = []

print(f"\nbaseline: macro_f1={env.baseline_macro_f1:.4f}  minority_f1={env.baseline_minority_f1:.4f}\n")

for episode in range(1, args.n_episodes + 1):
    buffer.clear()
    env.reset()

    ep_reward = 0.0
    ep_valid  = 0

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

        buffer.add(
            all_states=all_states,         # cpu tensor
            action_idx=action_idx.item(),  # int
            log_prob=log_prob.item(),      # float
            reward=reward,
            value=value.item(),
        )

        if done:
            break

    T = len(buffer.rewards)
    if T == 0:
        continue

    returns    = buffer.compute_returns()                              # (T,)
    old_lp_t   = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)  # (T,)
    values_t   = torch.tensor(buffer.values,    dtype=torch.float32, device=device)  # (T,)

    advantages = returns - values_t
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

        new_lp  = torch.stack(new_lp_list)    # (T,)
        new_val = torch.stack(new_val_list)   # (T,)
        entropy = torch.stack(ent_list).mean()

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
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()

    # ---- eval + logging ----
    macro_f1, minority_f1 = env.get_current_f1()
    episode_rewards.append(ep_reward)
    episode_macro_f1s.append(macro_f1)

    delta_macro   = macro_f1   - env.baseline_macro_f1
    delta_minority = minority_f1 - env.baseline_minority_f1

    log({
        "episode": episode,
        "reward": ep_reward,
        "macro_f1": macro_f1,
        "minority_f1": minority_f1,
        "delta_macro_f1": delta_macro,
        "delta_minority_f1": delta_minority,
        "valid_actions": ep_valid,
        "n_edges": env.current_edge_index.size(1),
        "loss_clip": loss_clip.item(),
        "loss_val": loss_val.item(),
        "entropy": entropy.item(),
    })

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        torch.save(policy.state_dict(), OUT_DIR / "best_policy.pt")

    if episode % 10 == 0 or episode <= 5:
        print(f"ep {episode:04d} | reward={ep_reward:+.4f} | "
              f"macro_f1={macro_f1:.4f} (d={delta_macro:+.4f}) | "
              f"minority_f1={minority_f1:.4f} (d={delta_minority:+.4f}) | "
              f"edges={env.current_edge_index.size(1)}")

# ============================================================
# final summary
# ============================================================
print(f"\n{'='*55}")
print(f"  DONE | {args.dataset} | {args.n_episodes} episodes")
print(f"{'='*55}")
print(f"  baseline macro_f1:  {env.baseline_macro_f1:.4f}")
print(f"  best macro_f1:      {best_macro_f1:.4f}  "
      f"(delta={best_macro_f1 - env.baseline_macro_f1:+.4f})")
print(f"  avg reward (last 50): {np.mean(episode_rewards[-50:]):.4f}")

results = {
    "dataset": args.dataset,
    "split_seed": args.split_seed,
    "n_episodes": args.n_episodes,
    "baseline_macro_f1": env.baseline_macro_f1,
    "baseline_minority_f1": env.baseline_minority_f1,
    "best_macro_f1": best_macro_f1,
    "delta_macro_f1": best_macro_f1 - env.baseline_macro_f1,
    "episode_rewards": episode_rewards,
    "episode_macro_f1s": episode_macro_f1s,
    "args": vars(args),
}
(OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
print(f"\nresults saved to {OUT_DIR / 'results.json'}")

if args.wandb:
    wandb.finish()
