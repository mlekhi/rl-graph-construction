"""
train_rl.py

PPG (Phasic Policy Gradient) training loop for RL graph construction.

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

# local
import sys
sys.path.insert(0, str(Path(__file__).parent))
from graph_env import GraphEnv
from policy_net import PolicyNet, AuxValueHead

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
parser.add_argument("--lr_policy",    type=float, default=3e-4)
parser.add_argument("--lr_value",     type=float, default=1e-3)
parser.add_argument("--clip_eps",     type=float, default=0.2,  help="PPO clip ratio")
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--n_policy_epochs",  type=int, default=4,  help="PPG policy phase epochs")
parser.add_argument("--n_aux_epochs",     type=int, default=6,  help="PPG auxiliary phase epochs")
parser.add_argument("--aux_every",        type=int, default=32, help="run aux phase every N episodes")
parser.add_argument("--smoke_test",   action="store_true", help="run 20 episodes only")
parser.add_argument("--seed",         type=int, default=42)
parser.add_argument("--wandb",        action="store_true")
args = parser.parse_args()

if args.smoke_test:
    args.n_episodes = 20

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "runs" / f"rl_{args.dataset.lower()}" / f"ppg_seed{args.split_seed}"
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
        name=f"PPG-{args.dataset}-seed{args.split_seed}",
        config=vars(args),
    )
    log = wandb.log
else:
    log = lambda d, **kw: None  # no-op


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

aux_head = AuxValueHead(
    edge_state_dim=env.edge_state_dim,
    hidden_dims=[256, 128],
).to(device)

opt_policy = AdamW(policy.parameters(), lr=args.lr_policy)
opt_aux    = AdamW(aux_head.parameters(), lr=args.lr_value)

print(f"\nPPG training | {args.dataset} | {args.n_episodes} episodes | "
      f"max_steps={args.max_steps} | device={device}")
print(f"policy params: {sum(p.numel() for p in policy.parameters()):,}")


# ============================================================
# rollout buffer
# ============================================================
class RolloutBuffer:
    """Stores one episode of experience. Only stores chosen edge state per step."""
    def __init__(self):
        self.chosen_states = []  # edge state of chosen action (1 per step)
        self.log_probs_old = []  # log prob under old policy
        self.rewards       = []  # reward at each step
        self.values        = []  # critic value at each step

    def clear(self):
        self.__init__()

    def add(self, chosen_state, log_prob, reward, value):
        self.chosen_states.append(chosen_state)
        self.log_probs_old.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)

    def compute_returns(self, gamma=0.99):
        """Discounted returns."""
        returns = []
        R = 0.0
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32, device=device)


buffer = RolloutBuffer()
aux_buffer_states  = []  # accumulate across episodes for aux phase
aux_buffer_returns = []


# ============================================================
# training loop
# ============================================================
best_macro_f1 = env.baseline_macro_f1
episode_rewards = []
episode_macro_f1s = []

print(f"\nbaseline: macro_f1={env.baseline_macro_f1:.4f}  minority_f1={env.baseline_minority_f1:.4f}\n")

for episode in range(1, args.n_episodes + 1):
    buffer.clear()
    node_states, _ = env.reset()

    ep_reward = 0.0
    ep_valid  = 0

    for step in range(args.max_steps):
        # node-first: pick node by entropy weight, evaluate its candidates only
        node_i     = env.sample_node_by_entropy()
        candidates = env.get_node_candidates(node_i)

        if not candidates:
            continue

        # build edge states for this node's candidates (~10-25 candidates)
        all_states = env.get_edge_states_batch(candidates)  # (N_cands, edge_state_dim)

        # sample action
        policy.eval()
        with torch.no_grad():
            dist, value = policy.get_action_dist(all_states)
            action_idx  = dist.sample()
            log_prob    = dist.log_prob(action_idx)

        action       = candidates[action_idx.item()]
        chosen_state = all_states[action_idx]

        # step env
        node_states, reward, done, info = env.step(action)
        ep_reward += reward
        ep_valid  += int(info["valid_action"])

        buffer.add(
            chosen_state=chosen_state.cpu(),
            log_prob=log_prob.cpu(),
            reward=reward,
            value=value.cpu(),
        )

        if done:
            break

    # ---- PPG policy phase ----
    returns    = buffer.compute_returns()
    chosen_t   = torch.stack(buffer.chosen_states).to(device)  # (T, edge_state_dim)
    old_lp_t   = torch.stack(buffer.log_probs_old).to(device)  # (T,)
    values_t   = torch.stack(buffer.values).squeeze().to(device)  # (T,)

    advantages = returns - values_t
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy.train()
    for _ in range(args.n_policy_epochs):
        logits, new_values = policy(chosen_t)
        # treat each chosen state as a single-candidate set -> log_prob = log_sigmoid(logit)
        new_lp = F.logsigmoid(logits.squeeze())
        entropy = torch.distributions.Bernoulli(logits=logits.squeeze()).entropy().mean()

        ratio     = (new_lp - old_lp_t).exp()
        loss_clip = -torch.min(
            ratio * advantages,
            ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * advantages
        ).mean()
        loss_val  = F.mse_loss(new_values.squeeze(), returns)
        loss_ent  = -args.entropy_coef * entropy
        loss      = loss_clip + 0.5 * loss_val + loss_ent

        opt_policy.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        opt_policy.step()

    # accumulate chosen states for aux phase
    aux_buffer_states.append(chosen_t.detach().cpu())
    aux_buffer_returns.extend(returns.tolist())

    # ---- PPG auxiliary phase (every aux_every episodes) ----
    if episode % args.aux_every == 0 and aux_buffer_states:
        all_chosen = torch.cat(aux_buffer_states, dim=0).to(device)  # (N, edge_state_dim)
        all_returns = torch.tensor(aux_buffer_returns, dtype=torch.float32, device=device)

        for _ in range(args.n_aux_epochs):
            aux_vals = aux_head(all_chosen)
            loss_aux = F.mse_loss(aux_vals.squeeze(), all_returns)

            with torch.no_grad():
                old_logits = policy(all_chosen)[0].detach()
            new_logits = policy(all_chosen)[0]
            distill_loss = F.kl_div(
                F.log_softmax(new_logits, dim=-1),
                F.softmax(old_logits, dim=-1),
                reduction="batchmean"
            )

            loss = loss_aux + distill_loss
            opt_aux.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(aux_head.parameters(), max_norm=0.5)
            opt_aux.step()

        aux_buffer_states.clear()
        aux_buffer_returns.clear()

    # ---- eval + logging ----
    macro_f1, minority_f1 = env.get_current_f1()
    episode_rewards.append(ep_reward)
    episode_macro_f1s.append(macro_f1)

    delta_macro   = macro_f1 - env.baseline_macro_f1
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
        "policy_loss": loss_clip.item(),
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

# save results
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
