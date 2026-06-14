"""
one-off: does the trained policy actually improve TEST F1, or is the
reported best_macro_f1 a policy_val noise-maximization artifact?

for a given run: load frozen SAGE + best policy, roll out episodes, and at each
reward step record (policy_val F1, test F1) of the current graph. Then ask:
 - converged: mean final test F1 vs baseline test F1
 - selection transfer: take the step with best POLICY_VAL F1 (how best_policy.pt
   was chosen) and look at its TEST F1 -- does the val peak transfer to test?
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from graph_env import GraphEnv
from policy_net import PolicyNet
from sklearn.metrics import precision_recall_fscore_support

DATASET = sys.argv[1] if len(sys.argv) > 1 else "Cora"
BETA    = sys.argv[2] if len(sys.argv) > 2 else "1.0"
N_ROLLOUTS = 10
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)
env = GraphEnv(dataset=DATASET, split_seed=SEED, beta=float(BETA), knn_k=10, max_steps=50, reward_every=5)
dev = env.device

def f1_on(mask, edge_index):
    env.sage.eval()
    with torch.no_grad():
        logits = env.sage(env.x, edge_index)
        pred = logits[mask].argmax(1).cpu().numpy()
        true = env.y[mask].cpu().numpy()
    _, _, f1, _ = precision_recall_fscore_support(true, pred, average="macro", zero_division=0)
    return float(f1)

base_val  = f1_on(env.policy_val_mask, env.original_edge_index.to(dev))
base_test = f1_on(env.test_mask,       env.original_edge_index.to(dev))
print(f"\n{DATASET} beta={BETA}")
print(f"baseline:  policy_val={base_val:.4f}  test={base_test:.4f}")

policy = PolicyNet(edge_state_dim=env.edge_state_dim, hidden_dims=[256,128], dropout=0.1).to(dev)
sd = torch.load(f"runs/rl_{DATASET.lower()}/ppo_seed{SEED}_beta{BETA}_knnfix/best_policy.pt", map_location=dev)
policy.load_state_dict(sd); policy.eval()

def rollout(greedy):
    final_val, final_test = [], []
    n_iters = 1 if greedy else N_ROLLOUTS   # greedy on edge-choice is deterministic given node RNG; still varies via node sampling, so keep N
    n_iters = N_ROLLOUTS
    for r in range(n_iters):
        env.reset()
        for step in range(env.max_steps):
            node_i = env.sample_node_by_entropy()
            cands  = env.get_node_candidates(node_i)
            if not cands:
                continue
            states = env.get_edge_states_batch(cands).to(dev)
            with torch.no_grad():
                logits, _ = policy(states)
                a = logits.argmax().item() if greedy else torch.distributions.Categorical(logits=logits).sample().item()
            env.step(cands[a])
        final_val.append(f1_on(env.policy_val_mask, env.current_edge_index))
        final_test.append(f1_on(env.test_mask, env.current_edge_index))
    return np.array(final_val), np.array(final_test)

for mode, greedy in [("stochastic", False), ("greedy(argmax)", True)]:
    fv, ft = rollout(greedy)
    print(f"\n--- {mode} | {N_ROLLOUTS} rollouts ---")
    print(f"  final policy_val={fv.mean():.4f}±{fv.std():.4f}  test={ft.mean():.4f}±{ft.std():.4f}")
    print(f"  vs baseline:  val {fv.mean()-base_val:+.4f}   TEST {ft.mean()-base_test:+.4f}")
