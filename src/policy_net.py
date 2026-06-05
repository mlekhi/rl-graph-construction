"""
policy_net.py

PolicyNet: MLP that takes edge-pair state and outputs action probabilities.

Input per candidate edge (node_i, node_j, op):
    [s_i || s_j || cosine_sim || in_original_graph || op_is_add]
    where s_i = [h_i || sage_logits_i || degree_i || homophily_i || entropy_i]

Output:
    scalar logit -> sigmoid -> probability of taking this action

Also includes a value head (critic) and optionally a learnable beta parameter.

Learnable beta (NoisyNet-inspired):
    log_beta = nn.Parameter -- learned alongside policy weights
    beta = exp(log_beta) -- always positive
    reward = norm_delta_f1 + beta * norm_delta_homophily
    beta gets gradients through the value loss (returns depend on beta)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    MLP policy network over edge-pair states.

    Takes a batch of edge states and outputs:
    - action_logits: scalar per edge (actor head)
    - value: scalar per graph state (critic head, used in PPG)
    """

    def __init__(
        self,
        edge_state_dim: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.1,
        learnable_beta: bool = False,
        beta_init: float = 1.0,
    ):
        super().__init__()
        self.edge_state_dim = edge_state_dim
        self.learnable_beta = learnable_beta

        # shared trunk
        layers = []
        in_dim = edge_state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # actor head: outputs logit per candidate edge
        self.actor_head = nn.Linear(in_dim, 1)

        # critic head: outputs scalar value from mean pooled trunk output
        self.critic_head = nn.Linear(in_dim, 1)

        # learnable beta: log_beta is a scalar parameter, beta = exp(log_beta)
        # initialized so beta starts at beta_init
        if learnable_beta:
            import math
            self.log_beta = nn.Parameter(torch.tensor(math.log(beta_init)))
        else:
            self.log_beta = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, edge_states: torch.Tensor):
        """
        Args:
            edge_states: (N, edge_state_dim) -- batch of candidate edge states

        Returns:
            action_logits: (N,) -- unnormalized log-probs for each candidate
            value: scalar -- estimated value of current graph state
        """
        x = self.trunk(edge_states)
        action_logits = self.actor_head(x).squeeze(-1)   # (N,)
        value = self.critic_head(x.mean(dim=0)).squeeze() # scalar (mean pool)
        return action_logits, value

    @property
    def beta(self):
        """Current beta value. Learnable if log_beta is a parameter, else None."""
        if self.log_beta is not None:
            return self.log_beta.exp()
        return None

    def get_action_dist(self, edge_states: torch.Tensor):
        """
        Returns a categorical distribution over candidate edges.
        Sample from this to get the chosen action index.
        """
        logits, value = self(edge_states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value

    def evaluate_actions(self, edge_states: torch.Tensor, action_indices: torch.Tensor):
        """
        For PPG update: compute log_prob and entropy of taken actions.

        Args:
            edge_states: (N, edge_state_dim)
            action_indices: (B,) indices into edge_states for each taken action

        Returns:
            log_probs: (B,)
            entropy: scalar
            values: (B,)
        """
        logits, _ = self(edge_states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(action_indices)
        entropy = dist.entropy()

        # value per action step (recompute with action subset)
        x = self.trunk(edge_states[action_indices])
        values = self.critic_head(x).squeeze(-1)

        return log_probs, entropy, values


class AuxValueHead(nn.Module):
    """
    Auxiliary value head for PPG's auxiliary phase.
    Distills the critic's knowledge into a separate head to prevent
    policy interference during value function updates.
    """

    def __init__(self, edge_state_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        layers = []
        in_dim = edge_state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, edge_states: torch.Tensor):
        """Returns value estimate from mean-pooled edge states."""
        return self.net(edge_states.mean(dim=0)).squeeze()
