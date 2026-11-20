"""Prompt-conditioned support GNN. Target queries never enter the support graph."""
import torch
from torch import nn


class SupportGraphRouter(nn.Module):
    """Two rounds of support-query -> prompt-conditioned model message passing.

    Parallel query--model edges retain prompt identity and observed P/C. Model
    states are indexed by prompt (not learned pair IDs); arbitrary pool sizes
    use the same parameters. No target outcome is an argument to score().
    """
    def __init__(self, query_dim, task_dim, prompt_dim, model_dim, hidden=128):
        super().__init__()
        self.q = nn.Linear(query_dim + task_dim, hidden)
        self.p = nn.Linear(prompt_dim, hidden)
        self.m = nn.Linear(model_dim, hidden)
        self.messages = nn.ModuleList([nn.Sequential(
            nn.Linear(3 * hidden + 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden)) for _ in range(2)])
        self.updates = nn.ModuleList([nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.ReLU())
            for _ in range(2)])
        self.score_head = nn.Sequential(nn.Linear(4 * hidden, hidden),
                                        nn.ReLU(), nn.Linear(hidden, 1))

    def encode_support(self, support_x, prompt_x, model_x, outcomes, observed):
        # Shapes: support [K,D], outcomes [K,P,M,2], mask [K,P,M].
        if outcomes.shape != (*observed.shape, 2):
            raise ValueError('outcomes/mask shape mismatch')
        k, p, m = observed.shape
        if (k, p, m) != (len(support_x), len(prompt_x), len(model_x)):
            raise ValueError('support and pool alignment mismatch')
        if observed.dtype != torch.bool:
            raise ValueError('observed must be boolean')
        q = self.q(support_x)
        px, mx = self.p(prompt_x), self.m(model_x)
        h = mx.unsqueeze(0).expand(p, -1, -1)
        # Mask before the message MLP: absent outcomes cannot affect scores.
        safe = torch.where(observed.unsqueeze(-1), outcomes, torch.zeros_like(outcomes))
        for message, update in zip(self.messages, self.updates):
            features = torch.cat((q[:, None, None, :].expand(-1, p, m, -1),
                                  px[None, :, None, :].expand(k, -1, m, -1),
                                  h[None].expand(k, -1, -1, -1), safe), -1)
            messages = message(features) * observed.unsqueeze(-1)
            mean = messages.sum(0) / observed.sum(0).clamp_min(1).unsqueeze(-1)
            updated = update(torch.cat((h, mean), -1))
            h = torch.where(observed.any(0).unsqueeze(-1), updated, h)
        return px, h

    def score(self, target_x, encoded):
        px, h = encoded
        q = self.q(target_x)
        n, p, m = len(q), len(px), h.shape[1]
        features = torch.cat((q[:, None, None, :].expand(-1, p, m, -1),
                              px[None, :, None, :].expand(n, -1, m, -1),
                              h[None].expand(n, -1, -1, -1),
                              q[:, None, None, :] * h[None]), -1)
        return self.score_head(features).squeeze(-1)
