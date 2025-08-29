# step3_second_stage.py
from __future__ import annotations
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from step1_clusters import ActionCatalog, Cluster
from step3_action_payload import ActionPayloadEncoder

class FhatMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, int] = (256, 128), dropout: float = 0.1):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),  # scalar reward
        )

    def forward(self, x):  # x: [B, in_dim]
        return self.net(x)  # [B, 1]

class FhatScorer:
    """
    Convenience wrapper to score actions and compute per-cluster argmax/value.
    """
    def __init__(self, model: FhatMLP, catalog: ActionCatalog, payload_enc: ActionPayloadEncoder, device: torch.device):
        self.model = model.to(device)
        self.catalog = catalog
        self.payload = payload_enc
        self.device = device

        # Precompute candidate action ids by cluster (exclude REJECT)
        self.full_ids    = catalog.actions_in_cluster(Cluster.FULL)
        self.partial_ids = catalog.actions_in_cluster(Cluster.PARTIAL)

    @torch.no_grad()
    def score_actions(self, X: torch.Tensor, action_ids: List[int]) -> torch.Tensor:
        """
        X: [B, d_x]; returns scores [B, len(action_ids)]
        """
        self.model.eval()
        B = X.shape[0]
        payloads = torch.from_numpy(self.payload.batch_encode(action_ids)).to(self.device)  # [K, d_a]
        X = X.to(self.device)

        # Broadcast X against K actions by concatenating payloads
        K = payloads.shape[0]
        X_rep = X.unsqueeze(1).expand(B, K, X.shape[1])      # [B, K, d_x]
        A_rep = payloads.unsqueeze(0).expand(B, K, payloads.shape[1])  # [B, K, d_a]
        XA = torch.cat([X_rep, A_rep], dim=-1).reshape(B * K, -1)      # [B*K, d_x+d_a]

        scores = self.model(XA).reshape(B, K)  # [B, K]
        return scores

    @torch.no_grad()
    def cluster_argmax(self, X: torch.Tensor, cluster: Cluster) -> List[int]:
        """
        For each sample in X, return the best action_id within cluster.
        """
        ids = self.full_ids if cluster == Cluster.FULL else self.partial_ids
        scores = self.score_actions(X, ids)  # [B, K]
        best_idx = torch.argmax(scores, dim=1)  # [B]
        return [ids[i] for i in best_idx.tolist()]

    @torch.no_grad()
    def cluster_value(self, X: torch.Tensor, cluster: Cluster) -> torch.Tensor:
        """
        For each sample in X, return max predicted reward within the cluster.
        """
        ids = self.full_ids if cluster == Cluster.FULL else self.partial_ids
        scores = self.score_actions(X, ids)  # [B, K]
        return torch.max(scores, dim=1).values  # [B]
