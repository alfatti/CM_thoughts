# step5_first_stage_policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from step1_clusters import ActionCatalog, Cluster
from step2_context_featurizer import RFQContext, RFQFeatureBuilder
from step3_datasets import LoggedExample, ActionPayloadEncoder, FhatDesignMatrixBuilder
from step3_fhat_models import FhatModel
from step4_logging_policy import LoggingPolicyModel

# -----------------------------
# Shallow policy: pi^1st(c|x)
# -----------------------------
class ShallowPolicyNN(nn.Module):
    def __init__(self, in_dim: int, n_clusters: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_clusters),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (log_probs [B,C], probs [B,C])."""
        logits = self.net(X)
        logp = self.logsoftmax(logits)
        p = torch.exp(logp)
        return logp, p

# -----------------------------
# Config
# -----------------------------
@dataclass
class FirstStageConfig:
    hidden: int = 64
    lr: float = 5e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    epochs: int = 15
    grad_clip: float = 5.0
    # importance weight safety
    propensity_floor: float = 1e-3
    w_clip: float = 50.0
    # optional: tiny entropy regularization (helps exploration early)
    entropy_reg: float = 0.0

# -----------------------------
# Precompute training tensors
# -----------------------------
def build_stage1_tensors(
    logs: List[LoggedExample],
    fb: RFQFeatureBuilder,
    fhat: FhatModel,
    pi0: LoggingPolicyModel,
    catalog: ActionCatalog,
    ape: ActionPayloadEncoder,
    dm: FhatDesignMatrixBuilder,
) -> Dict[str, np.ndarray]:
    """
    Returns a dict of numpy arrays:
      X          : [N, d_x]          model features
      c_idx      : [N]               cluster index in {0,1,2}
      r          : [N, 1]            realized reward
      fhat_xa    : [N, 1]            control-variate at logged action
      pi0_probs  : [N, 3]            logging cluster propensities (floored, renormed)
      fhat_clust : [N, 3]            per-cluster max-a fhat(x,a) (REJECT/FULL/PARTIAL)
    """
    rfqs = [z.rfq for z in logs]
    X = fb.transform_batch(rfqs).astype(np.float32)                         # [N,d_x]
    r = np.array([z.reward for z in logs], dtype=np.float32).reshape(-1, 1) # [N,1]

    # cluster index
    from step1_clusters import Cluster
    c_idx = np.array([int(catalog.cluster_of(z.action_id)) for z in logs], dtype=np.int64)

    # pi0 cluster probabilities (already floors/clips + renorm inside)
    pi0_probs = pi0.predict_proba(rfqs).astype(np.float32)                  # [N,3]

    # control variate fhat(x, a_logged)
    aid_list = [z.action_id for z in logs]
    A_logged = ape.encode_many(aid_list)                                    # [N,d_a]
    fhat_xa = fhat.predict(X, A_logged).astype(np.float32)                  # [N,1]

    # per-cluster fhat^{pi2nd}(x, c) — max-a in each cluster via argmax scorer
    fhat_clust = np.zeros((X.shape[0], 3), dtype=np.float32)
    for c in [Cluster.REJECT, Cluster.FULL, Cluster.PARTIAL]:
        vals = fhat.cluster_value(dm, X, catalog, c)                        # [N,1]
        fhat_clust[:, int(c)] = vals.reshape(-1)

    return dict(X=X, c_idx=c_idx, r=r, fhat_xa=fhat_xa, pi0_probs=pi0_probs, fhat_clust=fhat_clust)

# -----------------------------
# OPE (cluster-level DR)
# -----------------------------
def cluster_dr_ope(
    policy: ShallowPolicyNN,
    X: np.ndarray,
    c_idx: np.ndarray,
    r: np.ndarray,
    fhat_xa: np.ndarray,
    pi0_probs: np.ndarray,
    fhat_clust: np.ndarray,
    cfg: FirstStageConfig,
    device: Optional[torch.device] = None,
) -> float:
    """
    \hat V = 1/N sum_i [ (pi_theta(c_i|x_i)/pi0(c_i|x_i)) * (r_i - fhat(x_i, a_i))
                        + sum_c pi_theta(c|x_i) * fhat^{pi2nd}(x_i,c) ]
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    with torch.no_grad():
        tX = torch.from_numpy(X).to(device)
        logp, p = policy(tX)                                # [N,3]
        P = p.cpu().numpy()

    eps = cfg.propensity_floor
    pi0 = np.clip(pi0_probs, eps, 1.0)
    pi0 = pi0 / pi0.sum(axis=1, keepdims=True)

    # Term A
    idx = np.arange(X.shape[0])
    pi_c = P[idx, c_idx]                                    # [N]
    w = pi_c / pi0[idx, c_idx]
    w = np.clip(w, 0.0, cfg.w_clip)
    A = w.reshape(-1,1) * (r - fhat_xa)                     # [N,1]

    # Term B
    B = (P * fhat_clust).sum(axis=1, keepdims=True)         # [N,1]

    Vhat = float(np.mean(A + B))
    return Vhat

# -----------------------------
# Trainer
# -----------------------------
def train_first_stage(
    train_tensors: Dict[str, np.ndarray],
    dev_tensors: Dict[str, np.ndarray],
    cfg: FirstStageConfig,
    input_dim: int,
    seed: int = 1337,
    verbose: bool = True,
) -> Tuple[ShallowPolicyNN, Dict[str, float]]:
    """
    Trains pi^1st with the POTEC gradient estimator (unbiased):
      g_i = w_i * (r_i - fhat_i) * ∇ log pi(c_i|x_i) + sum_c pi(c|x_i) * fhat^{pi2nd}(x_i,c) ∇ log pi(c|x_i)
    We implement this by maximizing:
      L = mean_i[ w_detached * (r - fhat) * log pi(c_i|x_i) + sum_c pi(c|x_i) * fhat_clust_detached ]
    where w_detached uses pi(c_i|x_i).detach()/pi0(c_i|x_i), and fhat terms are detached.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = ShallowPolicyNN(in_dim=input_dim, n_clusters=3, hidden=cfg.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Build DataLoaders
    def to_tensor_dataset(tensors: Dict[str, np.ndarray]) -> TensorDataset:
        X = torch.from_numpy(tensors["X"]).float()
        c = torch.from_numpy(tensors["c_idx"]).long()
        r = torch.from_numpy(tensors["r"]).float()
        fxa = torch.from_numpy(tensors["fhat_xa"]).float()
        pi0 = torch.from_numpy(tensors["pi0_probs"]).float()
        fc = torch.from_numpy(tensors["fhat_clust"]).float()
        return TensorDataset(X, c, r, fxa, pi0, fc)

    train_ds = to_tensor_dataset(train_tensors)
    dev_ds   = to_tensor_dataset(dev_tensors)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dev_dl   = DataLoader(dev_ds, batch_size=4096, shuffle=False, drop_last=False)

    best_ope = -1e18
    best_state = None
    history: Dict[str, float] = {}

    for epoch in range(cfg.epochs):
        model.train()
        total_obj = 0.0
        n_ex = 0

        for (X, c, r, fxa, pi0, fc) in train_dl:
            X, c, r, fxa, pi0, fc = X.to(device), c.to(device), r.to(device), fxa.to(device), pi0.to(device), fc.to(device)
            logp, p = model(X)                               # [B,3] each

            # --- Term A: w_detached * (r - fhat) * log pi(c|x)
            # pi(c|x) from current model
            pi_c = p.gather(1, c.view(-1,1)).squeeze(1)      # [B]
            # importance weight with pi(c|x).detach() in numerator
            eps = cfg.propensity_floor
            pi0_c = torch.clamp(pi0.gather(1, c.view(-1,1)).squeeze(1), min=eps)  # [B]
            w = (pi_c.detach() / pi0_c).clamp(0.0, cfg.w_clip)                    # [B]
            # control variate
            adv = (r - fxa).squeeze(1)                        # [B]
            logp_c = logp.gather(1, c.view(-1,1)).squeeze(1)  # [B]
            termA = w * adv * logp_c                          # [B]

            # --- Term B: sum_c pi(c|x) * fhat^{pi2nd}(x,c) with fhat detached
            termB = (p * fc.detach()).sum(dim=1)              # [B]

            # --- Optional entropy reg (encourage non-degenerate early policies)
            if cfg.entropy_reg > 0.0:
                entropy = -(p * logp).sum(dim=1)              # [B]
                obj = (termA + termB + cfg.entropy_reg * entropy).mean()
            else:
                obj = (termA + termB).mean()

            loss = -obj
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            total_obj += float(obj.item()) * X.size(0)
            n_ex += X.size(0)

        # ---- Eval OPE on dev
        # materialize full dev arrays
        Xd = dev_tensors["X"]; cd = dev_tensors["c_idx"]; rd = dev_tensors["r"]
        fxa_d = dev_tensors["fhat_xa"]; pi0d = dev_tensors["pi0_probs"]; fcd = dev_tensors["fhat_clust"]
        dev_ope = cluster_dr_ope(model, Xd, cd, rd, fxa_d, pi0d, fcd, cfg, device=device)

        if verbose:
            print(f"[epoch {epoch:02d}] train_obj={total_obj/n_ex:.6f}  dev_OPE={dev_ope:.6f}")

        history[f"epoch_{epoch}/dev_OPE"] = dev_ope

        if dev_ope > best_ope:
            best_ope = dev_ope
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"dev_OPE": best_ope}
