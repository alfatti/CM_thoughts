
"""
lightgcn_causal_training_with_noactivity_backoff.py

LightGCN-W (causal, rolling-window) trainer with:
- BPR + Dynamic Negative Sampling (DNS)
- No-activity backoff at eval/serve time:
  Tier A: use U_t[u] if user active in window
  Tier B: else use last cached user embedding from most recent active day
  Tier C: else use RecentPop scores for the window
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Sparse adjacency + model ----------------

def build_symmetric_normalized_adj(num_nodes: int,
                                   edge_index: torch.Tensor,
                                   edge_weight: torch.Tensor) -> torch.sparse.FloatTensor:
    if edge_index.numel() == 0:
        idx = torch.zeros((2, 0), dtype=torch.long, device=edge_index.device)
        val = torch.zeros((0,), dtype=torch.float32, device=edge_index.device)
        return torch.sparse_coo_tensor(idx, val, size=(num_nodes, num_nodes)).coalesce()
    src, dst, w = edge_index[0], edge_index[1], edge_weight
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=w.device)
    deg.index_add_(0, src, w); deg.index_add_(0, dst, w)
    deg = torch.clamp(deg, min=1e-12)
    norm = w / torch.sqrt(deg[src] * deg[dst])
    indices = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    values  = torch.cat([norm, norm])
    return torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()

class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, emb_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.num_nodes = num_users + num_items
        self.num_layers = num_layers
        self.emb = nn.Embedding(self.num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, A_norm: torch.sparse.FloatTensor):
        xk = self.emb.weight
        out = xk
        for k in range(1, self.num_layers + 1):
            xk = torch.sparse.mm(A_norm, xk)
            out = out + xk * (1.0 / (k + 1))
        return out[:self.num_users], out[self.num_users:]

# ---------------- BPR with Dynamic Negative Sampling ----------------

def bpr_loss_dns(user_embs: torch.Tensor, item_embs: torch.Tensor,
                 pos_pairs: List[Tuple[int, int]], num_items: int,
                 neg_ratio: int = 10, dns_pool_size: int = 50,
                 device: Optional[torch.device] = None) -> torch.Tensor:
    if len(pos_pairs) == 0:
        return torch.tensor(0.0, device=device if device else 'cpu', requires_grad=True)
    u_idx = torch.tensor([u for (u,i) in pos_pairs], dtype=torch.long, device=device)
    i_pos = torch.tensor([i for (u,i) in pos_pairs], dtype=torch.long, device=device)
    u_vec = user_embs[u_idx]; i_pos_vec = item_embs[i_pos]
    P = u_idx.shape[0]
    pool = torch.randint(low=0, high=num_items, size=(P, dns_pool_size), device=device)
    pool[pool == i_pos.unsqueeze(1)] = (pool[pool == i_pos.unsqueeze(1)] + 1) % num_items
    items_pool = item_embs[pool]
    scores_pool = torch.einsum('pd,pkd->pk', u_vec, items_pool)
    hard_idx = scores_pool.argmax(dim=1)
    i_neg_hard = pool[torch.arange(P, device=device), hard_idx]
    i_neg_all = i_neg_hard.unsqueeze(1).repeat(1, neg_ratio).reshape(-1)
    u_idx_all = u_idx.unsqueeze(1).repeat(1, neg_ratio).reshape(-1)
    i_pos_all = i_pos.unsqueeze(1).repeat(1, neg_ratio).reshape(-1)
    u_vec_all = user_embs[u_idx_all]
    i_pos_vec_all = item_embs[i_pos_all]
    i_neg_vec_all = item_embs[i_neg_all]
    pos = (u_vec_all * i_pos_vec_all).sum(dim=1)
    neg = (u_vec_all * i_neg_vec_all).sum(dim=1)
    return -F.logsigmoid(pos - neg).mean()

# ---------------- Snapshot helpers + metrics ----------------

def extract_pos_pairs_from_snapshot(snapshot: Dict[str, Any]) -> List[Tuple[int,int]]:
    if isinstance(snapshot, dict):
        ei = snapshot['edge_index']
        if isinstance(ei, np.ndarray):
            ei = torch.as_tensor(ei, dtype=torch.long)
        num_users = snapshot['num_users']
    else:
        ei = snapshot.edge_index
        num_users = snapshot.num_users
    if ei.numel() == 0: return []
    src = ei[0]; dst_local = ei[1] - num_users
    pairs = torch.stack([src, dst_local], dim=1).unique(dim=0)
    return [(int(u.item()), int(i.item())) for u,i in pairs]

def recall_at_k(ranked_items: List[int], relevant: set, k: int = 50) -> float:
    topk = ranked_items[:k]; return sum(1 for it in topk if it in relevant) / max(len(relevant), 1)

def average_precision(ranked_items: List[int], relevant: set) -> float:
    if not relevant: return 0.0
    hits, s = 0, 0.0
    for r,it in enumerate(ranked_items, start=1):
        if it in relevant:
            hits += 1; s += hits / r
    return s / len(relevant)

# ---------------- No-activity backoff helpers ----------------

def user_has_edges_today(snapshot, user_idx: int) -> bool:
    if isinstance(snapshot, dict):
        ei = snapshot['edge_index']
        if isinstance(ei, np.ndarray):
            ei = torch.as_tensor(ei, dtype=torch.long)
    else:
        ei = snapshot.edge_index
    return (ei[0] == int(user_idx)).any().item()

def build_recentpop_scores(snapshot, device, num_users: int, num_items: int) -> torch.Tensor:
    if isinstance(snapshot, dict):
        ei = snapshot['edge_index']
        ei = torch.as_tensor(ei, dtype=torch.long, device=device) if isinstance(ei, np.ndarray) else ei.to(device)
    else:
        ei = snapshot.edge_index.to(device)
    item_idx = ei[1] - num_users
    return torch.bincount(item_idx, minlength=num_items).float()

def get_scores_with_backoff(u: int, U_t: torch.Tensor, I_t: torch.Tensor, snapshot_t,
                            last_cache: Dict[int, torch.Tensor],
                            recentpop_scores_t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if user_has_edges_today(snapshot_t, u):
        e_u = U_t[u]; return (e_u.unsqueeze(0) @ I_t.T).squeeze(0)
    if u in last_cache and last_cache[u] is not None:
        e_u = last_cache[u].to(device=device, dtype=I_t.dtype)
        return (e_u.unsqueeze(0) @ I_t.T).squeeze(0)
    return recentpop_scores_t

def evaluate_day_backoff(model, snapshot, device, last_cache: Dict[int, torch.Tensor], K: int = 50):
    if isinstance(snapshot, dict):
        ei, ew = snapshot['edge_index'], snapshot['edge_weight']
        ei = torch.as_tensor(ei, dtype=torch.long, device=device) if isinstance(ei, np.ndarray) else ei.to(device)
        ew = torch.as_tensor(ew, dtype=torch.float32, device=device) if isinstance(ew, np.ndarray) else ew.to(device)
        Uc, Ic = snapshot['num_users'], snapshot['num_items']
        N = Uc + Ic
    else:
        ei, ew = snapshot.edge_index.to(device), snapshot.edge_weight.to(device)
        Uc, Ic, N = snapshot.num_users, snapshot.num_items, snapshot.num_nodes
    A = build_symmetric_normalized_adj(N, ei, ew).to(device)
    with torch.no_grad():
        U_t, I_t = model(A)

    pos_pairs = extract_pos_pairs_from_snapshot(snapshot)
    rel = {}
    for u,i in pos_pairs:
        rel.setdefault(u, set()).add(i)

    recentpop_t = build_recentpop_scores(snapshot, device=device, num_users=Uc, num_items=Ic)

    recalls, maps = [], []
    for u, relevant in rel.items():
        scores = get_scores_with_backoff(u, U_t, I_t, snapshot, last_cache, recentpop_t, device)
        ranked = torch.argsort(scores, descending=True).tolist()
        recalls.append(recall_at_k(ranked, relevant, k=K))
        maps.append(average_precision(ranked, relevant))
    if not recalls: return 0.0, 0.0
    return float(np.mean(recalls)), float(np.mean(maps))

# ---------------- Training loop (with backoff cache) ----------------

def train_lightgcn_w_with_noactivity_backoff(
    snapshots: Dict[str, Any],
    train_days: List[str],
    val_days: Optional[List[str]] = None,
    test_days: Optional[List[str]] = None,
    emb_dim: int = 64,
    num_layers: int = 1,
    lr: float = 1e-4,
    epochs: int = 5,
    neg_ratio: int = 10,
    dns_pool_size: int = 50,
    eval_every: int = 1,
    device: Optional[str] = None
):
    meta = snapshots['__meta__']
    Uc, Ic = meta['num_users'], meta['num_items']
    N = Uc + Ic

    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = LightGCN(Uc, Ic, emb_dim=emb_dim, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # user_idx -> last valid embedding (CPU tensor)
    last_valid_user_embedding: Dict[int, torch.Tensor] = {}

    def A_of(snp):
        if isinstance(snp, dict):
            ei, ew = snp['edge_index'], snp['edge_weight']
            ei = torch.as_tensor(ei, dtype=torch.long, device=device) if isinstance(ei, np.ndarray) else ei.to(device)
            ew = torch.as_tensor(ew, dtype=torch.float32, device=device) if isinstance(ew, np.ndarray) else ew.to(device)
        else:
            ei, ew = snp.edge_index.to(device), snp.edge_weight.to(device)
        return build_symmetric_normalized_adj(N, ei, ew)

    history = []
    for ep in range(1, epochs+1):
        model.train(); tot, n = 0.0, 0
        for day in train_days:
            snp = snapshots.get(day)
            if snp is None: continue
            pos = extract_pos_pairs_from_snapshot(snp)
            if not pos: continue

            A = A_of(snp).to(device)
            U, I = model(A)

            # cache embeddings for active users today
            if isinstance(snp, dict):
                ei_cpu = snp['edge_index']
                ei_cpu = torch.as_tensor(ei_cpu, dtype=torch.long) if isinstance(ei_cpu, np.ndarray) else ei_cpu
                active_users = torch.unique(ei_cpu[0]).tolist()
            else:
                active_users = torch.unique(snp.edge_index[0]).tolist()
            for u in active_users:
                last_valid_user_embedding[int(u)] = U[int(u)].detach().cpu()

            loss = bpr_loss_dns(U, I, pos, Ic, neg_ratio=neg_ratio, dns_pool_size=dns_pool_size, device=device)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss.item()); n += 1

        log = {'epoch': ep, 'train_loss': tot / max(n,1)}

        if val_days and ep % eval_every == 0:
            model.eval(); r, m = [], []
            for day in val_days:
                snp = snapshots.get(day)
                if snp is None: continue
                rv, mv = evaluate_day_backoff(model, snp, device=device, last_cache=last_valid_user_embedding, K=50)
                r.append(rv); m.append(mv)
            if r:
                log['val_recall@50'] = float(np.mean(r))
                log['val_mAP'] = float(np.mean(m))

        history.append(log); print(log)

    test_metrics = None
    if test_days:
        model.eval(); r, m = [], []
        for day in test_days:
            snp = snapshots.get(day)
            if snp is None: continue
            rv, mv = evaluate_day_backoff(model, snp, device=device, last_cache=last_valid_user_embedding, K=50)
            r.append(rv); m.append(mv)
        if r:
            test_metrics = {'test_recall@50': float(np.mean(r)), 'test_mAP': float(np.mean(m))}
            print(test_metrics)

    return model, history, test_metrics
