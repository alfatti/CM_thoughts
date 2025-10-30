# LightGCN-W (causal rolling-window) training loop with BPR + DNS
# Minimal notebook version. Works without torch_geometric.

import math, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

# ---------- Sparse adjacency + model ----------

def build_symmetric_normalized_adj(num_nodes: int,
                                   edge_index: torch.Tensor,  # [2,E]
                                   edge_weight: torch.Tensor  # [E]
                                   ) -> torch.sparse.FloatTensor:
    if edge_index.numel() == 0:
        idx = torch.zeros((2,0), dtype=torch.long)
        val = torch.zeros((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, size=(num_nodes, num_nodes))
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
        self.emb = nn.Embedding(self.num_nodes, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.num_layers = num_layers

    def forward(self, A_norm: torch.sparse.FloatTensor):
        xk = self.emb.weight
        out = xk
        for k in range(1, self.num_layers + 1):
            xk = torch.sparse.mm(A_norm, xk)
            out = out + xk * (1.0 / (k + 1))
        return out[:self.num_users], out[self.num_users:]

# ---------- BPR with Dynamic Negative Sampling ----------

def bpr_loss_dns(user_embs: torch.Tensor, item_embs: torch.Tensor,
                 pos_pairs: List[Tuple[int,int]], num_items: int,
                 neg_ratio: int = 10, dns_pool_size: int = 50,
                 device: Optional[torch.device] = None) -> torch.Tensor:
    if len(pos_pairs) == 0:
        return torch.tensor(0.0, device=device if device else 'cpu', requires_grad=True)
    u_idx = torch.tensor([u for (u,i) in pos_pairs], dtype=torch.long, device=device)
    i_pos = torch.tensor([i for (u,i) in pos_pairs], dtype=torch.long, device=device)
    u_vec = user_embs[u_idx]            # [P,D]
    i_pos_vec = item_embs[i_pos]        # [P,D]

    P = u_idx.shape[0]
    pool = torch.randint(low=0, high=num_items, size=(P, dns_pool_size), device=device)
    pool[pool == i_pos.unsqueeze(1)] = (pool[pool == i_pos.unsqueeze(1)] + 1) % num_items
    items_pool = item_embs[pool]        # [P,K,D]
    scores_pool = torch.einsum('pd,pkd->pk', u_vec, items_pool)
    hard_idx = scores_pool.argmax(dim=1)
    i_neg_hard = pool[torch.arange(P, device=device), hard_idx]  # [P]

    i_neg_all = i_neg_hard.unsqueeze(1).repeat(1, neg_ratio).reshape(-1)
    u_idx_all = u_idx.unsqueeze(1).repeat(1, neg_ratio).reshape(-1)
    i_pos_all = i_pos.unsqueeze(1).repeat(1, neg_ratio).reshape(-1)
    u_vec_all = user_embs[u_idx_all]
    i_pos_vec_all = item_embs[i_pos_all]
    i_neg_vec_all = item_embs[i_neg_all]

    pos = (u_vec_all * i_pos_vec_all).sum(dim=1)
    neg = (u_vec_all * i_neg_vec_all).sum(dim=1)
    return -F.logsigmoid(pos - neg).mean()

# ---------- Snapshot helpers + metrics ----------

def extract_pos_pairs_from_snapshot(snapshot: Dict[str, Any]) -> List[Tuple[int,int]]:
    import numpy as _np
    if isinstance(snapshot, dict):
        ei, ew = snapshot['edge_index'], snapshot['edge_weight']
        if isinstance(ei, _np.ndarray): ei = torch.as_tensor(ei, dtype=torch.long)
        if isinstance(ew, _np.ndarray): ew = torch.as_tensor(ew, dtype=torch.float32)
        num_users = snapshot['num_users']
    else:
        ei, ew = snapshot.edge_index, snapshot.edge_weight
        num_users = snapshot.num_users
    if ei.numel() == 0: return []
    pairs = torch.stack([ei[0], ei[1]-num_users], dim=1).unique(dim=0)
    return [(int(u.item()), int(i.item())) for u,i in pairs]

def recall_at_k(ranked, relevant: set, k=50):
    topk = ranked[:k]; return sum(1 for it in topk if it in relevant) / max(len(relevant), 1)

def average_precision(ranked, relevant: set):
    if not relevant: return 0.0
    hits=0; s=0.0
    for r,it in enumerate(ranked,1):
        if it in relevant:
            hits+=1; s+=hits/r
    return s/len(relevant)

def evaluate_day(model, snapshot, device, K=50):
    import numpy as _np
    if isinstance(snapshot, dict):
        ei, ew = snapshot['edge_index'], snapshot['edge_weight']
        ei = torch.as_tensor(ei, dtype=torch.long, device=device) if isinstance(ei, _np.ndarray) else ei.to(device)
        ew = torch.as_tensor(ew, dtype=torch.float32, device=device) if isinstance(ew, _np.ndarray) else ew.to(device)
        Uc, Ic = snapshot['num_users'], snapshot['num_items']
        N = Uc + Ic
    else:
        ei, ew = snapshot.edge_index.to(device), snapshot.edge_weight.to(device)
        Uc, Ic, N = snapshot.num_users, snapshot.num_items, snapshot.num_nodes
    A = build_symmetric_normalized_adj(N, ei, ew).to(device)
    with torch.no_grad():
        U, I = model(A)
    rel = {}
    for u,i in extract_pos_pairs_from_snapshot(snapshot):
        rel.setdefault(u, set()).add(i)
    rvals, apvals = [], []
    for u,relevant in rel.items():
        scores = (U[u].unsqueeze(0) @ I.T).squeeze(0)
        ranked = torch.argsort(scores, descending=True).tolist()
        rvals.append(recall_at_k(ranked, relevant, k=K))
        apvals.append(average_precision(ranked, relevant))
    if not rvals: return 0.0, 0.0
    return float(np.mean(rvals)), float(np.mean(apvals))

# ---------- Training loop ----------

def train_lightgcn_w(snapshots: Dict[str,Any], train_days: List[str],
                     val_days: Optional[List[str]]=None, test_days: Optional[List[str]]=None,
                     emb_dim=64, num_layers=1, lr=1e-4, epochs=5,
                     neg_ratio=10, dns_pool_size=50, eval_every=1, device=None):
    meta = snapshots['__meta__']
    Uc, Ic = meta['num_users'], meta['num_items']
    N = Uc + Ic
    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = LightGCN(Uc, Ic, emb_dim=emb_dim, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def A_of(snp):
        import numpy as _np
        if isinstance(snp, dict):
            ei, ew = snp['edge_index'], snp['edge_weight']
            ei = torch.as_tensor(ei, dtype=torch.long, device=device) if isinstance(ei, _np.ndarray) else ei.to(device)
            ew = torch.as_tensor(ew, dtype=torch.float32, device=device) if isinstance(ew, _np.ndarray) else ew.to(device)
        else:
            ei, ew = snp.edge_index.to(device), snp.edge_weight.to(device)
        return build_symmetric_normalized_adj(N, ei, ew)

    hist=[]
    for ep in range(1, epochs+1):
        model.train(); tot=0.0; n=0
        for day in train_days:
            snp = snapshots.get(day); 
            if snp is None: continue
            pos = extract_pos_pairs_from_snapshot(snp)
            if not pos: continue
            A = A_of(snp)
            U,I = model(A)
            loss = bpr_loss_dns(U, I, pos, Ic, neg_ratio=neg_ratio, dns_pool_size=dns_pool_size, device=device)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss.item()); n+=1
        log={'epoch':ep,'train_loss': tot/max(n,1)}
        if val_days and ep % eval_every==0:
            model.eval(); r, m = [], []
            for day in val_days:
                snp = snapshots.get(day); 
                if snp is None: continue
                rv, mv = evaluate_day(model, snp, device=device, K=50)
                r.append(rv); m.append(mv)
            if r: log['val_recall@50']=float(np.mean(r)); log['val_mAP']=float(np.mean(m))
        hist.append(log); print(log)

    test_metrics=None
    if test_days:
        model.eval(); r, m = [], []
        for day in test_days:
            snp = snapshots.get(day); 
            if snp is None: continue
            rv, mv = evaluate_day(model, snp, device=device, K=50)
            r.append(rv); m.append(mv)
        if r: test_metrics={'test_recall@50':float(np.mean(r)),'test_mAP':float(np.mean(m))}; print(test_metrics)
    return model, hist, test_metrics

# ---------- DEMO: synthetic run (swap in your real trades DF below) ----------

# If you already ran the builder cell earlier, you have this file:
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("temporal_builder", "/mnt/data/build_pyg_temporal_snapshots.py")
temporal_builder = module_from_spec(spec); import sys
sys.modules["temporal_builder"]=temporal_builder; spec.loader.exec_module(temporal_builder)

# Synthetic demo (replace with your real df_trades)
dates = pd.to_datetime([
    '2025-09-01','2025-09-01','2025-09-02','2025-09-02','2025-09-02',
    '2025-09-03','2025-09-03','2025-09-04','2025-09-05','2025-09-06'
])
df_trades = pd.DataFrame({
    'client_id': ['u1','u1','u1','u2','u1','u2','u2','u3','u1','u2'],
    'cusip':     ['c1','c1','c2','c1','c1','c2','c3','c1','c3','c1'],
    'trade_date': dates,
    'qty':       [1,2,1,1,3,1,1,1,5,2],
    'maturity':  pd.to_datetime(['2026-01-01']*10)
})

# Build causal snapshots as in the paper: w=2 days, inverse time weighting
snapshots = temporal_builder.construct_daily_snapshots(
    df=df_trades,
    user_col='client_id',
    item_col='cusip',
    time_col='trade_date',
    qty_col='qty',
    maturity_col='maturity',
    window_days=2,
    calendar='D',
    time_weighting='inverse_delta',
    normalize=False,
    keep_empty_days=False,
    as_torch=True
)

# Day splits (toy)
all_days = sorted([d for d in snapshots.keys() if d != '__meta__'])
n = len(all_days)
n_train = max(1, int(0.6*n))
n_val   = max(1, int(0.2*n))
train_days = all_days[:n_train]
val_days   = all_days[n_train:n_train+n_val]
test_days  = all_days[n_train+n_val:]

# Train LightGCN-W
model, history, test_metrics = train_lightgcn_w(
    snapshots=snapshots,
    train_days=train_days,
    val_days=val_days,
    test_days=test_days,
    emb_dim=32,
    num_layers=1,    # try 2 or 3 like the ablation
    lr=1e-3,
    epochs=3,        # e.g., 40 with early stopping for real runs
    neg_ratio=5,
    dns_pool_size=20,
    eval_every=1
)
