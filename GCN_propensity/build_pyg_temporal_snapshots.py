
"""
build_pyg_temporal_snapshots.py

Construct causal LightGCN-ready daily snapshots from time-stamped trade data.
- Input: Pandas DataFrame with columns:
    - user_col:     string (e.g., 'client_id')
    - item_col:     string (e.g., 'cusip')
    - time_col:     datetime-like (e.g., 'trade_date')
    - qty_col:      optional numeric, used as interaction magnitude (default: 1 per row)
    - notional_col: optional numeric, used as interaction magnitude (ignored if qty provided)
    - maturity_col: optional datetime-like for items to exclude after maturity
- Output: dict[date_str] -> PyG Data (if installed) or dict with tensors

Design choices
--------------
1) Causality: For each target day t, we aggregate interactions from [t - w, t)
2) Interest strength: Multiple interactions within the window aggregate into the edge weight.
   If time_weighting='inverse_delta', each event is weighted by 1 / max(delta_days, 1).
3) Bipartite graph: user nodes (0..U-1), item nodes offset by U (U..U+I-1).
4) Normalization: We expose raw weights plus degree-based symmetric norm if requested.
5) Negative sampling: a helper returns valid negatives for a user on day t, excluding matured items.

Author: ChatGPT
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List, Any

# Optional imports (torch, torch_geometric). We only import when needed.
try:
    import torch
except Exception:
    torch = None

def _to_torch(x, dtype=None):
    if torch is None:
        return x
    return torch.as_tensor(x, dtype=dtype)

def build_id_mappings(
    df: pd.DataFrame,
    user_col: str,
    item_col: str
) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    users = df[user_col].astype('category')
    items = df[item_col].astype('category')
    user2idx = {u: i for i, u in enumerate(users.cat.categories.tolist())}
    item2idx = {i: j for j, i in enumerate(items.cat.categories.tolist())}
    return user2idx, item2idx

def _ensure_dt(s: pd.Series) -> pd.Series:
    if not np.issubdtype(s.dtype, np.datetime64):
        return pd.to_datetime(s)
    return s

def _normalize_weights_symmetric(
    num_users: int,
    num_items: int,
    edge_index: np.ndarray,
    edge_weight: np.ndarray
) -> np.ndarray:
    """
    Symmetric degree normalization: w_ij / sqrt(deg_u * deg_i)
    where degrees computed on bipartite edges (sum of weights).
    """
    # edge_index shape (2, E): [src_row; dst_row] in global index space
    src = edge_index[0]
    dst = edge_index[1]
    # compute degree for users and items separately
    # users: 0..U-1; items: U..U+I-1
    deg = np.zeros(num_users + num_items, dtype=np.float64)
    np.add.at(deg, src, edge_weight)
    np.add.at(deg, dst, edge_weight)
    # degrees for a bipartite edge: use product of node degrees
    denom = np.sqrt(np.maximum(deg[src], 1e-12) * np.maximum(deg[dst], 1e-12))
    return edge_weight / denom

def construct_daily_snapshots(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    time_col: str,
    qty_col: Optional[str] = None,
    notional_col: Optional[str] = None,
    maturity_col: Optional[str] = None,
    window_days: int = 2,
    calendar: str = 'D',
    time_weighting: str = 'uniform',   # 'uniform' or 'inverse_delta'
    normalize: bool = False,
    keep_empty_days: bool = False,
    as_torch: bool = True,
) -> Dict[str, Any]:
    """
    Build causal rolling-window bipartite graphs for each target day.

    Parameters
    ----------
    df : DataFrame
        Trades with user, item, time and optional magnitude columns.
    user_col, item_col, time_col : str
        Column names.
    qty_col, notional_col : Optional[str]
        If provided, used as magnitude; qty_col takes precedence.
    maturity_col : Optional[str]
        If provided, exclude items whose maturity < target_day.
    window_days : int
        Window width w (days), using [t-w, t) to predict day t.
    calendar : str
        Pandas offset alias for day buckets (e.g., 'D' daily, 'B' business day).
    time_weighting : str
        'uniform' => each event contributes 1 (or its magnitude).
        'inverse_delta' => each event contributes magnitude / max(delta_days, 1).
    normalize : bool
        If True, apply symmetric degree normalization to edge weights.
    keep_empty_days : bool
        If True, include days with no interactions in the output (empty graphs).
    as_torch : bool
        If True and torch is available, tensors are torch tensors; else numpy.

    Returns
    -------
    snapshots : dict[str, object]
        Each value is either:
         - a torch_geometric.data.Data object (if torch_geometric available), or
         - a dict with fields: {
               'edge_index', 'edge_weight', 'num_nodes', 'num_users', 'num_items',
               'user_index', 'item_index'
           }
    """
    assert window_days >= 1, "window_days must be >= 1"
    df = df.copy()
    # coerce dtypes
    df[time_col] = _ensure_dt(df[time_col])
    if maturity_col is not None:
        df[maturity_col] = _ensure_dt(df[maturity_col])

    # choose magnitude column
    mag_col = None
    if qty_col is not None and qty_col in df.columns:
        mag_col = qty_col
    elif notional_col is not None and notional_col in df.columns:
        mag_col = notional_col

    # floor timestamps to calendar buckets (e.g., daily)
    df['__day__'] = df[time_col].dt.to_period(calendar).dt.start_time

    # build ID mappings (global over dataset horizon)
    user2idx, item2idx = build_id_mappings(df, user_col, item_col)
    num_users = len(user2idx)
    num_items = len(item2idx)

    # pre-map ids to indices
    df['__u__'] = df[user_col].map(user2idx)
    df['__i__'] = df[item_col].map(item2idx)

    # make a day index
    all_days = pd.period_range(df['__day__'].min(), df['__day__'].max(), freq=calendar).to_timestamp()
    # group by day for fast slicing
    by_day = {d: g.drop(columns=[user_col, item_col]) for d, g in df.groupby('__day__')}

    snapshots: Dict[str, Any] = {}

    # Helper to build one day's graph
    def build_one_day(target_day: pd.Timestamp) -> Any:
        # causal window [t - w, t)
        start = (target_day - pd.Timedelta(days=window_days))
        window_days_list = [d for d in all_days if (d >= start) and (d < target_day)]
        if len(window_days_list) == 0:
            # No history available
            if keep_empty_days:
                return _empty_graph(num_users, num_items, as_torch=as_torch)
            return None

        # concat events in the window
        frames = [by_day[d] for d in window_days_list if d in by_day]
        if len(frames) == 0:
            if keep_empty_days:
                return _empty_graph(num_users, num_items, as_torch=as_torch)
            return None
        hist = pd.concat(frames, axis=0, ignore_index=True)

        # Optionally exclude matured items (maturity < target_day)
        if maturity_col is not None and maturity_col in df.columns:
            # We need item-level maturity; if given per row, filter rows older than maturity
            hist = hist[(hist[maturity_col].isna()) | (hist[maturity_col] >= target_day)]

        # compute delta days (for inverse time weighting)
        if time_weighting == 'inverse_delta':
            # use the original timestamp (not floored), so compute from time_col
            hist['__delta__'] = (target_day - hist[time_col]).dt.days.clip(lower=1)
        else:
            hist['__delta__'] = 1

        # magnitude per row
        if mag_col is not None:
            hist['__mag__'] = hist[mag_col].astype(float).fillna(0.0)
        else:
            hist['__mag__'] = 1.0

        if time_weighting == 'inverse_delta':
            hist['__w__'] = hist['__mag__'] / hist['__delta__']
        else:
            hist['__w__'] = hist['__mag__']

        # aggregate per (u, i)
        agg = hist.groupby(['__u__', '__i__'], as_index=False)['__w__'].sum()

        if agg.empty:
            if keep_empty_days:
                return _empty_graph(num_users, num_items, as_torch=as_torch)
            return None

        # build edge_index in global node space: users [0..U-1], items [U..U+I-1]
        u_idx = agg['__u__'].to_numpy(dtype=np.int64)
        i_idx = agg['__i__'].to_numpy(dtype=np.int64)
        item_offset = num_users
        src = u_idx
        dst = i_idx + item_offset

        edge_index = np.stack([src, dst], axis=0)  # shape (2, E)
        edge_weight = agg['__w__'].to_numpy(dtype=np.float32)

        if normalize:
            edge_weight = _normalize_weights_symmetric(num_users, num_items, edge_index, edge_weight).astype(np.float32)

        # Package as PyG Data if available
        data_obj = _package_graph(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_users=num_users,
            num_items=num_items,
            as_torch=as_torch
        )
        return data_obj

    # main loop across target days
    for t in all_days:
        obj = build_one_day(t)
        if obj is not None:
            snapshots[t.strftime('%Y-%m-%d')] = obj
        elif keep_empty_days:
            snapshots[t.strftime('%Y-%m-%d')] = _empty_graph(num_users, num_items, as_torch=as_torch)

    meta = dict(
        user2idx=user2idx,
        item2idx=item2idx,
        num_users=num_users,
        num_items=num_items,
        calendar=calendar,
        window_days=window_days,
        time_weighting=time_weighting,
        normalize=normalize
    )
    snapshots['__meta__'] = meta
    return snapshots

def _empty_graph(num_users: int, num_items: int, as_torch: bool = True):
    edge_index = np.zeros((2, 0), dtype=np.int64)
    edge_weight = np.zeros((0,), dtype=np.float32)
    return _package_graph(edge_index, edge_weight, num_users, num_items, as_torch=as_torch)

def _package_graph(edge_index: np.ndarray, edge_weight: np.ndarray,
                   num_users: int, num_items: int, as_torch: bool = True) -> Any:
    """
    Try to return torch_geometric.data.Data if available; else a lightweight dict.
    """
    # Convert to desired tensor type
    if as_torch and (torch is not None):
        ei = _to_torch(edge_index, dtype=torch.long)
        ew = _to_torch(edge_weight, dtype=torch.float32)
    else:
        ei, ew = edge_index, edge_weight

    # Attempt to create a PyG Data object
    try:
        from torch_geometric.data import Data
        num_nodes = num_users + num_items
        data = Data(
            edge_index=ei,
            edge_weight=ew,
            num_nodes=num_nodes
        )
        # Help the model by storing meta
        data.num_users = num_users
        data.num_items = num_items
        # boolean mask for user/item nodes (useful in models)
        if as_torch and (torch is not None):
            user_mask = torch.zeros(num_nodes, dtype=torch.bool)
            user_mask[:num_users] = True
            item_mask = ~user_mask
        else:
            user_mask = np.concatenate([np.ones(num_users, dtype=bool), np.zeros(num_items, dtype=bool)])
            item_mask = ~user_mask
        data.user_mask = user_mask
        data.item_mask = item_mask
        return data
    except Exception:
        # Fallback to plain dict
        return dict(
            edge_index=ei,
            edge_weight=ew,
            num_nodes=(num_users + num_items),
            num_users=num_users,
            num_items=num_items
        )

def valid_negative_items_on_day(
    snapshots: Dict[str, Any],
    day_str: str,
    user_idx: int,
    interacted_item_idxs: Optional[List[int]] = None
) -> np.ndarray:
    """
    Returns candidate negative item indices for a given user on a given day.
    Excludes items that are connected to the user that day (if provided).

    Note: This function uses only the snapshot's num_items meta. For real-world
    use, also exclude matured/delisted items as of that day (pass a prefiltered pool).
    """
    if '__meta__' not in snapshots:
        raise ValueError("Snapshots missing '__meta__' entry.")
    num_items = snapshots['__meta__']['num_items']
    all_items = np.arange(num_items, dtype=np.int64)
    if interacted_item_idxs is None:
        return all_items
    mask = np.ones(num_items, dtype=bool)
    mask[np.asarray(interacted_item_idxs, dtype=np.int64)] = False
    return all_items[mask]
