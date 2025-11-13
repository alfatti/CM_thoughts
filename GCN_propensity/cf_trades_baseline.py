#!/usr/bin/env python
"""
cf_trades_baseline.py

Simple collaborative filtering baseline on trade data.

Expected columns in the CSV:
    - COUNTERPARTY (str)
    - CUSIP        (str)
    - TRADEDATE    (str/date)
    - TRADE_AMT    (float, optional)

Unit of time: DAY
"""

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

from implicit.als import AlternatingLeastSquares


# ------------------------
# Data loading & splitting
# ------------------------

def load_trades(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Parse dates and normalize to day
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["TRADEDATE", "COUNTERPARTY", "CUSIP"])
    return df


def train_test_split_by_day(df: pd.DataFrame, test_days: int = 30):
    """
    Chronological split by TRADEDATE.
    Last `test_days` form the test set, rest is train.
    """
    max_date = df["TRADEDATE"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)

    train_df = df[df["TRADEDATE"] < test_start].copy()
    test_df = df[df["TRADEDATE"] >= test_start].copy()

    return train_df, test_df


# ------------------------
# Weighting schemes
# ------------------------

def compute_weights(train_df: pd.DataFrame,
                    mode: str = "freq",
                    tau: float = 30.0) -> pd.DataFrame:
    """
    Compute (COUNTERPARTY, CUSIP, weight) for training interactions.

    mode ∈ {"binary", "freq", "notional", "hybrid", "time_decay"}.
    tau: time-decay horizon in days for "time_decay" mode.
    """

    df = train_df.copy()

    # Daily aggregation first (unit = day)
    # trade_count: number of trades in that day
    # total_notional_day: sum(notional) in that day (if available)
    agg_dict = {"COUNTERPARTY": "first", "CUSIP": "first"}
    if "TRADE_AMT" in df.columns:
        df["TRADE_AMT"] = pd.to_numeric(df["TRADE_AMT"], errors="coerce")
        agg = (
            df.groupby(["COUNTERPARTY", "CUSIP", "TRADEDATE"])
              .agg(trade_count=("CUSIP", "size"),
                   total_notional_day=("TRADE_AMT", "sum"))
              .reset_index()
        )
    else:
        agg = (
            df.groupby(["COUNTERPARTY", "CUSIP", "TRADEDATE"])
              .agg(trade_count=("CUSIP", "size"))
              .reset_index()
        )
        agg["total_notional_day"] = np.nan

    # Base per-day strength
    if mode == "binary":
        # Just "traded or not" each day
        agg["day_weight"] = 1.0

    elif mode == "freq":
        # More trades in a day → more weight; damped with log
        agg["day_weight"] = np.log1p(agg["trade_count"])

    elif mode == "notional":
        if "TRADE_AMT" not in train_df.columns:
            raise ValueError("notional mode requires TRADE_AMT column.")
        # sqrt to damp very large notionals
        agg["day_weight"] = np.sqrt(agg["total_notional_day"].clip(lower=0.0).fillna(0.0))

    elif mode == "hybrid":
        if "TRADE_AMT" not in train_df.columns:
            raise ValueError("hybrid mode requires TRADE_AMT column.")
        agg["day_weight"] = (
            np.log1p(agg["trade_count"]) *
            np.sqrt(agg["total_notional_day"].clip(lower=0.0).fillna(0.0))
        )

    elif mode == "time_decay":
        # Frequency + time-decay
        # More recent trades → higher weight
        max_date = agg["TRADEDATE"].max()
        days_ago = (max_date - agg["TRADEDATE"]).dt.days
        time_weight = np.exp(-days_ago / tau)
        agg["day_weight"] = np.log1p(agg["trade_count"]) * time_weight

    else:
        raise ValueError(f"Unknown weighting mode: {mode}")

    # Aggregate across the whole training period: sum daily weights
    weights = (
        agg.groupby(["COUNTERPARTY", "CUSIP"])["day_weight"]
           .sum()
           .reset_index()
           .rename(columns={"day_weight": "weight"})
    )

    # Normalize per user so each user's strongest link ≈ 1.0 (optional but useful)
    weights["weight"] = weights["weight"].astype(float)
    
    def _safe_max(x):
        m = x.max()
        return m if m > 1e-9 else 1e-9
    
    max_per_user = weights.groupby("COUNTERPARTY")["weight"].transform(_safe_max)
    weights["weight"] = weights["weight"] / max_per_user

    return weights


# ------------------------
# Matrix building
# ------------------------

def build_user_item_matrix(weights_df: pd.DataFrame):
    """
    Build (CSR matrix, user_index, item_index) from weighted interactions.

    user_index: dict raw_user_id -> row index
    item_index: dict raw_item_id -> col index
    """
    users = weights_df["COUNTERPARTY"].astype("category")
    items = weights_df["CUSIP"].astype("category")

    weights = weights_df["weight"].values.astype(np.float32)

    user_ids = users.cat.codes.values
    item_ids = items.cat.codes.values

    num_users = users.cat.categories.size
    num_items = items.cat.categories.size

    mat = sp.csr_matrix((weights, (user_ids, item_ids)), shape=(num_users, num_items))

    user_index = {u: i for i, u in enumerate(users.cat.categories)}
    item_index = {c: j for j, c in enumerate(items.cat.categories)}

    return mat, user_index, item_index


# ------------------------
# Evaluation (Recall@K)
# ------------------------

def evaluate_recall_at_k(model,
                         train_mat: sp.csr_matrix,
                         test_df: pd.DataFrame,
                         user_index: dict,
                         item_index: dict,
                         k: int = 20) -> float:
    """
    Simple Recall@K:
    - For each user with at least one test trade:
        recall_u = (# test CUSIPs in top-K recs) / (# unique CUSIPs in test for that user)
    - Return average over users.
    """
    # Build user -> set of test item indices
    test_df = test_df.copy()
    # Filter to users/items seen in training
    test_df = test_df[test_df["COUNTERPARTY"].isin(user_index.keys()) &
                      test_df["CUSIP"].isin(item_index.keys())]

    if test_df.empty:
        print("No overlapping users/items between train and test. Recall cannot be computed.")
        return np.nan

    user_test_items = {}
    for u, g in test_df.groupby("COUNTERPARTY"):
        idxs = {item_index[c] for c in g["CUSIP"].unique() if c in item_index}
        if idxs:
            user_test_items[u] = idxs

    recalls = []
    for u, test_items in user_test_items.items():
        uid = user_index[u]
        # Recommend top-K items for this user
        # filter_already_liked_items=True to avoid trivial training items
        # Recommend top-K items for this user
        # NOTE: different implicit versions return either:
        #  (i) list of (item_id, score), or
        # (ii) tuple (ids_array, scores_array)
        recs = model.recommend(
            uid,
            train_mat[uid],
            N=k,
            filter_already_liked_items=True,
            # uncomment next line if you prefer the (ids, scores) form explicitly:
            # return_scores=True,
        )
        
        # --- Normalize to a set of item ids regardless of return shape ---
        def _to_id_set(recs_obj):
            # case A: tuple: (ids_array, scores_array)
            if isinstance(recs_obj, tuple) and len(recs_obj) == 2:
                ids, _scores = recs_obj
                return set(map(int, np.asarray(ids).ravel()))
            # case B: ndarray shape (N,2): [[id, score], ...]
            if isinstance(recs_obj, np.ndarray) and recs_obj.ndim == 2 and recs_obj.shape[1] >= 1:
                return set(map(int, recs_obj[:, 0].ravel()))
            # case C: list of (id, score) tuples
            try:
                first = recs_obj[0]
                if isinstance(first, (list, tuple, np.ndarray)) and len(first) >= 1:
                    return {int(r[0]) for r in recs_obj}
            except Exception:
                pass
            # Fallback: assume it's a flat list/array of ids
            return set(map(int, np.asarray(recs_obj).ravel()))
        
        rec_item_ids = _to_id_set(recs)
        hit_count = len(test_items & rec_item_ids)
        recall_u = hit_count / float(len(test_items))
        recalls.append(recall_u)

    if not recalls:
        return np.nan

    return float(np.mean(recalls))


# ------------------------
# Main training pipeline
# ------------------------

def run_pipeline(args):
    print(f"Loading trades from {args.input} ...")
    df = load_trades(args.input)
    print(f"Total rows: {len(df)}; date range: {df['TRADEDATE'].min()} → {df['TRADEDATE'].max()}")

    print(f"Splitting train/test with last {args.test_days} days as test ...")
    train_df, test_df = train_test_split_by_day(df, test_days=args.test_days)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    print(f"Computing weights with mode='{args.weight_mode}' ...")
    weights_df = compute_weights(train_df, mode=args.weight_mode, tau=args.tau)
    print(f"Training interactions (unique COUNTERPARTY–CUSIP pairs): {len(weights_df)}")

    print("Building user–item matrix ...")
    train_mat, user_index, item_index = build_user_item_matrix(weights_df)
    print(f"Matrix shape: users={train_mat.shape[0]}, items={train_mat.shape[1]}")

    print("Fitting ALS model (implicit CF) ...")
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.reg,
        iterations=args.iterations
    )
    # implicit ALS expects positive confidence values; we can just pass the weights
    model.fit(train_mat)

    print(f"Evaluating Recall@{args.k} on test set ...")
    recall_k = evaluate_recall_at_k(
        model,
        train_mat,
        test_df,
        user_index,
        item_index,
        k=args.k
    )
    print(f"Recall@{args.k}: {recall_k:.4f}" if not np.isnan(recall_k) else "Recall@K: NaN (no overlap)")

    # You could also return the model and mappings if used as a module
    return model, user_index, item_index


# ------------------------
# CLI
# ------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Collaborative filtering baseline on trade data (unit = day).")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to CSV file with trade data.")
    parser.add_argument("--weight-mode", type=str, default="freq",
                        choices=["binary", "freq", "notional", "hybrid", "time_decay"],
                        help="Weighting scheme for implicit feedback.")
    parser.add_argument("--tau", type=float, default=30.0,
                        help="Time-decay horizon (days) for time_decay mode.")
    parser.add_argument("--test-days", type=int, default=30,
                        help="Number of most recent days to use as test set.")
    parser.add_argument("--factors", type=int, default=64,
                        help="Number of latent factors for ALS.")
    parser.add_argument("--reg", type=float, default=0.1,
                        help="Regularization parameter for ALS.")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of ALS iterations.")
    parser.add_argument("--k", type=int, default=20,
                        help="K for Recall@K.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
