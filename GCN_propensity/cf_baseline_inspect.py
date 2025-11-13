#!/usr/bin/env python
"""
1. Sanity-check the Input Data
Verify row counts for raw / train / test.
Check training date range and whether chronological split worked.
Count unique counterparties and CUSIPs in train and test.
Check overlap:
How many test counterparties also appear in train?
How many test CUSIPs appear in train?
If overlap is low → recall will be structurally capped (cold start), not a model error.
2. Inspect Interaction Weights
Summaries of the computed weights: min, max, percentiles.
Look at distribution:
Are weights all tiny?
Are there extreme outliers (single user–CUSIP dominating)?
Investigate top-weight pairs:
Do they make sense in business terms?
Do they correspond to active/high-volume trading relationships?
3. Inspect User–Item Matrix Structure
Overall matrix shape and density (it will be sparse, but how sparse?).
Nonzero (NNZ) distribution:
Count NNZ per user → identify users with very few interactions.
Count NNZ per item → identify rarely traded CUSIPs.
If many users or CUSIPs have only 1 trade → CF model has limited capacity to learn good representations.
4. Inspect Learned Embeddings (ALS parameters)
Shapes of user_factors and item_factors.
Check for NaNs / Infs → indicates instability or bad scaling.
Norm distribution:
Are user and item embeddings collapsed to near-zero?
Are some norms huge while others tiny?
Healthy models have diverse but not extreme norm distributions.
5. Inspect Local Behavior (Client-Level & Bond-Level)
5.1 For a chosen CUSIP
Look at the most similar items:
Are neighbors plausible substitutes?
Same rating/sector/issuer?
5.2 For a chosen Counterparty
Compare:
top traded CUSIPs historically
top recommended CUSIPs
Should see nearby bonds, not random noise or pure popularity.
6. Inspect Global Recommendation Patterns
For many users, collect top-N recommendations.
Count how often each CUSIP appears.
If the same handful of CUSIPs appear in almost all users’ top lists →
model behaves too much like MostPop.
Could indicate:
bad weighting design
insufficient latent dimensionality
too strong regularization
very skewed training activity
7. Business-Level Sanity Tests
Pick a few known counterparties or CUSIPs:
Do recommendations reflect what their actual trading profile looks like?
Are recommended CUSIPs in the same issuer family, same duration bucket, same rating?
Do some recommendations reflect recent shifts in trading behavior?
8. Temporal Considerations
Even in ALS (static CF), ask:
Does the train/test separation introduce too much drift?
Are recommendation failures explained by CUSIPs not existing anymore or clients not active in training?
9. Pitfalls to Watch For
Column normalization issues (CUSIP strings mismatched).
Duplicate counterparties due to whitespace or inconsistent IDs.
Weight scaling that suppresses personalization (e.g., all weights ≈ 0.01).
Overly sparse clients causing noisy recommendations.
Time-decay weights making everything vanish for long histories.

Simple collaborative filtering baseline on trade data + debug/inspection tools.

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

import matplotlib.pyplot as plt


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
        agg["day_weight"] = 1.0

    elif mode == "freq":
        agg["day_weight"] = np.log1p(agg["trade_count"])

    elif mode == "notional":
        if "TRADE_AMT" not in train_df.columns:
            raise ValueError("notional mode requires TRADE_AMT column.")
        agg["day_weight"] = np.sqrt(agg["total_notional_day"].clip(lower=0.0).fillna(0.0))

    elif mode == "hybrid":
        if "TRADE_AMT" not in train_df.columns:
            raise ValueError("hybrid mode requires TRADE_AMT column.")
        agg["day_weight"] = (
            np.log1p(agg["trade_count"]) *
            np.sqrt(agg["total_notional_day"].clip(lower=0.0).fillna(0.0))
        )

    elif mode == "time_decay":
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

    # Normalize per user so each user's strongest link ≈ 1.0
    weights["weight"] = weights["weight"].astype(float)
    max_per_user = weights.groupby("COUNTERPARTY")["weight"].transform(lambda x: x.max().clip(lower=1e-9))
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
    test_df = test_df.copy()
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
        recs = model.recommend(uid,
                               train_mat[uid],
                               N=k,
                               filter_already_liked_items=True)
        rec_item_ids = {item_id for item_id, score in recs}

        hit_count = len(test_items & rec_item_ids)
        recall_u = hit_count / float(len(test_items))
        recalls.append(recall_u)

    if not recalls:
        return np.nan

    return float(np.mean(recalls))


# ------------------------
# Debug / inspection utilities
# ------------------------

def print_basic_data_stats(df, train_df, test_df):
    print("\n=== BASIC DATA STATS ===")
    print(f"Raw rows: {len(df)}")
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    print(f"Date range: {df['TRADEDATE'].min()} → {df['TRADEDATE'].max()}")

    print(f"Train unique COUNTERPARTY: {train_df.COUNTERPARTY.nunique()}")
    print(f"Train unique CUSIP       : {train_df.CUSIP.nunique()}")
    print(f"Test  unique COUNTERPARTY: {test_df.COUNTERPARTY.nunique()}")
    print(f"Test  unique CUSIP       : {test_df.CUSIP.nunique()}")

    train_users = set(train_df.COUNTERPARTY.unique())
    test_users = set(test_df.COUNTERPARTY.unique())
    train_items = set(train_df.CUSIP.unique())
    test_items = set(test_df.CUSIP.unique())

    print(f"Test users in train: {len(train_users & test_users)} / {len(test_users)}")
    print(f"Test CUSIPs in train: {len(train_items & test_items)} / {len(test_items)}")


def inspect_weights_distribution(weights_df):
    print("\n=== WEIGHT DISTRIBUTION ===")
    print(weights_df["weight"].describe())
    plt.figure()
    plt.hist(weights_df["weight"], bins=50)
    plt.yscale("log")
    plt.title("Interaction weight distribution (log y-scale)")
    plt.xlabel("weight")
    plt.ylabel("count (log)")
    plt.tight_layout()
    plt.show(block=False)


def inspect_matrix_sparsity(train_mat):
    print("\n=== MATRIX SPARSITY ===")
    num_users, num_items = train_mat.shape
    nnz = train_mat.nnz
    density = nnz / (num_users * num_items)
    print(f"Matrix shape: {train_mat.shape}")
    print(f"Non-zero entries: {nnz}")
    print(f"Density: {density:.6e}")

    user_nnz = np.diff(train_mat.indptr)
    item_nnz = np.diff(train_mat.tocsc().indptr)

    print("\nUser nnz stats:")
    print(pd.Series(user_nnz).describe())
    print("\nItem nnz stats:")
    print(pd.Series(item_nnz).describe())


def inspect_embedding_norms(model):
    print("\n=== EMBEDDING NORMS ===")
    user_f = model.user_factors
    item_f = model.item_factors

    print("user_factors shape:", user_f.shape)
    print("item_factors shape:", item_f.shape)

    print("Any NaNs in user_factors?", np.isnan(user_f).any())
    print("Any NaNs in item_factors?", np.isnan(item_f).any())
    print("Any Infs in user_factors?", np.isinf(user_f).any())
    print("Any Infs in item_factors?", np.isinf(item_f).any())

    user_norms = np.linalg.norm(user_f, axis=1)
    item_norms = np.linalg.norm(item_f, axis=1)

    print("\nUser norm stats:")
    print(pd.Series(user_norms).describe())
    print("\nItem norm stats:")
    print(pd.Series(item_norms).describe())

    plt.figure()
    plt.hist(user_norms, bins=50)
    plt.title("User embedding norms")
    plt.xlabel("L2 norm")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show(block=False)

    plt.figure()
    plt.hist(item_norms, bins=50)
    plt.title("Item embedding norms")
    plt.xlabel("L2 norm")
    plt.ylabel("count")
    plt.tight_layout()
    plt.show(block=False)


def inspect_global_recs(model, train_mat, item_index):
    print("\n=== GLOBAL REC PATTERNS (How many users see each item in top-K) ===")
    inv_item_index = {j: c for c, j in item_index.items()}
    global_recs = {}
    num_users = train_mat.shape[0]

    # To keep runtime reasonable, optionally subsample users if huge
    max_users_to_scan = min(num_users, 5000)

    for u_id in range(max_users_to_scan):
        recs = model.recommend(
            u_id,
            train_mat[u_id],
            N=20,
            filter_already_liked_items=True
        )

        # recs can be: [(item_id, score), ...] or rows with extra fields
        for r in recs:
            # r is something indexable; take the first element as item_id
            item_id = int(r[0])
            global_recs.setdefault(item_id, 0)
            global_recs[item_id] += 1

    from collections import Counter
    top_items = Counter(global_recs).most_common(20)
    print("Top items by how often they appear in top-20 lists:")
    for item_id, count in top_items:
        print(f"{inv_item_index[item_id]}  -> in {count} user top-20 lists")



def inspect_user_recs(model,
                      train_df,
                      train_mat,
                      user_index,
                      item_index,
                      user_name: str,
                      k: int = 20):
    print(f"\n=== USER-LEVEL INSPECTION: {user_name} ===")
    if user_name not in user_index:
        print("User not found in training set.")
        return

    inv_item_index = {j: c for c, j in item_index.items()}
    uid = user_index[user_name]

    # Historical top trades
    print("\nTop traded CUSIPs for this user in TRAIN:")
    user_trades = train_df[train_df["COUNTERPARTY"] == user_name]
    print(user_trades["CUSIP"].value_counts().head(20))

    # Recommended CUSIPs
    recs = model.recommend(uid,
                           train_mat[uid],
                           N=k,
                           filter_already_liked_items=True)

    print(f"\nTop {k} recommended CUSIPs for this user:")
    for item_id, score in recs:
        print(f"{inv_item_index[item_id]}  (score={score:.4f})")


def inspect_cusip_neighbors(model, item_index, cusip: str, k: int = 10):
    print(f"\n=== CUSIP-LEVEL INSPECTION: {cusip} ===")
    if cusip not in item_index:
        print("CUSIP not found in training set.")
        return

    inv_item_index = {j: c for c, j in item_index.items()}
    item_id = item_index[cusip]
    neighbors = model.similar_items(item_id, N=k)

    print(f"Most similar CUSIPs to {cusip}:")
    for nid, score in neighbors:
        print(f"{inv_item_index[nid]}  (similarity={score:.4f})")


def run_debug_suite(df, train_df, test_df,
                    weights_df, train_mat,
                    model, user_index, item_index,
                    args):
    """
    Run a set of diagnostics to build intuition and catch issues.
    """
    print_basic_data_stats(df, train_df, test_df)
    inspect_weights_distribution(weights_df)
    inspect_matrix_sparsity(train_mat)
    inspect_embedding_norms(model)
    inspect_global_recs(model, train_mat, item_index)

    if args.inspect_user:
        inspect_user_recs(model, train_df, train_mat, user_index, item_index,
                          user_name=args.inspect_user,
                          k=args.k)

    if args.inspect_cusip:
        inspect_cusip_neighbors(model, item_index,
                                cusip=args.inspect_cusip,
                                k=10)


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

    if args.inspect:
        run_debug_suite(df, train_df, test_df,
                        weights_df, train_mat,
                        model, user_index, item_index,
                        args)

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

    # Debug / inspection flags
    parser.add_argument("--inspect", action="store_true",
                        help="Run debug/inspection suite after training.")
    parser.add_argument("--inspect-user", type=str, default=None,
                        help="COUNTERPARTY ID to inspect recommendations for.")
    parser.add_argument("--inspect-cusip", type=str, default=None,
                        help="CUSIP to inspect nearest neighbors for.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
