#!/usr/bin/env python
"""
cf_tuning.py

Hyperparameter tuning for implicit ALS collaborative filtering
on trade data, using different weighting schemes.

Designed to be gentle on a developer laptop:
- modest hyperparameter grid
- optional user subsampling in evaluation
"""

import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

from implicit.als import AlternatingLeastSquares
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ----------------------------------------------------
# Core utilities (mirroring your baseline CF script)
# ----------------------------------------------------

def load_trades(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["TRADEDATE", "COUNTERPARTY", "CUSIP"])
    return df


def train_test_split_by_day(df: pd.DataFrame, test_days: int = 30):
    max_date = df["TRADEDATE"].max()
    test_start = max_date - pd.Timedelta(days=test_days - 1)

    train_df = df[df["TRADEDATE"] < test_start].copy()
    test_df = df[df["TRADEDATE"] >= test_start].copy()
    return train_df, test_df


def compute_weights(train_df: pd.DataFrame,
                    mode: str = "freq",
                    tau: float = 30.0) -> pd.DataFrame:
    """
    Compute (COUNTERPARTY, CUSIP, weight) for training interactions.

    mode ∈ {"binary", "freq", "notional", "hybrid", "time_decay"}.
    """
    df = train_df.copy()

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

    weights = (
        agg.groupby(["COUNTERPARTY", "CUSIP"])["day_weight"]
           .sum()
           .reset_index()
           .rename(columns={"day_weight": "weight"})
    )

    # Normalize per user (keeps things numerically tame)
    weights["weight"] = weights["weight"].astype(float)
    max_per_user = weights.groupby("COUNTERPARTY")["weight"].transform(
        lambda x: x.max().clip(lower=1e-9)
    )
    weights["weight"] = weights["weight"] / max_per_user
    return weights


def build_user_item_matrix(weights_df: pd.DataFrame):
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


def evaluate_recall_at_k_fast(model,
                              train_mat: sp.csr_matrix,
                              test_df: pd.DataFrame,
                              user_index: dict,
                              item_index: dict,
                              k: int = 20,
                              max_eval_users: int | None = None) -> float:
    """
    Recall@K with optional user subsampling to save time.
    """
    test_df = test_df.copy()
    test_df = test_df[test_df["COUNTERPARTY"].isin(user_index.keys()) &
                      test_df["CUSIP"].isin(item_index.keys())]

    if test_df.empty:
        return np.nan

    # Build user -> set(test_items)
    user_test_items = {}
    for u, g in test_df.groupby("COUNTERPARTY"):
        idxs = {item_index[c] for c in g["CUSIP"].unique() if c in item_index}
        if idxs:
            user_test_items[u] = idxs

    recalls = []
    for idx, (u, test_items) in enumerate(user_test_items.items()):
        if max_eval_users is not None and idx >= max_eval_users:
            break

        uid = user_index[u]
        recs = model.recommend(
            uid,
            train_mat[uid],
            N=k,
            filter_already_liked_items=True,
        )
        rec_item_ids = {item_id for item_id, score in recs}
        hit_count = len(test_items & rec_item_ids)
        recall_u = hit_count / float(len(test_items))
        recalls.append(recall_u)

    if not recalls:
        return np.nan

    return float(np.mean(recalls))


# ----------------------------------------------------
# Hyperparameter tuning
# ----------------------------------------------------

def tune_als_hyperparams(
    csv_path: str,
    weight_mode: str = "freq",
    test_days: int = 30,
    k: int = 20,
    tau: float = 30.0,
    max_eval_users: int | None = 500,  # subsample users for speed
):
    """
    Run a small grid search over ALS hyperparameters for a given weighting scheme.
    Returns (best_model, best_params, results_df).
    """

    print(f"Loading trades from {csv_path} ...")
    df = load_trades(csv_path)
    print(f"Rows: {len(df)}, date range: {df['TRADEDATE'].min()} → {df['TRADEDATE'].max()}")

    print(f"Splitting train/test (last {test_days} days as test) ...")
    train_df, test_df = train_test_split_by_day(df, test_days=test_days)
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    print(f"Computing weights with mode='{weight_mode}' ...")
    weights_df = compute_weights(train_df, mode=weight_mode, tau=tau)
    print(f"Unique (COUNTERPARTY, CUSIP) pairs in train: {len(weights_df)}")

    print("Building user–item matrix ...")
    train_mat, user_index, item_index = build_user_item_matrix(weights_df)
    num_users, num_items = train_mat.shape
    print(f"Matrix shape: users={num_users}, items={num_items}, nnz={train_mat.nnz}")

    # Hyperparameter grid (kept small for dev laptop)
    factors_grid = [32, 64]
    reg_grid = [0.01, 0.05, 0.1]
    iter_grid = [10, 20]

    results = []
    combo_id = 0

    print("\nStarting hyperparameter search ...")
    for factors in factors_grid:
        for reg in reg_grid:
            for iters in iter_grid:
                combo_id += 1
                print(f"[{combo_id}] factors={factors}, reg={reg}, iters={iters} ...", end=" ")

                model = AlternatingLeastSquares(
                    factors=factors,
                    regularization=reg,
                    iterations=iters,
                    use_gpu=False  # be nice to laptop
                )
                model.fit(train_mat)

                recall = evaluate_recall_at_k_fast(
                    model,
                    train_mat,
                    test_df,
                    user_index,
                    item_index,
                    k=k,
                    max_eval_users=max_eval_users,
                )

                print(f"Recall@{k}={recall:.4f}" if not np.isnan(recall) else f"Recall@{k}=NaN")

                results.append({
                    "factors": factors,
                    "regularization": reg,
                    "iterations": iters,
                    "recall_at_k": recall,
                })

    results_df = pd.DataFrame(results)
    # Sort best first (higher recall better)
    results_df = results_df.sort_values("recall_at_k", ascending=False)
    print("\n=== Hyperparameter search results (top 10) ===")
    print(results_df.head(10).to_string(index=False))

    # Best combo
    if results_df["recall_at_k"].notna().any():
        best_row = results_df.iloc[0]
        best_params = {
            "factors": int(best_row["factors"]),
            "regularization": float(best_row["regularization"]),
            "iterations": int(best_row["iterations"]),
        }
        print("\nBest params:", best_params)

        # Re-train best model on full train_mat
        best_model = AlternatingLeastSquares(
            factors=best_params["factors"],
            regularization=best_params["regularization"],
            iterations=best_params["iterations"],
            use_gpu=False
        )
        best_model.fit(train_mat)

        return best_model, best_params, results_df
    else:
        print("\nAll recalls are NaN (likely no train-test overlap).")
        return None, None, results_df


# ----------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for ALS CF on trade data.")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to CSV file with trade data.")
    parser.add_argument("--weight-mode", type=str, default="freq",
                        choices=["binary", "freq", "notional", "hybrid", "time_decay"],
                        help="Weighting scheme.")
    parser.add_argument("--test-days", type=int, default=30,
                        help="Number of last days to use as test.")
    parser.add_argument("--k", type=int, default=20,
                        help="K for Recall@K.")
    parser.add_argument("--tau", type=float, default=30.0,
                        help="Time-decay horizon for time_decay mode.")
    parser.add_argument("--max-eval-users", type=int, default=500,
                        help="Max number of users to evaluate for each combo (for speed).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune_als_hyperparams(
        csv_path=args.input,
        weight_mode=args.weight_mode,
        test_days=args.test_days,
        k=args.k,
        tau=args.tau,
        max_eval_users=args.max_eval_users,
    )
