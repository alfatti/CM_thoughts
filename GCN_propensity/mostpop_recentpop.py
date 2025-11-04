#!/usr/bin/env python
"""
Rolling-window backtest for MostPop and RecentPop baselines
on a trading dataset with ACCOUNTID, CUSIP, TRADE_DATETIME.

Usage:
    python backtest_popularity.py path/to/trades.csv \
        --window_size 5 \
        --k 50 \
        --test_start 2024-01-01

Defaults:
    window_size (RecentPop): 5 days
    k (top-K): 50
    test_start: 80% into the sorted unique dates (automatic split)
"""

import argparse
from datetime import timedelta

import pandas as pd
import numpy as np


def prepare_data(df,
                 account_col="ACCOUNTID",
                 cusip_col="CUSIP",
                 datetime_col="TRADE_DATETIME"):
    """
    Normalize datetime and create a TRADE_DATE column (midnight Timestamp).
    Filters out rows with missing critical fields.
    """
    # Drop rows missing any key column
    df = df.dropna(subset=[account_col, cusip_col, datetime_col])

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["TRADE_DATE"] = df[datetime_col].dt.floor("D")

    # Sort for sanity
    df = df.sort_values("TRADE_DATE").reset_index(drop=True)
    return df


def compute_popularity(history_df, cusip_col="CUSIP"):
    """
    Compute item popularity (counts) from a history subset.
    Returns a pandas Series: index=CUSIP, values=count (sorted descending).
    """
    if history_df.empty:
        return pd.Series(dtype=float)

    pop = history_df.groupby(cusip_col).size()
    pop = pop.sort_values(ascending=False)
    return pop


def evaluate_for_day(
    df_day,
    history_df,
    recent_history_df,
    account_col="ACCOUNTID",
    cusip_col="CUSIP",
    k=50,
):
    """
    Evaluate MostPop and RecentPop for a single day.

    df_day: trades on day t
    history_df: trades strictly before t
    recent_history_df: trades in (t - w, t)

    Returns:
        dict with sums of metrics and counts, to be aggregated later.
    """

    # Popularity distributions
    mostpop = compute_popularity(history_df, cusip_col=cusip_col)
    recentpop = compute_popularity(recent_history_df, cusip_col=cusip_col)

    # If no history, model can't rank anything: skip day
    has_mostpop = not mostpop.empty
    has_recentpop = not recentpop.empty

    # Prepare rankings
    if has_mostpop:
        mostpop_rank = list(mostpop.index)
    else:
        mostpop_rank = []

    if has_recentpop:
        recentpop_rank = list(recentpop.index)
    else:
        recentpop_rank = []

    # Metrics accumulators
    stats = {
        "mostpop_hit_sum": 0.0,
        "mostpop_recall_sum": 0.0,
        "mostpop_count": 0,

        "recentpop_hit_sum": 0.0,
        "recentpop_recall_sum": 0.0,
        "recentpop_count": 0,
    }

    # For each account active today
    for acc, df_acc in df_day.groupby(account_col):
        positives = set(df_acc[cusip_col].unique())
        if len(positives) == 0:
            continue

        # MOSTPOP evaluation
        if has_mostpop:
            top_k_most = mostpop_rank[:k]
            hit_items = positives.intersection(top_k_most)
            hit = 1.0 if len(hit_items) > 0 else 0.0
            recall = len(hit_items) / float(len(positives))

            stats["mostpop_hit_sum"] += hit
            stats["mostpop_recall_sum"] += recall
            stats["mostpop_count"] += 1

        # RECENTPOP evaluation
        if has_recentpop:
            top_k_recent = recentpop_rank[:k]
            hit_items_r = positives.intersection(top_k_recent)
            hit_r = 1.0 if len(hit_items_r) > 0 else 0.0
            recall_r = len(hit_items_r) / float(len(positives))

            stats["recentpop_hit_sum"] += hit_r
            stats["recentpop_recall_sum"] += recall_r
            stats["recentpop_count"] += 1

    return stats


def backtest_popularity(
    df,
    account_col="ACCOUNTID",
    cusip_col="CUSIP",
    datetime_col="TRADE_DATETIME",
    window_size=5,
    k=50,
    test_start=None,
):
    """
    Main backtest loop:
    - For each day t in the test period:
        * History = trades before t  (for MostPop)
        * Recent history = trades in (t - window_size, t) (for RecentPop)
        * Evaluate per-account Hit@K and Recall@K
    - Aggregate across all (t, account).

    Returns:
        summary dict with average metrics.
    """

    df = prepare_data(df, account_col=account_col,
                      cusip_col=cusip_col, datetime_col=datetime_col)

    all_dates = df["TRADE_DATE"].drop_duplicates().sort_values().tolist()
    if len(all_dates) < 2:
        raise ValueError("Not enough distinct dates to run a backtest.")

    # Determine test start
    if test_start is not None:
        test_start = pd.to_datetime(test_start).floor("D")
    else:
        # By default: start test at ~80% into the sample
        split_idx = int(0.8 * len(all_dates))
        test_start = all_dates[split_idx]

    print(f"Number of distinct trading days: {len(all_dates)}")
    print(f"Test period starts at: {test_start.date()}")

    # For convenience, keep pre-split DataFrames
    # (We still filter by date each iteration)
    stats_global = {
        "mostpop_hit_sum": 0.0,
        "mostpop_recall_sum": 0.0,
        "mostpop_count": 0,
        "recentpop_hit_sum": 0.0,
        "recentpop_recall_sum": 0.0,
        "recentpop_count": 0,
    }

    for current_date in all_dates:
        if current_date < test_start:
            continue

        # Today's trades
        df_day = df[df["TRADE_DATE"] == current_date]
        if df_day.empty:
            continue

        # History for MostPop: all trades strictly before current_date
        history_mask = df["TRADE_DATE"] < current_date
        history_df = df[history_mask]

        # If there is absolutely no prior history, skip this day
        if history_df.empty:
            continue

        # Recent history for RecentPop: (t - window, t)
        start_recent = current_date - timedelta(days=window_size)
        recent_mask = (df["TRADE_DATE"] >= start_recent) & (
            df["TRADE_DATE"] < current_date
        )
        recent_history_df = df[recent_mask]

        day_stats = evaluate_for_day(
            df_day=df_day,
            history_df=history_df,
            recent_history_df=recent_history_df,
            account_col=account_col,
            cusip_col=cusip_col,
            k=k,
        )

        # Aggregate
        for key in stats_global:
            stats_global[key] += day_stats[key]

    # Compute global averages
    results = {}

    if stats_global["mostpop_count"] > 0:
        results["MostPop_Hit@{}".format(k)] = (
            stats_global["mostpop_hit_sum"] / stats_global["mostpop_count"]
        )
        results["MostPop_Recall@{}".format(k)] = (
            stats_global["mostpop_recall_sum"] / stats_global["mostpop_count"]
        )
    else:
        results["MostPop_Hit@{}".format(k)] = np.nan
        results["MostPop_Recall@{}".format(k)] = np.nan

    if stats_global["recentpop_count"] > 0:
        results["RecentPop_Hit@{}".format(k)] = (
            stats_global["recentpop_hit_sum"] / stats_global["recentpop_count"]
        )
        results["RecentPop_Recall@{}".format(k)] = (
            stats_global["recentpop_recall_sum"] / stats_global["recentpop_count"]
        )
    else:
        results["RecentPop_Hit@{}".format(k)] = np.nan
        results["RecentPop_Recall@{}".format(k)] = np.nan

    results["Num_eval_user_days_MostPop"] = stats_global["mostpop_count"]
    results["Num_eval_user_days_RecentPop"] = stats_global["recentpop_count"]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Rolling-window backtest for MostPop and RecentPop baselines."
    )
    parser.add_argument("csv_path", type=str, help="Path to trades CSV file.")
    parser.add_argument(
        "--account_col", type=str, default="ACCOUNTID",
        help="Column name for account IDs."
    )
    parser.add_argument(
        "--cusip_col", type=str, default="CUSIP",
        help="Column name for CUSIP / instrument ID."
    )
    parser.add_argument(
        "--datetime_col", type=str, default="TRADE_DATETIME",
        help="Column name for trade timestamp."
    )
    parser.add_argument(
        "--window_size", type=int, default=5,
        help="Window size (days) for RecentPop."
    )
    parser.add_argument(
        "--k", type=int, default=50,
        help="Top-K for Hit@K and Recall@K."
    )
    parser.add_argument(
        "--test_start", type=str, default=None,
        help="Test period start date (YYYY-MM-DD). "
             "If omitted, uses 80%% time split."
    )

    args = parser.parse_args()

    print(f"Loading data from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    results = backtest_popularity(
        df=df,
        account_col=args.account_col,
        cusip_col=args.cusip_col,
        datetime_col=args.datetime_col,
        window_size=args.window_size,
        k=args.k,
        test_start=args.test_start,
    )

    print("\n=== Backtest Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
