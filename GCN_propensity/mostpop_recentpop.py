#!/usr/bin/env python
"""
Let $\mathcal{U}$ denote the set of institutional accounts (clients) and 
$\mathcal{I}$ the set of credit instruments (CUSIPs). 
Each trade event is represented by a triplet $(u, i, t)$ where 
$u \in \mathcal{U}$, $i \in \mathcal{I}$, and $t$ denotes the trade date. 
The complete dataset is therefore 
$\mathcal{E} = \{(u, i, t)\}$, chronologically ordered by $t$.

For each day $t$, we define:
\[
\mathcal{E}_{<t} = \{(u,i,t') \mid t' < t\}, \quad
\mathcal{E}_{t,w} = \{(u,i,t') \mid t-w < t' < t\},
\]
where $w$ is the lookback window (in days).

The goal is to produce, for each active account $u$ and each day $t$,
a ranked list of candidate items $i \in \mathcal{I}$ that the account is most likely to trade at $t$.

\subsection{Baselines}

\paragraph{MostPop (Global Popularity).}
This non-personalized baseline ranks items according to their total
historical interaction counts over the entire training set:
\[
\text{score}_{\text{MostPop}}(i, t)
= |\{(u, i, t') \mid t' < t\}|.
\]
Items with higher cumulative counts receive higher ranks.
This model is static and does not adapt to temporal shifts.

\paragraph{RecentPop (Windowed Popularity).}
To introduce temporal awareness, the RecentPop baseline restricts
the popularity computation to a rolling window of $w$ days preceding $t$:
\[
\text{score}_{\text{RecentPop}}(i, t)
= |\{(u, i, t') \mid t-w < t' < t\}|.
\]
This allows the model to emphasize recently active CUSIPs, 
adapting to short-lived market attention and bond issuance/maturity cycles.

Both baselines generate a global ranking of all items $\mathcal{I}$,
which is then applied identically to every user $u$.

\subsection{Rolling-Window Evaluation}

The backtesting framework simulates daily recommendation rounds.  
For each trading day $t$ in the test period:

\begin{enumerate}
    \item Construct $\mathcal{E}_{<t}$ and $\mathcal{E}_{t,w}$.
    \item Compute item popularity under each baseline.
    \item For each active user $u$ on day $t$, form the recommendation list 
    $\mathcal{R}^u_t = [i_1, i_2, \dots, i_K]$ by ranking items according to the
    chosen baseline’s scores.
    \item Compare $\mathcal{R}^u_t$ against the user’s actual traded items 
    $\mathcal{T}^u_t = \{i \mid (u,i,t) \in \mathcal{E}\}$.
\end{enumerate}

\subsection{Evaluation Metrics}

To assess the quality of the daily rankings, we adopt standard
Top-$K$ recommendation metrics:

\paragraph{Hit@K.}
A binary indicator that equals 1 if at least one of the items traded by user $u$ on day $t$
appears within the top-$K$ recommendations:
\[
\text{Hit@K}_{u,t} = 
\begin{cases}
1, & \text{if } \mathcal{T}^u_t \cap \mathcal{R}^u_t[:K] \neq \emptyset, \\
0, & \text{otherwise.}
\end{cases}
\]

\paragraph{Recall@K.}
Measures the fraction of traded items recovered within the top-$K$ recommendations:
\[
\text{Recall@K}_{u,t} =
\frac{|\mathcal{T}^u_t \cap \mathcal{R}^u_t[:K]|}{|\mathcal{T}^u_t|}.
\]

These per-user daily metrics are averaged across all active $(u,t)$ pairs
in the test period:
\[
\text{Metric} = \frac{1}{|\mathcal{Q}|} 
\sum_{(u,t)\in\mathcal{Q}} \text{Metric}_{u,t},
\]
where $\mathcal{Q}$ is the set of all evaluation user-days.

\subsection{Remarks}

\begin{itemize}
    \item The rolling-window procedure avoids temporal leakage by 
    restricting each prediction at time $t$ to interactions strictly before $t$.
    \item The window size $w$ controls temporal responsiveness:
    smaller $w$ captures short-term market shifts, while larger $w$
    smooths popularity trends.
    \item Although both baselines ignore personalization, they serve as 
    strong reference points for temporal models such as 
    LightGCN-W or LightGCN-FW.
\end{itemize}

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
    df = prepare_data(df, account_col=account_col,
                      cusip_col=cusip_col, datetime_col=datetime_col)

    all_dates = df["TRADE_DATE"].drop_duplicates().sort_values().tolist()
    if len(all_dates) < 2:
        raise ValueError("Not enough distinct dates to run a backtest.")

    if test_start is not None:
        test_start = pd.to_datetime(test_start).floor("D")
    else:
        split_idx = int(0.8 * len(all_dates))
        test_start = all_dates[split_idx]

    print(f"Number of distinct trading days: {len(all_dates)}")
    print(f"Test period starts at: {test_start.date()}")

    stats_global = {
        "mostpop_hit_sum": 0.0,
        "mostpop_recall_sum": 0.0,
        "mostpop_count": 0,
        "recentpop_hit_sum": 0.0,
        "recentpop_recall_sum": 0.0,
        "recentpop_count": 0,
    }

    # new: store per-day metrics
    daily_records = []

    for current_date in all_dates:
        if current_date < test_start:
            continue

        df_day = df[df["TRADE_DATE"] == current_date]
        if df_day.empty:
            continue

        history_mask = df["TRADE_DATE"] < current_date
        history_df = df[history_mask]
        if history_df.empty:
            continue

        from datetime import timedelta
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

        # aggregate into globals
        for key in stats_global:
            stats_global[key] += day_stats[key]

        # ---- compute per-day averages over accounts ----
        day_record = {"date": current_date}

        # MostPop
        if day_stats["mostpop_count"] > 0:
            day_record[f"MostPop_Hit@{k}"] = (
                day_stats["mostpop_hit_sum"] / day_stats["mostpop_count"]
            )
            day_record[f"MostPop_Recall@{k}"] = (
                day_stats["mostpop_recall_sum"] / day_stats["mostpop_count"]
            )
        else:
            day_record[f"MostPop_Hit@{k}"] = np.nan
            day_record[f"MostPop_Recall@{k}"] = np.nan

        # RecentPop
        if day_stats["recentpop_count"] > 0:
            day_record[f"RecentPop_Hit@{k}"] = (
                day_stats["recentpop_hit_sum"] / day_stats["recentpop_count"]
            )
            day_record[f"RecentPop_Recall@{k}"] = (
                day_stats["recentpop_recall_sum"] / day_stats["recentpop_count"]
            )
        else:
            day_record[f"RecentPop_Hit@{k}"] = np.nan
            day_record[f"RecentPop_Recall@{k}"] = np.nan

        day_record["num_accounts_mostpop"] = day_stats["mostpop_count"]
        day_record["num_accounts_recentpop"] = day_stats["recentpop_count"]

        daily_records.append(day_record)

    # ---- global averages (unchanged) ----
    results = {}
    if stats_global["mostpop_count"] > 0:
        results[f"MostPop_Hit@{k}"] = (
            stats_global["mostpop_hit_sum"] / stats_global["mostpop_count"]
        )
        results[f"MostPop_Recall@{k}"] = (
            stats_global["mostpop_recall_sum"] / stats_global["mostpop_count"]
        )
    else:
        results[f"MostPop_Hit@{k}"] = np.nan
        results[f"MostPop_Recall@{k}"] = np.nan

    if stats_global["recentpop_count"] > 0:
        results[f"RecentPop_Hit@{k}"] = (
            stats_global["recentpop_hit_sum"] / stats_global["recentpop_count"]
        )
        results[f"RecentPop_Recall@{k}"] = (
            stats_global["recentpop_recall_sum"] / stats_global["recentpop_count"]
        )
    else:
        results[f"RecentPop_Hit@{k}"] = np.nan
        results[f"RecentPop_Recall@{k}"] = np.nan

    results["Num_eval_user_days_MostPop"] = stats_global["mostpop_count"]
    results["Num_eval_user_days_RecentPop"] = stats_global["recentpop_count"]

    # new: return both global summary and per-day DataFrame
    daily_df = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)
    return results, daily_df



results, daily_df = backtest_popularity(
    df,
    window_size=5,
    k=50,
    test_start="2023-01-01",
)

print(results)
print(daily_df.head())
