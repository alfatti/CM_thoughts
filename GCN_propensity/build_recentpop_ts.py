#!/usr/bin/env python
"""
Build a panel of rolling RecentPop scores for each CUSIP.

Given raw trades with at least:
    - CUSIP
    - TRADE_DATETIME   (anything pandas.to_datetime can parse)

We produce a daily panel with:
    - TRADE_DATE
    - CUSIP
    - daily_trades   = # trades of that CUSIP on that day
    - recentpop_w    = rolling sum of daily_trades over last w days (including today)

RecentPop_w(i, t) = sum_{d=t-w+1}^t daily_trades(i, d)
"""

import argparse
from datetime import timedelta

import pandas as pd


def build_recentpop_panel(
    df,
    cusip_col="CUSIP",
    datetime_col="TRADE_DATETIME",
    window_size=5,
    full_calendar=True,
):
    """
    Build a panel of rolling RecentPop scores for each CUSIP.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trades data.
    cusip_col : str
        Column name for CUSIP / instrument identifier.
    datetime_col : str
        Column name for trade timestamp.
    window_size : int
        Rolling window length in days.
    full_calendar : bool
        If True: build a dense panel over every calendar day between
        min and max TRADE_DATE (even if no trades happened that day).
        If False: only keep days where at least one trade occurred.

    Returns
    -------
    panel : pd.DataFrame
        Columns:
            TRADE_DATE  (datetime64[ns])
            CUSIP       (as in input)
            daily_trades
            recentpop_w
    """

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # 1. Normalize dates
    df = df.copy()
    df = df.dropna(subset=[cusip_col, datetime_col])
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["TRADE_DATE"] = df[datetime_col].dt.floor("D")

    # 2. Aggregate to daily counts per CUSIP
    daily_counts = (
        df.groupby(["TRADE_DATE", cusip_col])
        .size()
        .rename("daily_trades")
        .reset_index()
    )

    # 3. Build panel index (date × CUSIP)
    cusips = daily_counts[cusip_col].unique()

    if full_calendar:
        min_date = daily_counts["TRADE_DATE"].min()
        max_date = daily_counts["TRADE_DATE"].max()
        all_dates = pd.date_range(min_date, max_date, freq="D")
    else:
        all_dates = (
            daily_counts["TRADE_DATE"].drop_duplicates().sort_values().to_list()
        )

    idx = pd.MultiIndex.from_product(
        [all_dates, cusips],
        names=["TRADE_DATE", cusip_col],
    )

    # 4. Reindex to full panel, fill missing daily_trades with 0
    panel = (
        daily_counts
        .set_index(["TRADE_DATE", cusip_col])
        .reindex(idx)
        .sort_index()
    )
    panel["daily_trades"] = panel["daily_trades"].fillna(0).astype(int)

    # 5. Rolling RecentPop per CUSIP
    # For each CUSIP separately, do a rolling sum of daily_trades.
    panel["recentpop_w"] = (
        panel.groupby(level=cusip_col)["daily_trades"]
        .rolling(window=window_size, min_periods=1)
        .sum()
        .reset_index(level=cusip_col, drop=True)
    )

    # 6. Tidy up
    panel = panel.reset_index()  # back to columns

    # Optional: sort
    panel = panel.sort_values(["TRADE_DATE", cusip_col]).reset_index(drop=True)

    return panel


def main():
    parser = argparse.ArgumentParser(
        description="Create rolling RecentPop panel per CUSIP."
    )
    parser.add_argument("csv_path", type=str, help="Path to raw trades CSV.")
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
        help="Rolling window size in days for RecentPop."
    )
    parser.add_argument(
        "--full_calendar", action="store_true",
        help="If set, include every calendar day between min and max TRADE_DATE."
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Optional path to save the panel as CSV."
    )

    args = parser.parse_args()

    print(f"Loading trades from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)

    panel = build_recentpop_panel(
        df,
        cusip_col=args.cusip_col,
        datetime_col=args.datetime_col,
        window_size=args.window_size,
        full_calendar=args.full_calendar,
    )

    print("Built RecentPop panel with shape:", panel.shape)
    print(panel.head())

    if args.output_path:
        print(f"Saving panel to: {args.output_path}")
        panel.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
