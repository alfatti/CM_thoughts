import pandas as pd
import numpy as np
from scipy import sparse

def build_account_panels(
    df,
    cusip_subset=None,                 # iterable of CUSIPs to keep; if None, use all
    date_col="TRADE_DATETIME",
    acct_col="ACCOUNTID",
    cusip_col="CUSIP",
    amt_col="TRADE_AMT",              # adjust to your amount column name
    freq="D",
):
    # 1) Normalize and aggregate to daily long format
    gdf = (
        df[[date_col, acct_col, cusip_col, amt_col]]
        .assign(date=pd.to_datetime(df[date_col]).dt.normalize())
        .groupby(["date", acct_col, cusip_col], observed=True, as_index=False)[amt_col]
        .sum()
    )

    # 2) Filter to CUSIP subset (if provided)
    if cusip_subset is not None:
        gdf = gdf[gdf[cusip_col].isin(set(cusip_subset))]

    # 3) Establish global date range (aligned for all accounts)
    if gdf.empty:
        return {}
    full_dates = pd.date_range(gdf["date"].min(), gdf["date"].max(), freq=freq)

    # 4) Category-encode keys for stable indexing
    #    Keep CUSIP categories fixed (subset order) for consistent column layout
    if cusip_subset is None:
        cusips = gdf[cusip_col].unique().tolist()
    else:
        cusips = list(cusip_subset)

    gdf[cusip_col] = pd.Categorical(gdf[cusip_col], categories=cusips, ordered=False)
    accounts = gdf[acct_col].unique().tolist()

    # 5) Build per-account CSR panels
    panels = {}
    col_indexer = {c:i for i,c in enumerate(cusips)}

    for a in accounts:
        sub = gdf[gdf[acct_col] == a][["date", cusip_col, amt_col]].copy()
        if sub.empty:
            # still return an all-zero matrix for full_dates × cusips
            X = sparse.csr_matrix((len(full_dates), len(cusips)), dtype=np.float64)
            panels[a] = {"dates": full_dates, "cusips": cusips, "X": X}
            continue

        # Map dates to row indices
        # (Use a fast align via reindex on a pivoted Series)
        sub["row"] = pd.Index(full_dates).get_indexer(sub["date"])
        sub["col"] = sub[cusip_col].map(col_indexer)

        # Drop any rows that fell outside the date_range (shouldn’t happen, but safe)
        sub = sub[(sub["row"] >= 0) & (sub["col"].notna())]
        if sub.empty:
            X = sparse.csr_matrix((len(full_dates), len(cusips)), dtype=np.float64)
        else:
            X = sparse.coo_matrix(
                (sub[amt_col].to_numpy(dtype=np.float64),
                 (sub["row"].to_numpy(), sub["col"].to_numpy(dtype=int))),
                shape=(len(full_dates), len(cusips)),
            ).tocsr()

        panels[a] = {"dates": full_dates, "cusips": cusips, "X": X}

    return panels
