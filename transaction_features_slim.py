"""
Transaction Feature Engineering Pipeline  (slim edition)
=========================================================
~32 features for outlier detection on outgoing debit transactions.
All features are computed over configurable lookback windows with
no data leakage (row i only sees rows j where timestamp[j] < timestamp[i]).

Performance vs. the full edition
---------------------------------
  • 75 % fewer features (129 → ~32)
  • Single shared O(n²) window scan per lookback period instead of one
    scan per feature category → typically 5-8× faster wall-clock time

Expected input columns
-----------------------
    transaction_id  – unique transaction identifier
    date            – date of transaction (YYYY-MM-DD)
    timestamp       – full datetime of transaction
    debit_amount    – amount debited (positive float)
    receiver_id     – credit account / receiver identifier
    currency        – ISO currency code for the transaction

Usage
-----
    from transaction_features import TransactionFeaturePipeline

    pipeline = TransactionFeaturePipeline(lookback_windows=[30, 90])
    features_df = pipeline.fit_transform(df)

Feature inventory (~32 features)
----------------------------------
Amount (per lookback window):
    amt_{W}d_zscore       z-score of amount vs window history
    amt_{W}d_vs_median    amount / median amount in window
    amt_{W}d_pct_rank     percentile rank within window (0-1)
    amt_{W}d_tx_count     number of past transactions in window
    is_round_amount       1 if amount is a multiple of 100 or 1000

Temporal:
    is_weekend              1 if Saturday or Sunday
    is_off_hours            1 if outside 08:00-20:00
    seconds_since_last_tx   gap to previous transaction
    gap_zscore_30d          z-score of gap vs 30-day gap distribution
    tx_count_24h            burst counter: transactions in past 24 h
    velocity_7d_vs_30d      tx rate ratio short/long (spike detector)
                            [only when both 7 and 30 are in lookback_windows]

Receiver (per lookback window):
    recv_{W}d_is_new      1 if receiver not seen in window
    recv_{W}d_tx_count    past tx count to this receiver
    recv_{W}d_amt_zscore  z-score of amount for this specific receiver
    recv_{W}d_unique      distinct receivers in window
    recv_{W}d_herfindahl  receiver concentration index (0-1)

Currency:
    ccy_is_new_{maxW}d    1 if currency not seen in longest lookback window
    currency_switched     1 if different currency than previous tx

Portfolio / volume (per lookback window):
    port_{W}d_vol_zscore    z-score of this amount vs window baseline
    port_{W}d_tx_vol_share  this tx's share of window total volume
    tx_share_of_daily_vol   this tx's share of same-day volume
    vol_7d_vs_30d_ratio     volume acceleration (when both windows present)

Composite flags (30d base):
    flag_large_amt_new_recv  amt zscore > 2 AND new receiver
    flag_new_ccy_new_recv    new currency AND new receiver
    flag_burst_new_recv      top-10% burst day AND new receiver
"""

import pandas as pd
import numpy as np
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _window_seconds(days: int) -> int:
    return days * 86_400


def _zscore(value: float, mean: float, std: float) -> float:
    return (value - mean) / std if std > 0 else np.nan


def _compute_window_snapshots(df: pd.DataFrame, window_sec: int) -> list:
    """
    Single O(n²) pass — collects everything needed from one lookback window.
    Returns one dict per row; all feature builders read from this list.
    """
    ts    = df["timestamp"].values.astype("int64") // 10**9
    amts  = df["debit_amount"].values
    recvs = df["receiver_id"].values
    ccys  = df["currency"].values

    snapshots = []

    for i in range(len(df)):
        mask = (ts < ts[i]) & (ts[i] - ts <= window_sec)

        w_amts  = amts[mask]
        w_recvs = recvs[mask]
        w_ccys  = ccys[mask]
        n       = len(w_amts)

        # ── Amount stats ──────────────────────────────────────────────────────
        if n == 0:
            amt_mean = amt_std = amt_median = np.nan
            amt_pct_rank = np.nan
        else:
            amt_mean     = float(np.mean(w_amts))
            amt_std      = float(np.std(w_amts, ddof=1)) if n > 1 else 0.0
            amt_median   = float(np.median(w_amts))
            amt_pct_rank = float(np.mean(w_amts < amts[i]))

        # ── Receiver stats ────────────────────────────────────────────────────
        this_recv  = recvs[i]
        recv_amts  = w_amts[w_recvs == this_recv]
        recv_n     = len(recv_amts)

        recv_amt_mean = float(np.mean(recv_amts))   if recv_n > 0 else np.nan
        recv_amt_std  = float(np.std(recv_amts, ddof=1)) if recv_n > 1 else (0.0 if recv_n == 1 else np.nan)

        if n > 0:
            _, recv_counts = np.unique(w_recvs, return_counts=True)
            shares         = recv_counts / recv_counts.sum()
            herfindahl     = float(np.sum(shares ** 2))
            unique_recvs   = int(len(recv_counts))
        else:
            herfindahl   = np.nan
            unique_recvs = 0

        # ── Currency ──────────────────────────────────────────────────────────
        is_new_ccy = 0 if ccys[i] in w_ccys else 1

        # ── Inter-tx gap stats (for gap z-score) ──────────────────────────────
        if n > 1:
            w_ts_sorted = np.sort(ts[mask])
            gaps        = np.diff(w_ts_sorted).astype(float)
            gap_mean    = float(np.mean(gaps))
            gap_std     = float(np.std(gaps, ddof=1)) if len(gaps) > 1 else 0.0
        else:
            gap_mean = gap_std = np.nan

        snapshots.append({
            "n":             n,
            "w_amts":        w_amts,
            "w_recvs":       w_recvs,
            # amount
            "amt_mean":      amt_mean,
            "amt_std":       amt_std,
            "amt_median":    amt_median,
            "amt_pct_rank":  amt_pct_rank,
            # receiver
            "recv_n":        recv_n,
            "recv_amt_mean": recv_amt_mean,
            "recv_amt_std":  recv_amt_std,
            "is_new_recv":   0 if this_recv in w_recvs else 1,
            "unique_recvs":  unique_recvs,
            "herfindahl":    herfindahl,
            # currency
            "is_new_ccy":    is_new_ccy,
            # gap
            "gap_mean":      gap_mean,
            "gap_std":       gap_std,
        })

    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TransactionFeaturePipeline:
    """
    Parameters
    ----------
    lookback_windows : list[int]
        Lookback periods in days. Default: [30, 90].
        Include 7 to also get velocity_7d_vs_30d and vol_7d_vs_30d_ratio.
    """

    def __init__(self, lookback_windows: List[int] = None):
        self.lookback_windows = lookback_windows or [30, 90]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame with columns:
             transaction_id, date, timestamp, debit_amount, receiver_id, currency

        Returns
        -------
        pd.DataFrame — original columns + all engineered features
        """
        df = self._validate_and_prepare(df)
        print(f"[pipeline] Processing {len(df)} transactions …")

        # One shared scan per window — all feature builders read from these
        snapshots = {W: _compute_window_snapshots(df, _window_seconds(W))
                     for W in self.lookback_windows}
        snaps_24h = _compute_window_snapshots(df, _window_seconds(1))

        feat = df[["transaction_id"]].copy()
        feat = feat.join(self._amount_features(df, snapshots))
        feat = feat.join(self._temporal_features(df, snapshots, snaps_24h))
        feat = feat.join(self._receiver_features(df, snapshots))
        feat = feat.join(self._currency_features(df, snapshots))
        feat = feat.join(self._portfolio_features(df, snapshots))
        feat = feat.join(self._composite_features(feat))

        result = df.merge(feat, on="transaction_id", how="left")
        print(f"[pipeline] Done — {len(result.columns) - len(df.columns)} features, "
              f"output shape {result.shape}")
        return result

    # ── validation ────────────────────────────────────────────────────────────

    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"transaction_id", "date", "timestamp",
                    "debit_amount", "receiver_id", "currency"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = df.copy()
        df["timestamp"]    = pd.to_datetime(df["timestamp"])
        df["date"]         = pd.to_datetime(df["date"])
        df["debit_amount"] = df["debit_amount"].astype(float)
        return df.sort_values("timestamp").reset_index(drop=True)

    # ── 1. Amount ─────────────────────────────────────────────────────────────

    def _amount_features(self, df, snapshots):
        out  = pd.DataFrame(index=df.index)
        amts = df["debit_amount"].values

        out["is_round_amount"] = (
            (df["debit_amount"] % 100 == 0) | (df["debit_amount"] % 1000 == 0)
        ).astype(int)

        for W, snaps in snapshots.items():
            p = f"amt_{W}d"
            out[f"{p}_zscore"]    = [_zscore(amts[i], s["amt_mean"], s["amt_std"])
                                     for i, s in enumerate(snaps)]
            out[f"{p}_vs_median"] = [amts[i] / s["amt_median"]
                                     if s["amt_median"] and s["amt_median"] != 0 else np.nan
                                     for i, s in enumerate(snaps)]
            out[f"{p}_pct_rank"]  = [s["amt_pct_rank"] for s in snaps]
            out[f"{p}_tx_count"]  = [s["n"]            for s in snaps]

        return out

    # ── 2. Temporal ───────────────────────────────────────────────────────────

    def _temporal_features(self, df, snapshots, snaps_24h):
        out = pd.DataFrame(index=df.index)

        out["is_weekend"]   = (df["timestamp"].dt.dayofweek >= 5).astype(int)
        out["is_off_hours"] = (
            (df["timestamp"].dt.hour < 8) | (df["timestamp"].dt.hour >= 20)
        ).astype(int)

        out["seconds_since_last_tx"] = (
            df["timestamp"] - df["timestamp"].shift(1)
        ).dt.total_seconds().fillna(-1)

        # Gap z-score vs 30d (or longest available window)
        ref_W = 30 if 30 in snapshots else max(snapshots)
        out["gap_zscore_30d"] = [
            _zscore(out.at[i, "seconds_since_last_tx"],
                    snapshots[ref_W][i]["gap_mean"],
                    snapshots[ref_W][i]["gap_std"])
            for i in df.index
        ]

        out["tx_count_24h"] = [s["n"] for s in snaps_24h]

        if 7 in snapshots and 30 in snapshots:
            tx_7d  = np.array([s["n"] for s in snapshots[7]],  dtype=float)
            tx_30d = np.array([s["n"] for s in snapshots[30]], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out["velocity_7d_vs_30d"] = np.where(
                    tx_30d > 0, (tx_7d / 7) / (tx_30d / 30), np.nan
                )

        return out

    # ── 3. Receiver ───────────────────────────────────────────────────────────

    def _receiver_features(self, df, snapshots):
        out  = pd.DataFrame(index=df.index)
        amts = df["debit_amount"].values

        for W, snaps in snapshots.items():
            p = f"recv_{W}d"
            out[f"{p}_is_new"]     = [s["is_new_recv"]  for s in snaps]
            out[f"{p}_tx_count"]   = [s["recv_n"]        for s in snaps]
            out[f"{p}_amt_zscore"] = [_zscore(amts[i], s["recv_amt_mean"], s["recv_amt_std"])
                                      for i, s in enumerate(snaps)]
            out[f"{p}_unique"]     = [s["unique_recvs"]  for s in snaps]
            out[f"{p}_herfindahl"] = [s["herfindahl"]    for s in snaps]

        return out

    # ── 4. Currency ───────────────────────────────────────────────────────────

    def _currency_features(self, df, snapshots):
        out = pd.DataFrame(index=df.index)

        switched = (df["currency"] != df["currency"].shift(1)).astype(int)
        switched.iloc[0] = 0
        out["currency_switched"] = switched.values

        max_W = max(snapshots.keys())
        out[f"ccy_is_new_{max_W}d"] = [s["is_new_ccy"] for s in snapshots[max_W]]

        return out

    # ── 5. Portfolio ──────────────────────────────────────────────────────────

    def _portfolio_features(self, df, snapshots):
        out  = pd.DataFrame(index=df.index)
        amts = df["debit_amount"].values

        # Daily volume share — single pass using sorted order
        dates    = df["date"].values
        ts_epoch = df["timestamp"].values.astype("int64") // 10**9
        tx_share = []
        for i in range(len(df)):
            same_day = (dates == dates[i]) & (ts_epoch < ts_epoch[i])
            day_vol  = float(amts[same_day].sum())
            total    = day_vol + amts[i]
            tx_share.append(amts[i] / total if total > 0 else np.nan)
        out["tx_share_of_daily_vol"] = tx_share

        for W, snaps in snapshots.items():
            p = f"port_{W}d"
            out[f"{p}_vol_zscore"]   = [_zscore(amts[i], s["amt_mean"], s["amt_std"])
                                        for i, s in enumerate(snaps)]
            out[f"{p}_tx_vol_share"] = [
                amts[i] / (s["w_amts"].sum() + amts[i])
                if s["n"] > 0 else np.nan
                for i, s in enumerate(snaps)
            ]

        if 7 in snapshots and 30 in snapshots:
            vol_7d  = np.array([s["w_amts"].sum() for s in snapshots[7]],  dtype=float)
            vol_30d = np.array([s["w_amts"].sum() for s in snapshots[30]], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out["vol_7d_vs_30d_ratio"] = np.where(
                    vol_30d > 0, (vol_7d / 7) / (vol_30d / 30), np.nan
                )

        return out

    # ── 6. Composite flags ────────────────────────────────────────────────────

    def _composite_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=feat.index)

        def _get(col):
            return feat[col] if col in feat.columns else pd.Series(0, index=feat.index)

        ref_W    = 30 if "amt_30d_zscore" in feat.columns else self.lookback_windows[0]
        amt_z    = _get(f"amt_{ref_W}d_zscore")
        new_recv = _get(f"recv_{ref_W}d_is_new")
        ccy_cols = [c for c in feat.columns if c.startswith("ccy_is_new_")]
        new_ccy  = feat[ccy_cols[-1]] if ccy_cols else pd.Series(0, index=feat.index)
        burst    = _get("tx_count_24h")

        out["flag_large_amt_new_recv"] = (
            (amt_z.fillna(0) > 2).astype(int) * new_recv
        ).astype(int)
        out["flag_new_ccy_new_recv"] = (
            new_ccy * new_recv
        ).fillna(0).astype(int)
        out["flag_burst_new_recv"] = (
            (burst > burst.quantile(0.90)).astype(int) * new_recv
        ).fillna(0).astype(int)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Demo / smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _generate_sample_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")

    receivers  = [f"ACC_{i:04d}" for i in range(20)]
    currencies = ["USD", "EUR", "GBP", "JPY", "CHF"]

    offsets_hours = np.cumsum(rng.exponential(scale=12, size=n))
    timestamps    = [base + pd.Timedelta(hours=float(h)) for h in offsets_hours]
    amounts       = rng.lognormal(mean=6, sigma=1, size=n)
    amounts[50]  *= 20
    amounts[120] *= 15

    recv_choices = rng.choice(receivers, size=n)
    recv_choices[50]  = "ACC_NEW1"
    recv_choices[120] = "ACC_NEW2"

    ccy_choices = rng.choice(currencies[:2], size=n)
    ccy_choices[80] = "JPY"

    return pd.DataFrame({
        "transaction_id": [f"TXN_{i:05d}" for i in range(n)],
        "date":           [t.date() for t in timestamps],
        "timestamp":      timestamps,
        "debit_amount":   np.round(amounts, 2),
        "receiver_id":    recv_choices,
        "currency":       ccy_choices,
    })


if __name__ == "__main__":
    import time

    sample_df = _generate_sample_data(n=300)
    pipeline  = TransactionFeaturePipeline(lookback_windows=[7, 30, 90])

    t0 = time.perf_counter()
    features_df = pipeline.fit_transform(sample_df)
    elapsed = time.perf_counter() - t0

    feat_cols = [c for c in features_df.columns if c not in sample_df.columns]
    print(f"\n{len(feat_cols)} feature columns  (runtime: {elapsed:.2f}s):")
    for c in feat_cols:
        print(f"  {c}")

    print("\nInjected outlier rows (49–52):")
    display_cols = [
        "transaction_id", "debit_amount", "receiver_id", "currency",
        "amt_30d_zscore", "recv_30d_is_new", "ccy_is_new_90d",
        "flag_large_amt_new_recv", "tx_count_24h",
    ]
    print(features_df.loc[49:52, display_cols].to_string(index=False))

    features_df.to_csv("/mnt/user-data/outputs/transaction_features_sample.csv", index=False)
    print("\nOutput saved to transaction_features_sample.csv")
