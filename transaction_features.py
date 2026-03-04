"""
Transaction Feature Engineering Pipeline
=========================================
Computes outlier-detection features for outgoing debit transactions
from a single origin account, across configurable lookback windows.

Expected input columns:
    - transaction_id     : unique transaction identifier
    - date               : date of transaction (YYYY-MM-DD)
    - timestamp          : full datetime of transaction
    - debit_amount       : amount debited (positive float)
    - receiver_id        : credit account / receiver identifier
    - currency           : currency of the transaction

Usage:
    from transaction_features import TransactionFeaturePipeline

    pipeline = TransactionFeaturePipeline(lookback_windows=[7, 30, 90])
    features = pipeline.fit_transform(df)
"""

import pandas as pd
import numpy as np
from typing import List


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _zscore(value: float, mean: float, std: float) -> float:
    """Safe z-score; returns NaN when std == 0."""
    return (value - mean) / std if std > 0 else np.nan


def _rolling_stats(df: pd.DataFrame, col: str, window_seconds: int) -> pd.DataFrame:
    """
    For each row i, compute stats over all rows j where:
        timestamp[j] < timestamp[i]   (strictly past)
        timestamp[i] - timestamp[j] <= window_seconds
    Returns a DataFrame of per-row {mean, std, median, max, count, sum, iqr, skew}.
    """
    results = []
    ts = df["timestamp"].values.astype("int64") // 10**9   # epoch seconds
    vals = df[col].values

    for i in range(len(df)):
        mask = (ts < ts[i]) & (ts[i] - ts < window_seconds)
        window_vals = vals[mask]
        if len(window_vals) == 0:
            results.append({
                "mean": np.nan, "std": np.nan, "median": np.nan,
                "max": np.nan,  "count": 0,    "sum": 0.0,
                "iqr": np.nan,  "skew": np.nan,
            })
        else:
            s = pd.Series(window_vals, dtype=float)
            q75, q25 = np.percentile(s, [75, 25])
            results.append({
                "mean":   s.mean(),
                "std":    s.std(ddof=1) if len(s) > 1 else 0.0,
                "median": s.median(),
                "max":    s.max(),
                "count":  len(s),
                "sum":    s.sum(),
                "iqr":    q75 - q25,
                "skew":   float(s.skew()) if len(s) > 2 else np.nan,
            })
    return pd.DataFrame(results, index=df.index)


def _window_seconds(days: int) -> int:
    return days * 86_400


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

class TransactionFeaturePipeline:
    """
    Parameters
    ----------
    lookback_windows : list of ints
        Lookback periods in days. Default: [7, 30, 90].
    burst_windows_hours : list of ints
        Short windows (in hours) for burst/velocity features. Default: [1, 6, 24].
    """

    def __init__(
        self,
        lookback_windows: List[int] = None,
        burst_windows_hours: List[int] = None,
    ):
        self.lookback_windows = lookback_windows or [7, 30, 90]
        self.burst_windows_hours = burst_windows_hours or [1, 6, 24]

    # ── public entry point ──────────────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Raw transactions for ONE origin account with columns:
            transaction_id, date, timestamp, debit_amount, receiver_id, currency.

        Returns
        -------
        pd.DataFrame
            One row per transaction; original columns + all engineered features.
        """
        df = self._validate_and_prepare(df)
        feat = df[["transaction_id"]].copy()

        print(f"[pipeline] Processing {len(df)} transactions …")

        feat = feat.join(self._amount_features(df))
        feat = feat.join(self._temporal_features(df))
        feat = feat.join(self._receiver_features(df))
        feat = feat.join(self._currency_features(df))
        feat = feat.join(self._portfolio_features(df))
        feat = feat.join(self._composite_features(feat))

        # Restore original columns at the front
        result = df.merge(feat, on="transaction_id", how="left")
        print(f"[pipeline] Done. Output shape: {result.shape}")
        return result

    # ── validation ──────────────────────────────────────────────────────────

    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"transaction_id", "date", "timestamp", "debit_amount",
                    "receiver_id", "currency"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = pd.to_datetime(df["date"])
        df["debit_amount"] = df["debit_amount"].astype(float)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ── 1. Amount features ───────────────────────────────────────────────────

    def _amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # Is the amount suspiciously round?
        out["is_round_amount"] = (
            (df["debit_amount"] % 100 == 0) | (df["debit_amount"] % 1000 == 0)
        ).astype(int)

        for W in self.lookback_windows:
            ws = _window_seconds(W)
            stats = _rolling_stats(df, "debit_amount", ws)
            prefix = f"amt_{W}d"

            out[f"{prefix}_mean"]       = stats["mean"]
            out[f"{prefix}_std"]        = stats["std"]
            out[f"{prefix}_median"]     = stats["median"]
            out[f"{prefix}_max"]        = stats["max"]
            out[f"{prefix}_iqr"]        = stats["iqr"]
            out[f"{prefix}_skew"]       = stats["skew"]
            out[f"{prefix}_tx_count"]   = stats["count"]

            # Z-score of current amount vs window
            out[f"{prefix}_zscore"] = [
                _zscore(df.at[i, "debit_amount"], stats.at[i, "mean"], stats.at[i, "std"])
                for i in df.index
            ]

            # Ratio vs mean / median / max
            out[f"{prefix}_vs_mean"]   = df["debit_amount"] / stats["mean"].replace(0, np.nan)
            out[f"{prefix}_vs_median"] = df["debit_amount"] / stats["median"].replace(0, np.nan)
            out[f"{prefix}_vs_max"]    = df["debit_amount"] / stats["max"].replace(0, np.nan)

            # Percentile rank within window (expensive but exact)
            percentiles = []
            ts = df["timestamp"].values.astype("int64") // 10**9
            vals = df["debit_amount"].values
            for i in range(len(df)):
                mask = (ts < ts[i]) & (ts[i] - ts < ws)
                w = vals[mask]
                if len(w) == 0:
                    percentiles.append(np.nan)
                else:
                    percentiles.append(float(np.mean(w < vals[i])))
            out[f"{prefix}_pct_rank"] = percentiles

        return out

    # ── 2. Temporal / frequency features ────────────────────────────────────

    def _temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        ts_epoch = df["timestamp"].values.astype("int64") // 10**9

        # Time-of-day / calendar
        out["hour_of_day"]  = df["timestamp"].dt.hour
        out["day_of_week"]  = df["timestamp"].dt.dayofweek   # 0=Mon
        out["is_weekend"]   = (out["day_of_week"] >= 5).astype(int)
        out["is_off_hours"] = ((out["hour_of_day"] < 8) | (out["hour_of_day"] >= 20)).astype(int)

        # Gap since previous transaction (seconds)
        prev_ts = df["timestamp"].shift(1)
        out["seconds_since_last_tx"] = (
            df["timestamp"] - prev_ts
        ).dt.total_seconds().fillna(-1)

        # ── Burst windows (short, in hours) ──
        for H in self.burst_windows_hours:
            ws = H * 3600
            counts = []
            for i in range(len(df)):
                mask = (ts_epoch < ts_epoch[i]) & (ts_epoch[i] - ts_epoch < ws)
                counts.append(int(mask.sum()))
            out[f"tx_count_{H}h"] = counts

        # ── Lookback windows ──
        for W in self.lookback_windows:
            ws = _window_seconds(W)
            prefix = f"freq_{W}d"

            # Transaction count in window
            counts, gaps_mean, gaps_std = [], [], []
            hour_counts, dow_counts = [], []

            for i in range(len(df)):
                mask = (ts_epoch < ts_epoch[i]) & (ts_epoch[i] - ts_epoch < ws)
                window_df = df[mask]
                n = len(window_df)
                counts.append(n)

                # Inter-transaction gap stats within window
                if n > 1:
                    gaps = window_df["timestamp"].diff().dt.total_seconds().dropna()
                    gaps_mean.append(gaps.mean())
                    gaps_std.append(gaps.std(ddof=1) if len(gaps) > 1 else 0.0)
                else:
                    gaps_mean.append(np.nan)
                    gaps_std.append(np.nan)

                # How often have we transacted at this hour / DOW historically?
                this_hour = df.at[i, "timestamp"].hour
                this_dow  = df.at[i, "timestamp"].dayofweek
                hour_counts.append(int((window_df["timestamp"].dt.hour == this_hour).sum()))
                dow_counts.append(int((window_df["timestamp"].dt.dayofweek == this_dow).sum()))

            out[f"{prefix}_tx_count"]       = counts
            out[f"{prefix}_gap_mean_sec"]   = gaps_mean
            out[f"{prefix}_gap_std_sec"]    = gaps_std
            out[f"{prefix}_same_hour_count"] = hour_counts
            out[f"{prefix}_same_dow_count"]  = dow_counts

        # Velocity ratio: tx_count last 7d / tx_count last 30d
        if 7 in self.lookback_windows and 30 in self.lookback_windows:
            out["velocity_7d_vs_30d"] = (
                out["freq_7d_tx_count"] / out["freq_30d_tx_count"].replace(0, np.nan)
            )

        # Gap z-score vs 30d baseline
        if 30 in self.lookback_windows:
            out["gap_zscore_30d"] = [
                _zscore(out.at[i, "seconds_since_last_tx"],
                        out.at[i, "freq_30d_gap_mean_sec"],
                        out.at[i, "freq_30d_gap_std_sec"])
                for i in df.index
            ]

        return out

    # ── 3. Receiver interaction features ────────────────────────────────────

    def _receiver_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ts_epoch = df["timestamp"].values.astype("int64") // 10**9
        receivers = df["receiver_id"].values
        amounts   = df["debit_amount"].values

        for W in self.lookback_windows:
            ws = _window_seconds(W)
            prefix = f"recv_{W}d"

            is_new, first_seen, recv_tx_count = [], [], []
            amt_vs_recv_mean, amt_vs_recv_std, amt_recv_zscore = [], [], []
            amt_vs_recv_max, unique_recv, new_recv_ratio = [], [], []
            herfindahl = []

            for i in range(len(df)):
                mask = (ts_epoch < ts_epoch[i]) & (ts_epoch[i] - ts_epoch < ws)
                window_df  = df[mask]
                this_recv  = receivers[i]
                this_amt   = amounts[i]

                # ── Novelty ──
                past_recv = window_df["receiver_id"].values
                seen       = this_recv in past_recv
                is_new.append(0 if seen else 1)

                # Days since first transaction to this receiver (all history)
                all_past_mask = (ts_epoch < ts_epoch[i])
                all_past = df[all_past_mask]
                recv_history = all_past[all_past["receiver_id"] == this_recv]
                if len(recv_history) > 0:
                    first_ts = recv_history["timestamp"].min()
                    first_seen.append(
                        (df.at[i, "timestamp"] - first_ts).total_seconds() / 86_400
                    )
                else:
                    first_seen.append(np.nan)

                # ── Receiver-specific amount stats ──
                recv_window = window_df[window_df["receiver_id"] == this_recv]["debit_amount"]
                recv_tx_count.append(len(recv_window))

                if len(recv_window) > 0:
                    rm, rs = recv_window.mean(), recv_window.std(ddof=1) if len(recv_window) > 1 else 0.0
                    amt_vs_recv_mean.append(this_amt / rm if rm != 0 else np.nan)
                    amt_vs_recv_std.append(this_amt - rm)
                    amt_recv_zscore.append(_zscore(this_amt, rm, rs))
                    amt_vs_recv_max.append(this_amt / recv_window.max())
                else:
                    amt_vs_recv_mean.append(np.nan)
                    amt_vs_recv_std.append(np.nan)
                    amt_recv_zscore.append(np.nan)
                    amt_vs_recv_max.append(np.nan)

                # ── Concentration (Herfindahl index) ──
                n_window = len(window_df)
                if n_window > 0:
                    recv_counts = window_df["receiver_id"].value_counts(normalize=True)
                    herfindahl.append(float((recv_counts ** 2).sum()))
                    unique_recv.append(len(recv_counts))
                    # Proportion of past-window txs whose receiver had never appeared before that tx
                    all_recv_before_window = df[ts_epoch < (ts_epoch[i] - ws)]["receiver_id"].values
                    new_flags = [1 if r not in all_recv_before_window else 0 for r in past_recv]
                    new_recv_ratio.append(float(np.mean(new_flags)) if new_flags else np.nan)
                else:
                    herfindahl.append(np.nan)
                    unique_recv.append(0)
                    new_recv_ratio.append(np.nan)

            out[f"{prefix}_is_new_receiver"]       = is_new
            out[f"{prefix}_first_seen_days_ago"]   = first_seen
            out[f"{prefix}_tx_count"]              = recv_tx_count
            out[f"{prefix}_amt_vs_mean"]           = amt_vs_recv_mean
            out[f"{prefix}_amt_deviation"]         = amt_vs_recv_std
            out[f"{prefix}_amt_zscore"]            = amt_recv_zscore
            out[f"{prefix}_amt_vs_max"]            = amt_vs_recv_max
            out[f"{prefix}_unique_receivers"]      = unique_recv
            out[f"{prefix}_new_recv_ratio"]        = new_recv_ratio
            out[f"{prefix}_herfindahl"]            = herfindahl

        # Receiver share of total volume (30d default)
        if 30 in self.lookback_windows:
            ws = _window_seconds(30)
            recv_vol_share = []
            for i in range(len(df)):
                mask = (ts_epoch < ts_epoch[i]) & (ts_epoch[i] - ts_epoch < ws)
                window_df = df[mask]
                total_vol = window_df["debit_amount"].sum()
                recv_vol  = window_df[window_df["receiver_id"] == receivers[i]]["debit_amount"].sum()
                recv_vol_share.append(recv_vol / total_vol if total_vol > 0 else np.nan)
            out["recv_30d_vol_share"] = recv_vol_share

        return out

    # ── 4. Currency features ─────────────────────────────────────────────────

    def _currency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ts_epoch  = df["timestamp"].values.astype("int64") // 10**9
        currencies = df["currency"].values

        # Currency switch vs previous transaction
        switched = (df["currency"] != df["currency"].shift(1)).astype(int)
        switched.iloc[0] = 0
        out["currency_switched"] = switched.values

        for W in self.lookback_windows:
            ws = _window_seconds(W)
            prefix = f"ccy_{W}d"

            is_new_ccy, ccy_tx_count, unique_ccy = [], [], []
            dominant_ccy_share = []

            for i in range(len(df)):
                mask = (ts_epoch < ts_epoch[i]) & (ts_epoch[i] - ts_epoch < ws)
                window_df   = df[mask]
                this_ccy    = currencies[i]
                past_ccys   = window_df["currency"].values

                is_new_ccy.append(0 if this_ccy in past_ccys else 1)
                ccy_tx_count.append(int((past_ccys == this_ccy).sum()))

                if len(past_ccys) > 0:
                    counts = pd.Series(past_ccys).value_counts(normalize=True)
                    unique_ccy.append(len(counts))
                    dominant_ccy_share.append(float(counts.iloc[0]))
                else:
                    unique_ccy.append(0)
                    dominant_ccy_share.append(np.nan)

            out[f"{prefix}_is_new_currency"]      = is_new_ccy
            out[f"{prefix}_tx_count"]             = ccy_tx_count
            out[f"{prefix}_unique_currencies"]    = unique_ccy
            out[f"{prefix}_dominant_ccy_share"]   = dominant_ccy_share

        return out

    # ── 5. Portfolio / volume features ──────────────────────────────────────

    def _portfolio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        ts_epoch = df["timestamp"].values.astype("int64") // 10**9
        amounts  = df["debit_amount"].values

        # Daily cumulative volume (same calendar day, before current tx)
        daily_vol_before = []
        for i in range(len(df)):
            same_day_mask = (
                (df["date"] == df.at[i, "date"]) &
                (ts_epoch < ts_epoch[i])
            )
            daily_vol_before.append(df[same_day_mask]["debit_amount"].sum())
        out["daily_vol_before_tx"] = daily_vol_before
        out["tx_share_of_daily_vol"] = [
            amounts[i] / (daily_vol_before[i] + amounts[i])
            if (daily_vol_before[i] + amounts[i]) > 0 else np.nan
            for i in range(len(df))
        ]

        for W in self.lookback_windows:
            ws = _window_seconds(W)
            prefix = f"port_{W}d"

            stats = _rolling_stats(df, "debit_amount", ws)
            out[f"{prefix}_total_volume"]    = stats["sum"]
            out[f"{prefix}_daily_avg_vol"]   = stats["sum"] / W
            out[f"{prefix}_vol_zscore"]      = [
                _zscore(amounts[i], stats.at[i, "mean"], stats.at[i, "std"])
                for i in df.index
            ]

        # Volume acceleration: 7d vs 30d
        if 7 in self.lookback_windows and 30 in self.lookback_windows:
            out["vol_7d_vs_30d_ratio"] = (
                out["port_7d_total_volume"] /
                (out["port_30d_total_volume"] / (30 / 7)).replace(0, np.nan)
            )

        return out

    # ── 6. Composite / interaction features ─────────────────────────────────

    def _composite_features(self, feat: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=feat.index)

        # Use 30d window as the default for composites
        def _safe_get(col):
            return feat[col] if col in feat.columns else pd.Series(np.nan, index=feat.index)

        new_recv    = _safe_get("recv_30d_is_new_receiver")
        amt_zscore  = _safe_get("amt_30d_zscore")
        off_hours   = _safe_get("is_off_hours")
        weekend     = _safe_get("is_weekend")
        new_ccy     = _safe_get("ccy_30d_is_new_currency")
        burst_24h   = _safe_get("tx_count_24h")

        # High amount + new receiver
        out["flag_large_amt_new_recv"]     = (
            (amt_zscore > 2).astype(int) * new_recv
        )
        # Off-hours + large amount
        out["flag_off_hours_large_amt"]    = (
            off_hours * (amt_zscore > 2).astype(int)
        )
        # New currency + new receiver (double novelty)
        out["flag_new_ccy_new_recv"]       = new_ccy * new_recv

        # Burst + new receivers
        out["flag_burst_new_recv"]         = (
            (burst_24h > burst_24h.quantile(0.90)).astype(int) * new_recv
        )

        # Weekend + off-hours
        out["flag_weekend_off_hours"]      = weekend * off_hours

        return out


# ─────────────────────────────────────────────
# Demo / smoke test
# ─────────────────────────────────────────────

def _generate_sample_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic transaction data for testing."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")

    receivers  = [f"ACC_{i:04d}" for i in range(20)]
    currencies = ["USD", "EUR", "GBP", "JPY", "CHF"]

    offsets_hours = np.cumsum(rng.exponential(scale=12, size=n))
    timestamps    = [base + pd.Timedelta(hours=float(h)) for h in offsets_hours]

    # Inject a few outliers
    amounts = rng.lognormal(mean=6, sigma=1, size=n)      # typical: ~$400
    amounts[50]  *= 20    # spike amount
    amounts[120] *= 15

    recv_choices = rng.choice(receivers, size=n, p=None)
    recv_choices[50]  = "ACC_NEW1"                         # new receiver
    recv_choices[120] = "ACC_NEW2"

    ccy_choices = rng.choice(currencies[:2], size=n)       # mostly USD/EUR
    ccy_choices[80] = "JPY"                                # rare currency

    return pd.DataFrame({
        "transaction_id": [f"TXN_{i:05d}" for i in range(n)],
        "date":           [t.date() for t in timestamps],
        "timestamp":      timestamps,
        "debit_amount":   np.round(amounts, 2),
        "receiver_id":    recv_choices,
        "currency":       ccy_choices,
    })


if __name__ == "__main__":
    print("Generating sample data …")
    sample_df = _generate_sample_data(n=200)

    pipeline = TransactionFeaturePipeline(
        lookback_windows=[7, 30, 90],
        burst_windows_hours=[1, 6, 24],
    )
    features_df = pipeline.fit_transform(sample_df)

    print(f"\nFeature columns ({len(features_df.columns)} total):")
    feat_cols = [c for c in features_df.columns if c not in sample_df.columns]
    for c in feat_cols:
        print(f"  {c}")

    print("\nSample rows (transaction 49–52 — around the injected spike):")
    display_cols = [
        "transaction_id", "debit_amount", "receiver_id", "currency",
        "amt_30d_zscore", "recv_30d_is_new_receiver", "ccy_30d_is_new_currency",
        "flag_large_amt_new_recv", "tx_count_24h",
    ]
    print(features_df.loc[49:52, display_cols].to_string(index=False))

    features_df.to_csv("/mnt/user-data/outputs/transaction_features_sample.csv", index=False)
    print("\nFull output saved to transaction_features_sample.csv")
