"""
account_propensity_logistic.py
--------------------------------
Per-account propensity model using logistic regression.

Input:
    - df_account: binary matrix (index=time, columns=CUSIPs, values={0,1})
    - target_cusip: one column name (CUSIP) to predict next-step interest for

Output:
    - TrainedAccountModel object with fitted model, metrics, and
      train/test data partitions

Dependencies:
    pip install pandas numpy scikit-learn
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
)


# ---------------------------------------------------------------------
# Helper: recency computation
# ---------------------------------------------------------------------

def _days_since_last_event(series: pd.Series) -> pd.Series:
    idx = np.arange(len(series))
    event_pos = np.where(series.values == 1, idx, np.nan)
    event_pos = pd.Series(event_pos, index=series.index)
    last_event_pos = event_pos.ffill()
    days_since = idx - last_event_pos.values
    days_since[pd.isna(last_event_pos.values)] = np.nan
    return pd.Series(days_since, index=series.index)


# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------

@dataclass
class AccountTargetDataset:
    X: pd.DataFrame
    y: pd.Series
    dates: pd.DatetimeIndex


def build_features_for_account_target(
    df_account: pd.DataFrame,
    target_cusip: str,
    window_short: int = 5,
    window_long: int = 20,
) -> AccountTargetDataset:
    """Create time-based features for one account–CUSIP pair."""
    assert target_cusip in df_account.columns, "Target CUSIP missing."

    df_account = df_account.sort_index()
    s_target = df_account[target_cusip]
    s_any = (df_account.sum(axis=1) > 0).astype(int)
    s_total = df_account.sum(axis=1)
    s_distinct = (df_account > 0).sum(axis=1)

    # Rolling counts
    f_t_short = s_target.rolling(window_short, min_periods=1).sum()
    f_t_long = s_target.rolling(window_long, min_periods=1).sum()
    f_any_short = s_any.rolling(window_short, min_periods=1).sum()
    f_any_long = s_any.rolling(window_long, min_periods=1).sum()
    f_total_short = s_total.rolling(window_short, min_periods=1).sum()
    f_total_long = s_total.rolling(window_long, min_periods=1).sum()
    f_distinct_short = s_distinct.rolling(window_short, min_periods=1).sum()
    f_distinct_long = s_distinct.rolling(window_long, min_periods=1).sum()

    # Ratios
    f_ratio_short = (f_t_short / f_any_short.replace(0, np.nan)).fillna(0)
    f_ratio_long = (f_t_long / f_any_long.replace(0, np.nan)).fillna(0)

    # Recency
    f_days_target = _days_since_last_event(s_target).fillna(len(df_account))
    f_days_any = _days_since_last_event(s_any).fillna(len(df_account))

    # Calendar
    if isinstance(df_account.index, pd.DatetimeIndex):
        day_of_week = df_account.index.dayofweek
        month = df_account.index.month
    else:
        day_of_week = pd.Series(0, index=df_account.index)
        month = pd.Series(1, index=df_account.index)

    features = pd.DataFrame({
        "f_target_last_1d": s_target,
        f"f_target_last_{window_short}d": f_t_short,
        f"f_target_last_{window_long}d": f_t_long,
        f"f_any_last_{window_short}d": f_any_short,
        f"f_any_last_{window_long}d": f_any_long,
        f"f_total_trades_{window_short}d": f_total_short,
        f"f_total_trades_{window_long}d": f_total_long,
        f"f_distinct_cusips_{window_short}d": f_distinct_short,
        f"f_distinct_cusips_{window_long}d": f_distinct_long,
        f"f_ratio_target_{window_short}d": f_ratio_short,
        f"f_ratio_target_{window_long}d": f_ratio_long,
        "f_days_since_target": f_days_target,
        "f_days_since_any": f_days_any,
        "day_of_week": day_of_week,
        "month": month,
    }, index=df_account.index)

    y = s_target.shift(-1).iloc[:-1]
    X = features.iloc[:-1]

    return AccountTargetDataset(X=X, y=y.astype(int), dates=X.index)


# ---------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------

@dataclass
class TrainedAccountModel:
    account_id: str
    target_cusip: str
    pipeline: Pipeline
    metrics: Dict[str, Any]
    dates_train: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


def train_propensity_model_for_account(
    account_id: str,
    df_account: pd.DataFrame,
    target_cusip: str,
    test_size: float = 0.2,
    window_short: int = 5,
    window_long: int = 20,
) -> TrainedAccountModel:
    """Train a logistic regression model for one account–CUSIP pair."""
    dataset = build_features_for_account_target(
        df_account, target_cusip, window_short, window_long
    )

    X, y, dates = dataset.X, dataset.y.values, dataset.dates
    n = len(X)
    n_test = int(np.floor(n * test_size))
    n_train = n - n_test

    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    dates_train, dates_test = dates[:n_train], dates[n_train:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        )),
    ])

    pipeline.fit(X_train, y_train)

    y_train_prob = pipeline.predict_proba(X_train)[:, 1]
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    # Metrics
    metrics = {}
    for name, func in [
        ("train_auc", roc_auc_score),
        ("test_auc", roc_auc_score),
    ]:
        try:
            if "train" in name:
                metrics[name] = func(y_train, y_train_prob)
            else:
                metrics[name] = func(y_test, y_test_prob)
        except ValueError:
            metrics[name] = np.nan

    try:
        metrics["test_avg_precision"] = average_precision_score(y_test, y_test_prob)
    except ValueError:
        metrics["test_avg_precision"] = np.nan

    metrics["test_brier"] = brier_score_loss(y_test, y_test_prob)
    metrics["classification_report"] = classification_report(
        y_test, y_test_pred, zero_division=0
    )

    print(f"\n=== Account {account_id}, Target {target_cusip} ===")
    print(f"Train={len(X_train)}  Test={len(X_test)}")
    print(f"AUC={metrics['test_auc']:.3f}  PR-AUC={metrics['test_avg_precision']:.3f}")
    print(metrics["classification_report"])

    return TrainedAccountModel(
        account_id=account_id,
        target_cusip=target_cusip,
        pipeline=pipeline,
        metrics=metrics,
        dates_train=dates_train,
        dates_test=dates_test,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


# ---------------------------------------------------------------------
# Example usage (replace with your own data)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Example with synthetic data
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    cusips = [f"CUSIP_{i}" for i in range(12)]
    rng = np.random.default_rng(123)
    df_account = pd.DataFrame(
        rng.binomial(1, 0.02, size=(120, len(cusips))),
        index=dates,
        columns=cusips,
    )

    account_id = "ACCT_001"
    target_cusip = "CUSIP_3"

    model = train_propensity_model_for_account(
        account_id=account_id,
        df_account=df_account,
        target_cusip=target_cusip,
        test_size=0.2,
    )

    # Example: inspect what's inside
    print("\nStored attributes:")
    print("X_train shape:", model.X_train.shape)
    print("y_train shape:", model.y_train.shape)
    print("X_test shape:", model.X_test.shape)
    print("y_test shape:", model.y_test.shape)
