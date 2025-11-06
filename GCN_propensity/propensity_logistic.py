"""
account_propensity_logistic.py

Per-account propensity model:
- Input: binary trade/interactions time series for one account across many CUSIPs
- Output: P(interest in a given target CUSIP at next time step)

Dependencies:
    pip install pandas numpy scikit-learn
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Tuple, Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
)
from sklearn.base import BaseEstimator


# ---------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------

def _days_since_last_event(series: pd.Series) -> pd.Series:
    """
    Given a 0/1 series, compute days since last 1.
    For indices before the first 1, returns NaN (can later be filled).
    """
    idx = np.arange(len(series))
    # positions where event happened
    event_pos = np.where(series.values == 1, idx, np.nan)
    event_pos = pd.Series(event_pos, index=series.index)
    last_event_pos = event_pos.ffill()
    days_since = idx - last_event_pos.values
    days_since[pd.isna(last_event_pos.values)] = np.nan
    return pd.Series(days_since, index=series.index)


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
    """
    Build a tabular dataset for (account, target_cusip).

    Parameters
    ----------
    df_account : DataFrame
        Rows = time (sorted index), columns = CUSIP IDs, values = 0/1.
    target_cusip : str
        CUSIP column to model propensity for.
    window_short : int
        Short rolling window (e.g. 5).
    window_long : int
        Long rolling window (e.g. 20).

    Returns
    -------
    AccountTargetDataset
        X: features DataFrame (per time step t)
        y: 0/1 label: interest in target at t+1
        dates: timestamp for each row (aligned with X,y)
    """
    assert target_cusip in df_account.columns, "Target CUSIP not in df_account."

    # Ensure sorted by time
    df_account = df_account.sort_index()
    T, M = df_account.shape

    # --- Basic series ---
    s_target = df_account[target_cusip]               # 0/1 series for target cusip
    s_any = (df_account.sum(axis=1) > 0).astype(int)  # any trade at time t
    s_total_trades = df_account.sum(axis=1)           # number of trades across all cusips at time t
    s_distinct_cusips = (df_account > 0).sum(axis=1)  # distinct cusips traded at time t

    # --- Rolling features (per time t, to predict t+1) ---
    # Target-specific
    f_target_last_1d = s_target  # trade in j at t
    f_target_last_short = s_target.rolling(window_short, min_periods=1).sum()
    f_target_last_long = s_target.rolling(window_long, min_periods=1).sum()

    # Portfolio activity
    f_any_last_short = s_any.rolling(window_short, min_periods=1).sum()
    f_any_last_long = s_any.rolling(window_long, min_periods=1).sum()

    f_total_trades_short = s_total_trades.rolling(window_short, min_periods=1).sum()
    f_total_trades_long = s_total_trades.rolling(window_long, min_periods=1).sum()

    f_distinct_cusips_short = s_distinct_cusips.rolling(window_short, min_periods=1).sum()
    f_distinct_cusips_long = s_distinct_cusips.rolling(window_long, min_periods=1).sum()

    # Ratios (avoid divide-by-zero)
    f_ratio_target_short = f_target_last_short / (f_any_last_short.replace(0, np.nan))
    f_ratio_target_long = f_target_last_long / (f_any_last_long.replace(0, np.nan))

    # --- Recency features ---
    f_days_since_target = _days_since_last_event(s_target)
    f_days_since_any = _days_since_last_event(s_any)

    # Fill NaNs in recency with a large number (no previous activity yet)
    max_len = len(df_account)
    f_days_since_target = f_days_since_target.fillna(max_len)
    f_days_since_any = f_days_since_any.fillna(max_len)

    # Fill NaNs in ratio features with 0 (no activity → share 0)
    f_ratio_target_short = f_ratio_target_short.fillna(0.0)
    f_ratio_target_long = f_ratio_target_long.fillna(0.0)

    # --- Calendar features (optional)
    # Only if index is datetime-like
    if isinstance(df_account.index, pd.DatetimeIndex):
        day_of_week = df_account.index.dayofweek  # 0=Monday
        month = df_account.index.month
    else:
        # fallback: integer index
        day_of_week = pd.Series(0, index=df_account.index)
        month = pd.Series(1, index=df_account.index)

    # --- Build feature DataFrame ---
    features = pd.DataFrame(
        {
            # target-focused
            "f_target_last_1d": f_target_last_1d,
            f"f_target_last_{window_short}d": f_target_last_short,
            f"f_target_last_{window_long}d": f_target_last_long,

            # portfolio activity
            f"f_any_last_{window_short}d": f_any_last_short,
            f"f_any_last_{window_long}d": f_any_last_long,
            f"f_total_trades_last_{window_short}d": f_total_trades_short,
            f"f_total_trades_last_{window_long}d": f_total_trades_long,
            f"f_distinct_cusips_last_{window_short}d": f_distinct_cusips_short,
            f"f_distinct_cusips_last_{window_long}d": f_distinct_cusips_long,

            # ratios
            f"f_ratio_target_{window_short}d": f_ratio_target_short,
            f"f_ratio_target_{window_long}d": f_ratio_target_long,

            # recency
            "f_days_since_target": f_days_since_target,
            "f_days_since_any": f_days_since_any,

            # calendar
            "day_of_week": day_of_week,
            "month": month,
        },
        index=df_account.index,
    )

    # --- Labels: interest in target at t+1 ---
    y = s_target.shift(-1)  # label at next time step
    features = features.iloc[:-1]  # drop last row (no next-step label)
    y = y.iloc[:-1]

    return AccountTargetDataset(
        X=features,
        y=y.astype(int),
        dates=features.index,
    )


# ---------------------------------------------------------------------
# Model container with sklearn-ish attributes
# ---------------------------------------------------------------------

@dataclass
class TrainedAccountModel(BaseEstimator):
    """
    A container for a trained account-level propensity model.

    Has sklearn-like behaviour (can be pickled with joblib) and
    stores train/test splits for inspection.
    """
    account_id: str
    target_cusip: str
    pipeline: Pipeline

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray

    dates_train: pd.DatetimeIndex
    dates_test: pd.DatetimeIndex

    metrics: Dict[str, Any]

    # Convenience wrappers to feel like an sklearn estimator
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)


# ---------------------------------------------------------------------
# Train / test split and model training
# ---------------------------------------------------------------------

def train_propensity_model_for_account(
    account_id: str,
    df_account: pd.DataFrame,
    target_cusip: str,
    test_size: float = 0.2,
    window_short: int = 5,
    window_long: int = 20,
) -> TrainedAccountModel:
    """
    Build features and train a logistic regression model for (account_id, target_cusip).

    Parameters
    ----------
    account_id : str
        Identifier for the account (used for bookkeeping).
    df_account : DataFrame
        Binary interaction matrix for this account (time x CUSIPs).
    target_cusip : str
        Which CUSIP to predict propensity for.
    test_size : float
        Fraction of data used as test set (time-based split).
    window_short, window_long : int
        Rolling windows for feature engineering.

    Returns
    -------
    TrainedAccountModel
    """

    dataset = build_features_for_account_target(
        df_account=df_account,
        target_cusip=target_cusip,
        window_short=window_short,
        window_long=window_long,
    )

    X = dataset.X
    y = dataset.y.values
    dates = dataset.dates

    n = len(X)
    n_test = int(np.floor(n * test_size))
    n_train = n - n_test

    X_train = X.iloc[:n_train].copy()
    y_train = y[:n_train]
    dates_train = dates[:n_train]

    X_test = X.iloc[n_train:].copy()
    y_test = y[n_train:]
    dates_test = dates[n_train:]

    # Build pipeline: scaler + logistic regression
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    class_weight="balanced",  # cope with heavy 0/1 imbalance
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)

    # Predictions: probabilities for class 1
    y_train_proba = pipe.predict_proba(X_train)[:, 1]
    y_test_proba = pipe.predict_proba(X_test)[:, 1]

    # Threshold 0.5 for hard labels (you can tweak this)
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    metrics: Dict[str, Any] = {}

    # AUC / PR-AUC / Brier
    try:
        metrics["train_auc"] = roc_auc_score(y_train, y_train_proba)
    except ValueError:
        metrics["train_auc"] = np.nan

    try:
        metrics["test_auc"] = roc_auc_score(y_test, y_test_proba)
    except ValueError:
        metrics["test_auc"] = np.nan

    try:
        metrics["test_avg_precision"] = average_precision_score(y_test, y_test_proba)
    except ValueError:
        metrics["test_avg_precision"] = np.nan

    metrics["test_brier"] = brier_score_loss(y_test, y_test_proba)

    # Optional – text report for quick inspection
    metrics["test_classification_report"] = classification_report(
        y_test,
        y_test_pred,
        output_dict=False,
        zero_division=0,
    )

    print(f"== Account {account_id}, Target CUSIP {target_cusip} ==")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Test AUC:      {metrics['test_auc']:.4f}")
    print(f"Test PR-AUC:   {metrics['test_avg_precision']:.4f}")
    print(f"Test Brier:    {metrics['test_brier']:.4f}")
    print("Classification report:")
    print(metrics["test_classification_report"])

    return TrainedAccountModel(
        account_id=account_id,
        target_cusip=target_cusip,
        pipeline=pipe,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        dates_train=dates_train,
        dates_test=dates_test,
        metrics=metrics,
    )


# ---------------------------------------------------------------------
# Example usage (you adapt this to your environment)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # You will replace this part with your actual data loading.
    #
    # Assumed data structure:
    #   client_cusip_binary: dict[account_id] -> DataFrame
    #   each DataFrame: index = timestamps (DatetimeIndex), columns = CUSIP IDs, values = 0/1

    # Example placeholders:
    # client_cusip_binary = pickle.load(open("client_cusip_binary.pkl", "rb"))

    # For now, I’ll create a random binary df for one account with 100 days and 10 CUSIPs
    dates = pd.date_range("2022-01-01", periods=100, freq="D")
    cusips = [f"CUSIP_{i}" for i in range(10)]
    rng = np.random.default_rng(123)
    df_example = pd.DataFrame(
        rng.binomial(1, 0.02, size=(100, 10)),  # very sparse
        index=dates,
        columns=cusips,
    )
    account_id_example = "ACCOUNT_ABC"
    target_cusip_example = "CUSIP_3"

    model_example = train_propensity_model_for_account(
        account_id=account_id_example,
        df_account=df_example,
        target_cusip=target_cusip_example,
        test_size=0.2,
        window_short=5,
        window_long=20,
    )

    # Example of saving with joblib:
    # from joblib import dump
    # dump(model_example, "ACCOUNT_ABC_CUSIP_3_propensity.pkl")
