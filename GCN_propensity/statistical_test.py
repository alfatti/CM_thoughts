"""
Statistical test: does recent interaction frequency predict next-day interest?
Model: logit(P(Y_{u,i,t+1}=1)) = α_u + γ_i + β * freq_{u,i,t}^{(w)}
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices

# ------------------------------------------------------------
# 1. Parameters
# ------------------------------------------------------------
WINDOW = 5   # rolling window in days
MIN_ACTIVITY = 10  # minimum total events per investor–CUSIP to include

# ------------------------------------------------------------
# 2. Load data
# ------------------------------------------------------------
# expected columns: ACCOUNTID, CUSIP, TRADE_DATE (datetime), INTEREST (0/1)
df = pd.read_csv("trades.csv", parse_dates=["TRADE_DATE"])

# ensure proper types
df = df.sort_values(["ACCOUNTID", "CUSIP", "TRADE_DATE"])
df["INTEREST"] = (df["INTEREST"] > 0).astype(int)

# ------------------------------------------------------------
# 3. Aggregate to daily panel (fill missing dates per pair)
# ------------------------------------------------------------
def make_panel(group):
    """ensure consecutive daily rows for each (ACCOUNTID, CUSIP)"""
    full_idx = pd.date_range(group.TRADE_DATE.min(), group.TRADE_DATE.max(), freq="D")
    g = group.set_index("TRADE_DATE").reindex(full_idx, fill_value=0)
    g = g.rename_axis("TRADE_DATE").reset_index()
    g["ACCOUNTID"] = group.ACCOUNTID.iloc[0]
    g["CUSIP"] = group.CUSIP.iloc[0]
    return g

panel = (
    df.groupby(["ACCOUNTID", "CUSIP"], group_keys=False)
      .apply(make_panel)
      .reset_index(drop=True)
)

# ------------------------------------------------------------
# 4. Create rolling frequency feature
# ------------------------------------------------------------
panel["freq_window"] = (
    panel.groupby(["ACCOUNTID", "CUSIP"])["INTEREST"]
         .transform(lambda x: x.rolling(WINDOW, min_periods=1).sum().shift(1))
)

# target: next-day interest
panel["target_nextday"] = (
    panel.groupby(["ACCOUNTID", "CUSIP"])["INTEREST"].shift(-1)
)

panel = panel.dropna(subset=["target_nextday"])  # drop last day per series
panel = panel.query("freq_window.notnull()")

# filter rarely active pairs to avoid quasi-separation
activity = panel.groupby(["ACCOUNTID", "CUSIP"])["INTEREST"].sum().reset_index()
active_pairs = activity.loc[activity.INTEREST >= MIN_ACTIVITY, ["ACCOUNTID", "CUSIP"]]
panel = panel.merge(active_pairs, on=["ACCOUNTID", "CUSIP"])

# ------------------------------------------------------------
# 5. Logistic regression with fixed effects (dummy encoding)
# ------------------------------------------------------------
# build design matrices using patsy (adds dummies automatically)
y, X = dmatrices(
    "target_nextday ~ freq_window + C(ACCOUNTID) + C(CUSIP)",
    data=panel,
    return_type="dataframe"
)

model = sm.Logit(y, X)
res = model.fit(disp=False)

# ------------------------------------------------------------
# 6. Report results
# ------------------------------------------------------------
beta = res.params["freq_window"]
pval = res.pvalues["freq_window"]
odds_ratio = np.exp(beta)
marginal_effect = (odds_ratio - 1) * 100  # % change in odds per +1 freq

print("\n=== Test: Predictive power of past frequency ===")
print(f"Rolling window (days): {WINDOW}")
print(f"β̂ (freq_window): {beta: .4f}")
print(f"p-value: {pval: .4g}")
print(f"Odds ratio: {odds_ratio: .3f}  →  +{marginal_effect: .1f}% odds per extra interaction")

if pval < 0.05:
    print("\n✅ Reject H₀: frequency carries significant predictive signal.")
else:
    print("\n❌ Fail to reject H₀: frequency not predictive beyond base rates.")

# optional: pseudo-R²
print(f"McFadden R²: {res.prsquared: .4f}")
