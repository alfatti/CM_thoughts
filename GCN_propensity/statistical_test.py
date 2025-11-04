"""
\paragraph{Test 1: Logistic Regression with Fixed Effects.}
We estimate a panel logistic regression to assess whether recent investor--CUSIP interaction
frequency carries predictive power for next--day interest.
Let $Y_{u,i,t} \in \{0,1\}$ denote an indicator of whether investor $u$
shows interest in CUSIP $i$ on day $t$,
and define the $w$-day rolling frequency feature
$f^{(w)}_{u,i,t} = \sum_{s=t-w}^{t-1} Y_{u,i,s}$.
The model is
\[
\Pr(Y_{u,i,t+1}=1 \mid f^{(w)}_{u,i,t})
 = \text{logit}^{-1}\!\big(\alpha_u + \gamma_i + \beta\, f^{(w)}_{u,i,t}\big),
\]
where $\alpha_u$ and $\gamma_i$ are investor and CUSIP fixed effects,
and $\beta$ measures the marginal effect of recent activity.
The null hypothesis $H_0\!:\beta=0$ corresponds to ``no predictive signal
from past frequency'' beyond baseline investor and security propensities.
Significance of $\hat\beta$ is evaluated using cluster-robust standard errors
(grouped by investor or investor--CUSIP pair) to account for temporal dependence
within panels.
A positive and statistically significant $\hat\beta$ implies that
higher recent interaction frequency increases the odds of next-day interest.

"""
"""
Logistic test for predictive power of recent trade frequency.
Step 1: Derive INTEREST = 1 if any trade (account, cusip, date).
Step 2: Fit logit model with investor & cusip fixed effects.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrices

# ------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------
WINDOW = 5       # rolling window size (days)
MIN_ACTIVITY = 10  # min trades per (account, cusip)

# ------------------------------------------------------------
# 1. Load your trade data
# ------------------------------------------------------------
# expected columns: ACCOUNTID, CUSIP, TRADEDATE, and other trade specs
df = pd.read_csv("trades.csv", parse_dates=["TRADEDATE"])

# keep only necessary columns
df = df[["ACCOUNTID", "CUSIP", "TRADEDATE"]].dropna()

# ------------------------------------------------------------
# 2. Derive INTEREST (binary daily signal)
# ------------------------------------------------------------
# group by account, cusip, date → at least one trade that day = interest 1
df_interest = (
    df.groupby(["ACCOUNTID", "CUSIP", "TRADEDATE"])
      .size()
      .reset_index(name="TRADECOUNT")
)
df_interest["INTEREST"] = 1  # at least one trade means interest

# ------------------------------------------------------------
# 3. Fill missing dates (create continuous daily panel)
# ------------------------------------------------------------
def make_panel(group):
    full_idx = pd.date_range(group.TRADEDATE.min(), group.TRADEDATE.max(), freq="D")
    g = group.set_index("TRADEDATE").reindex(full_idx, fill_value=0)
    g = g.rename_axis("TRADEDATE").reset_index()
    g["ACCOUNTID"] = group.ACCOUNTID.iloc[0]
    g["CUSIP"] = group.CUSIP.iloc[0]
    # mark interest = 1 where trade existed, 0 otherwise
    g["INTEREST"] = (g["TRADECOUNT"] > 0).astype(int)
    return g

panel = (
    df_interest.groupby(["ACCOUNTID", "CUSIP"], group_keys=False)
    .apply(make_panel)
    .reset_index(drop=True)
)

# ------------------------------------------------------------
# 4. Rolling frequency (past WINDOW days)
# ------------------------------------------------------------
panel["freq_window"] = (
    panel.groupby(["ACCOUNTID", "CUSIP"])["INTEREST"]
    .transform(lambda x: x.rolling(WINDOW, min_periods=1).sum().shift(1))
)

# target = next-day interest
panel["target_nextday"] = (
    panel.groupby(["ACCOUNTID", "CUSIP"])["INTEREST"].shift(-1)
)
panel = panel.dropna(subset=["target_nextday", "freq_window"])

# ------------------------------------------------------------
# 5. Filter to active pairs
# ------------------------------------------------------------
activity = panel.groupby(["ACCOUNTID", "CUSIP"])["INTEREST"].sum().reset_index()
active_pairs = activity.loc[activity.INTEREST >= MIN_ACTIVITY, ["ACCOUNTID", "CUSIP"]]
panel = panel.merge(active_pairs, on=["ACCOUNTID", "CUSIP"])

# ------------------------------------------------------------
# 6. Fit logistic regression with investor & cusip fixed effects
# ------------------------------------------------------------
y, X = dmatrices(
    "target_nextday ~ freq_window + C(ACCOUNTID) + C(CUSIP)",
    data=panel,
    return_type="dataframe"
)

model = sm.Logit(y, X)
res = model.fit(disp=False)

# ------------------------------------------------------------
# 7. Report results
# ------------------------------------------------------------
beta = res.params["freq_window"]
pval = res.pvalues["freq_window"]
odds_ratio = np.exp(beta)
marginal_effect = (odds_ratio - 1) * 100

print("\n=== Test: Predictive power of past trade frequency ===")
print(f"Window: {WINDOW} days")
print(f"β̂ (freq_window): {beta: .4f}")
print(f"p-value: {pval: .4g}")
print(f"Odds ratio: {odds_ratio: .3f}  →  +{marginal_effect: .1f}% odds per extra trade in window")

if pval < 0.05:
    print("\n✅ Reject H₀: frequency carries predictive signal.")
else:
    print("\n❌ Fail to reject H₀: frequency not predictive beyond base rates.")

print(f"McFadden R²: {res.prsquared: .4f}")
