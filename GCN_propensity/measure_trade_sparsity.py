import pandas as pd

# assume df has columns ACCOUNTID, CUSIP, TRADEDATE (as datetime)
df = pd.read_csv("trades.csv", parse_dates=["TRADEDATE"])
df = df.sort_values(["ACCOUNTID", "CUSIP", "TRADEDATE"])

# compute gap in days between consecutive trades per account-cusip
df["days_since_last_trade"] = (
    df.groupby(["ACCOUNTID", "CUSIP"])["TRADEDATE"]
      .diff()
      .dt.days
)

# drop the first trade per pair (NaN gap)
gaps = df.dropna(subset=["days_since_last_trade"])

# summarize
pair_avg = (
    gaps.groupby(["ACCOUNTID", "CUSIP"])["days_since_last_trade"]
         .mean()
         .reset_index(name="avg_gap_days")
)

# overall summary stats
print(pair_avg["avg_gap_days"].describe(percentiles=[0.5, 0.9, 0.99]))

overall_mean = pair_avg["avg_gap_days"].mean()
print(f"\nAverage time between trades (same account–CUSIP): {overall_mean:.2f} days")
