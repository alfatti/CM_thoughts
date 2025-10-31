import pandas as pd

# keep only the two columns, clean a bit, and dedupe exact pairs
pairs = (
    df[['COUNTERPARTY','ACCOUNTID']]
      .dropna()
      .assign(
          COUNTERPARTY=lambda s: s['COUNTERPARTY'].astype(str).str.strip(),
          ACCOUNTID=lambda s: s['ACCOUNTID'].astype(str).str.strip()
      )
      .drop_duplicates()
)

# 1) Many COUNTERPARTY -> one ACCOUNTID (what you expect)
acct_to_ctpy_count = (
    pairs.groupby('ACCOUNTID')['COUNTERPARTY'].nunique()
          .reset_index(name='n_counterparties')
)
many_to_one_accounts = acct_to_ctpy_count[acct_to_ctpy_count['n_counterparties'] > 1]

# (Optional) see the actual list of counterparties per account
acct_to_ctpy_list = (
    pairs.groupby('ACCOUNTID')['COUNTERPARTY']
         .agg(lambda x: sorted(set(x)))
         .reset_index(name='counterparty_list')
)
many_to_one_details = acct_to_ctpy_list[
    acct_to_ctpy_list['counterparty_list'].apply(len) > 1
]

# 2) Sanity check: do any counterparties map to multiple accounts? (would imply many-to-many)
ctpy_to_acct_count = (
    pairs.groupby('COUNTERPARTY')['ACCOUNTID'].nunique()
         .reset_index(name='n_accounts')
)
ctpy_with_multiple_accounts = ctpy_to_acct_count[ctpy_to_acct_count['n_accounts'] > 1]

# Quick summaries
print("Accounts shared by multiple counterparties:")
print(many_to_one_accounts.sort_values('n_counterparties', ascending=False).head(20))

print("\nDetails (account -> list of counterparties):")
print(many_to_one_details.head(10))

print("\nCounterparties that span multiple accounts (if any):")
print(ctpy_with_multiple_accounts.sort_values('n_accounts', ascending=False).head(20))
