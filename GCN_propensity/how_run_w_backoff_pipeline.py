# 1) Build snapshots (from your trades DF) using the builder you already have:
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("temporal_builder", "/mnt/data/build_pyg_temporal_snapshots.py")
temporal_builder = module_from_spec(spec); import sys
sys.modules["temporal_builder"] = temporal_builder; spec.loader.exec_module(temporal_builder)

snapshots = temporal_builder.construct_daily_snapshots(
    df=df_trades,                 # your dataframe
    user_col='client_id',
    item_col='cusip',
    time_col='trade_date',
    qty_col='qty',                # or notional_col='notional'
    maturity_col='maturity',      # optional
    window_days=2,
    calendar='D',
    time_weighting='inverse_delta',
    normalize=False,
    keep_empty_days=False,
    as_torch=True
)

# 2) Import the backoff trainer:
spec = spec_from_file_location("lgcn_backoff", "/mnt/data/lightgcn_causal_training_with_noactivity_backoff.py")
lgcn_backoff = module_from_spec(spec); sys.modules["lgcn_backoff"] = lgcn_backoff; spec.loader.exec_module(lgcn_backoff)

# 3) Split days and train:
all_days = sorted([d for d in snapshots.keys() if d != '__meta__'])
n = len(all_days); n_train = max(1, int(0.6*n)); n_val = max(1, int(0.2*n))
train_days = all_days[:n_train]; val_days = all_days[n_train:n_train+n_val]; test_days = all_days[n_train+n_val:]

model, history, test_metrics = lgcn_backoff.train_lightgcn_w_with_noactivity_backoff(
    snapshots, train_days, val_days=val_days, test_days=test_days,
    emb_dim=64, num_layers=1, lr=1e-3, epochs=5, neg_ratio=10, dns_pool_size=50, eval_every=1
)
