# step3_train_fhat.py
from __future__ import annotations
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
import joblib

from step1_clusters import ActionCatalog, Cluster
from step2_context_featurizer import RFQContext, FeatureSpec, RFQFeatureBuilder
from step3_action_payload import ActionPayloadEncoder
from step3_datasets import LoggedExample, FhatDataset
from step3_second_stage import FhatMLP

# --------------------
# Config
# --------------------
SEED = 42
BATCH_SIZE = 1024
LR = 3e-4
EPOCHS = 30
PATIENCE = 5  # early stop
DROPOUT = 0.1
HIDDEN = (256, 128)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTDIR = "./fhat_ckpt"

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --------------------
# Example: load/prepare your logs
# --------------------
def load_logged_examples() -> List[LoggedExample]:
    """
    Replace this stub with your real log reader.
    Expect a list of LoggedExample(rfq, action_id, reward).
    """
    examples: List[LoggedExample] = []

    # --- Example synthetic rows (REMOVE in real usage) ---
    from datetime import datetime
    from step0_action_space import Context as QuoteContext
    # Build a catalog with your default bins
    catalog = ActionCatalog()

    # synthetic contexts & actions
    for i in range(5000):
        rfq = RFQContext(
            cusip="123456AB7" if i % 2 == 0 else "999999ZZZ",
            request_qty=np.random.randint(50_000, 2_000_000),
            market_rate_bps=np.random.choice([80, 120, 350, 500]),
            client_id=f"C{i%25}",
            raw=dict(
                short_interest=np.random.uniform(0, 30),
                days_to_cover=np.random.uniform(0.5, 5),
                inventory_avail=np.random.randint(10_000, 5_000_000),
                price_dollar=np.random.uniform(5, 200),
                cusip_bucket=np.random.choice(["liquid", "special"]),
                client_tier=np.random.choice(["A","B","C"]),
                venue=np.random.choice(["RFS","POV","Chat"]),
                issuer=np.random.choice(["ACME INC","MEGA HOLDINGS","OMEGA PLC","ZEN LLC"]),
            ),
        )
        # choose non-REJECT 90% of time for synthetic demo
        if np.random.rand() < 0.1:
            aid = catalog.reject_action_id
        else:
            # choose some FULL/PARTIAL randomly
            full = np.random.rand() < 0.5
            if full:
                aid = catalog.action_from_tuple(
                    offset_bps=int(np.random.choice([-100,-50,0,50,100])),
                    fill_pct=100
                ).action_id
            else:
                aid = catalog.action_from_tuple(
                    offset_bps=int(np.random.choice([-100,-50,0,50,100])),
                    fill_pct=int(np.random.choice([25,50,75]))
                ).action_id

        # synthetic reward ~ acceptance(prob) * economics
        c = catalog.cluster_of(aid)
        if c == Cluster.REJECT:
            reward = 0.0
        else:
            # economics ~ fill_qty * (market + offset) bps * price
            offset_bps, fill_pct = catalog.tuple_from_id(aid)
            economics_scale = 1e-6  # keep numbers modest
            fee = (rfq.request_qty * (fill_pct/100.0)) * ((rfq.market_rate_bps+offset_bps)/1e4) * rfq.raw["price_dollar"]
            # acceptance falls with higher offset
            accept_prob = float(np.clip(1.2 - 0.004 * max(offset_bps, 0), 0.05, 1.0))
            reward = economics_scale * fee * accept_prob
        examples.append(LoggedExample(rfq=rfq, action_id=aid, reward=float(reward)))
    # --- end synthetic ---
    return examples

# --------------------
# Train / Eval loops
# --------------------
def make_feature_builder(spec: FeatureSpec, train_examples: List[LoggedExample]) -> RFQFeatureBuilder:
    rfqs = [ex.rfq for ex in train_examples]
    return RFQFeatureBuilder(spec).fit(rfqs)

def split_examples(examples: List[LoggedExample], test_size=0.2, val_size=0.1, seed=SEED):
    train_all, test = train_test_split(examples, test_size=test_size, random_state=seed)
    train, dev = train_test_split(train_all, test_size=val_size, random_state=seed)
    return train, dev, test

def train_fhat(train_ds: FhatDataset, dev_ds: FhatDataset,
               hidden=HIDDEN, dropout=DROPOUT, lr=LR, epochs=EPOCHS, patience=PATIENCE):
    device = DEVICE
    in_dim = train_ds.X.shape[1] + train_ds.A.shape[1]
    model = FhatMLP(in_dim=in_dim, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader   = DataLoader(dev_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_loss = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for X, A, y in train_loader:
            X, A, y = X.to(device), A.to(device), y.to(device)
            XA = torch.cat([X, A], dim=1)
            pred = model(XA).squeeze(1)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            tr_loss += loss.item() * X.size(0)
        tr_loss /= len(train_ds)

        # dev
        model.eval()
        dv_loss = 0.0
        with torch.no_grad():
            for X, A, y in dev_loader:
                X, A, y = X.to(device), A.to(device), y.to(device)
                pred = model(torch.cat([X, A], dim=1)).squeeze(1)
                loss = loss_fn(pred, y)
                dv_loss += loss.item() * X.size(0)
        dv_loss /= len(dev_ds)

        print(f"Epoch {epoch:02d} | train {tr_loss:.6f} | dev {dv_loss:.6f}")

        if dv_loss + 1e-9 < best_loss:
            best_loss = dv_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def main():
    set_seed(SEED)
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) Load logs
    examples = load_logged_examples()

    # 2) Split
    train_ex, dev_ex, test_ex = split_examples(examples, test_size=0.2, val_size=0.1)

    # 3) Build catalog, payload encoder
    catalog = ActionCatalog()
    payload_enc = ActionPayloadEncoder(catalog)

    # 4) Feature spec (adjust to your real columns)
    spec = FeatureSpec(
        numeric=["request_qty", "market_rate_bps", "short_interest", "days_to_cover", "inventory_avail", "price_dollar"],
        categorical=["cusip_bucket", "client_tier", "venue"],
        hashed=[("issuer", 64)],
        poly2_for_pi0=["request_qty", "market_rate_bps", "short_interest", "inventory_avail"],
    )

    # 5) Fit feature builder on train RFQs
    fb = make_feature_builder(spec, train_ex)

    # 6) Build datasets (REJECT samples auto-excluded)
    train_ds = FhatDataset(train_ex, fb, catalog, payload_enc)
    dev_ds   = FhatDataset(dev_ex,   fb, catalog, payload_enc)
    test_ds  = FhatDataset(test_ex,  fb, catalog, payload_enc)

    # 7) Train
    model = train_fhat(train_ds, dev_ds)

    # 8) Evaluate (MSE/Huber on test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    loss_fn = nn.SmoothL1Loss(beta=1.0)
    model.eval()
    tot = 0.0
    n = 0
    with torch.no_grad():
        for X, A, y in test_loader:
            X, A, y = X.to(DEVICE), A.to(DEVICE), y.to(DEVICE)
            pred = model(torch.cat([X, A], dim=1)).squeeze(1)
            loss = loss_fn(pred, y)
            tot += loss.item() * X.size(0)
            n += X.size(0)
    print(f"Test SmoothL1: {tot/n:.6f}")

    # 9) Save artifacts
    torch.save(model.state_dict(), os.path.join(OUTDIR, "fhat.pt"))
    joblib.dump(fb,            os.path.join(OUTDIR, "feature_builder.joblib"))
    joblib.dump(payload_enc,   os.path.join(OUTDIR, "payload_encoder.joblib"))
    joblib.dump(catalog,       os.path.join(OUTDIR, "action_catalog.joblib"))
    print("Saved to", OUTDIR)

if __name__ == "__main__":
    main()
