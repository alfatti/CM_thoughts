# step3_train_fhat.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import joblib

from step1_clusters import ActionCatalog, Cluster
from step2_context_featurizer import RFQContext, RFQFeatureBuilder, FeatureSpec
from step3_datasets import LoggedExample, ActionPayloadEncoder, FhatDesignMatrixBuilder
from step3_fhat_models import FhatModel, FhatConfig

def train_fhat(
    train_logs: List[LoggedExample],
    dev_logs: List[LoggedExample],
    feature_spec: FeatureSpec,
    kind: str = "mlp",  # "mlp" or "hgb"
    save_prefix: str = "./artifacts/fhat",
):
    # 1) Fit featurizer on TRAIN RFQs (you can also fit on train+dev)
    fb = RFQFeatureBuilder(feature_spec).fit([z.rfq for z in train_logs])

    # 2) Catalog & payload encoder
    catalog = ActionCatalog()  # use your rate/fill defaults or pass custom bins
    ape = ActionPayloadEncoder(catalog)

    # 3) Build design matrices
    dm = FhatDesignMatrixBuilder(fb, ape)
    X_tr, A_tr, y_tr = dm.build_supervised_arrays(train_logs)
    X_dv, A_dv, y_dv = dm.build_supervised_arrays(dev_logs)

    # 4) Train model
    cfg = FhatConfig(kind=kind)
    fhat = FhatModel(cfg)
    metrics = fhat.fit(X_tr, A_tr, y_tr, X_val=X_dv, A_val=A_dv, y_val=y_dv)
    print(f"[fhat/{kind}] dev metrics:", metrics)

    # 5) Save artifacts for later inference (POTEC stage-1 training & serving)
    #    We keep separate files so you can swap models later.
    joblib.dump(fb, f"{save_prefix}.featurizer.joblib")
    joblib.dump(ape, f"{save_prefix}.payload_encoder.joblib")
    joblib.dump(catalog, f"{save_prefix}.catalog.joblib")
    fhat.save(f"{save_prefix}.{kind}.bin")
    print(f"Saved: {save_prefix}.*")

    return fhat, fb, ape, catalog, dm

# ----------------------
# Example synthetic run
# ----------------------
if __name__ == "__main__":
    # 0) Spec (align with earlier step2 example)
    spec = FeatureSpec(
        numeric=["request_qty","market_rate_bps","short_interest","inventory_avail","price_dollar"],
        categorical=["cusip_bucket","client_tier","venue"],
        hashed=[("issuer", 32)],
        poly2_for_pi0=["request_qty","market_rate_bps","short_interest","inventory_avail"],
    )

    # 1) Build some fake logs just to exercise the code
    import random
    def mk_rfqi(i: int) -> RFQContext:
        return RFQContext(
            cusip=f"{100000+i:09d}",
            request_qty=float(random.choice([1e5,3e5,1e6,2e6])),
            market_rate_bps=int(random.choice([50,120,300,600])),
            client_id=f"C{i%17:03d}",
            raw=dict(
                short_interest=random.uniform(0,30),
                inventory_avail=float(random.choice([1e5,5e5,2e6,5e6])),
                price_dollar=random.uniform(5,100),
                cusip_bucket=random.choice(["liquid","special","mid"]),
                client_tier=random.choice(["A","B","C"]),
                venue=random.choice(["RFQ","RFS","POV"]),
                issuer=random.choice(["ACME","OMEGA","MEGA","ALPHA","BETA"]),
            ),
        )

    cat = ActionCatalog()
    # make a simple synthetic reward: higher when offset small positive & fill reasonable and inventory big
    def simulate_reward(rfq: RFQContext, aid: int) -> float:
        from step0_action_space import Context as QuoteContext
        act = cat.action_from_id(aid)
        dec = cat.realize(rfq.as_quote_context(), act)
        base = (dec.fill_qty/ max(rfq.request_qty,1.0))  # favor bigger fills
        rate_pen = -abs(dec.quote_rate_bps - rfq.market_rate_bps - 30)/600.0  # prefer ~+30bps
        inv_bonus = min(1.0, rfq.raw["inventory_avail"] / max(rfq.request_qty,1.0))
        if aid == cat.reject_action_id:
            return 0.0
        return float(100.0 * base + 20.0 * rate_pen + 10.0 * inv_bonus + np.random.normal(0, 3))

    logs: list[LoggedExample] = []
    for i in range(3000):
        rfq = mk_rfqi(i)
        # pretend logging policy: 15% reject, 35% full, 50% partial
        dice = random.random()
        if dice < 0.15:
            aid = cat.reject_action_id
        elif dice < 0.50:
            # FULL: pick random offset with fill=100
            full_ids = cat.actions_in_cluster(Cluster.FULL)
            aid = random.choice(full_ids)
        else:
            aid = random.choice(cat.actions_in_cluster(Cluster.PARTIAL))
        r = simulate_reward(rfq, aid)
        logs.append(LoggedExample(rfq=rfq, action_id=aid, reward=r))

    random.shuffle(logs)
    split = int(0.8*len(logs))
    train_logs, dev_logs = logs[:split], logs[split:]

    # 2) Train both variants (comment one if you like)
    train_fhat(train_logs, dev_logs, spec, kind="mlp", save_prefix="./artifacts/fhat_mlp")
    train_fhat(train_logs, dev_logs, spec, kind="hgb", save_prefix="./artifacts/fhat_hgb")
