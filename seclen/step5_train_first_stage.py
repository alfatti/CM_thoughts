# step5_train_first_stage.py
from __future__ import annotations
import random
import numpy as np
import joblib

from step1_clusters import ActionCatalog, Cluster
from step2_context_featurizer import RFQContext, RFQFeatureBuilder, FeatureSpec
from step3_datasets import LoggedExample, ActionPayloadEncoder, FhatDesignMatrixBuilder
from step3_fhat_models import FhatModel, FhatConfig
from step4_logging_policy import LoggingPolicyModel, Pi0Config
from step5_first_stage_policy import (
    FirstStageConfig,
    ShallowPolicyNN,
    build_stage1_tensors,
    train_first_stage,
)

# ---------- synthetic data ----------
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

def simulate_reward(cat: ActionCatalog, rfq: RFQContext, aid: int) -> float:
    # simple stationary synthetic: prefer ~+30bps offset, full fills, plentiful inventory
    act = cat.action_from_id(aid)
    dec = cat.realize(rfq.as_quote_context(), act)
    base = (dec.fill_qty / max(rfq.request_qty,1.0))                 # bigger fills
    rate_pen = -abs(dec.quote_rate_bps - rfq.market_rate_bps - 30)/600.0
    inv_bonus = min(1.0, rfq.raw["inventory_avail"] / max(rfq.request_qty,1.0))
    if aid == cat.reject_action_id:
        return 0.0
    return float(100.0 * base + 20.0 * rate_pen + 10.0 * inv_bonus + np.random.normal(0, 3))

def build_logs(n=5000):
    cat = ActionCatalog()
    logs = []
    for i in range(n):
        rfq = mk_rfqi(i)
        u = random.random()
        if u < 0.15:
            aid = cat.reject_action_id
        elif u < 0.50:
            aid = random.choice(cat.actions_in_cluster(Cluster.FULL))
        else:
            aid = random.choice(cat.actions_in_cluster(Cluster.PARTIAL))
        r = simulate_reward(cat, rfq, aid)
        logs.append(LoggedExample(rfq=rfq, action_id=aid, reward=r))
    random.shuffle(logs)
    cut = int(0.8 * len(logs))
    return logs[:cut], logs[cut:]

# ---------- main ----------
if __name__ == "__main__":
    random.seed(0); np.random.seed(0)

    # Feature spec and builder
    spec = FeatureSpec(
        numeric=["request_qty","market_rate_bps","short_interest","inventory_avail","price_dollar"],
        categorical=["cusip_bucket","client_tier","venue"],
        hashed=[("issuer", 32)],
        poly2_for_pi0=["request_qty","market_rate_bps","short_interest","inventory_avail"],
    )

    train_logs, dev_logs = build_logs(n=6000)
    fb = RFQFeatureBuilder(spec).fit([z.rfq for z in train_logs])

    # Catalog + encoders
    catalog = ActionCatalog()
    ape = ActionPayloadEncoder(catalog)
    dm = FhatDesignMatrixBuilder(fb, ape)

    # ---- Train fhat (choose 'mlp' or 'hgb')
    X_tr, A_tr, y_tr = dm.build_supervised_arrays(train_logs)
    X_dv, A_dv, y_dv = dm.build_supervised_arrays(dev_logs)

    fcfg = FhatConfig(kind="mlp", epochs=25, hidden=256, hidden2=128, dropout=0.1, lr=1e-3, batch_size=512)
    fhat = FhatModel(fcfg)
    print("Training fhat…")
    print(fhat.fit(X_tr, A_tr, y_tr, X_val=X_dv, A_val=A_dv, y_val=y_dv))

    # ---- Train pi0 (choose 'logit' or 'hgb')
    pcfg = Pi0Config(kind="logit", calibrate=False, propensity_floor=1e-3)
    pi0 = LoggingPolicyModel(pcfg, fb)
    print("Training pi0…")
    print(pi0.fit(train_logs, dev_logs))

    # ---- Precompute tensors for Stage-1 trainer
    print("Precomputing stage-1 tensors…")
    train_tensors = build_stage1_tensors(train_logs, fb, fhat, pi0, catalog, ape, dm)
    dev_tensors   = build_stage1_tensors(dev_logs,   fb, fhat, pi0, catalog, ape, dm)

    # ---- Train Stage-1 policy
    cfg = FirstStageConfig(
        hidden=64,
        lr=5e-3,
        epochs=20,
        batch_size=512,
        grad_clip=5.0,
        propensity_floor=1e-3,
        w_clip=50.0,
        entropy_reg=0.0,
    )
    print("Training stage-1 policy…")
    model, metrics = train_first_stage(train_tensors, dev_tensors, cfg, input_dim=train_tensors["X"].shape[1])
    print("Best dev OPE:", metrics["dev_OPE"])

    # ---- Save artifacts for serving
    joblib.dump(fb,        "./artifacts/stage1.featurizer.joblib")
    joblib.dump(catalog,   "./artifacts/stage1.catalog.joblib")
    joblib.dump(ape,       "./artifacts/stage1.payload_encoder.joblib")
    fhat.save("./artifacts/stage1.fhat.bin")
    joblib.dump(pcfg.__dict__, "./artifacts/stage1.pi0_cfg.json")
    torch = __import__("torch")
    torch.save(model.state_dict(), "./artifacts/stage1.pi1_shallow.pt")
    print("Artifacts saved in ./artifacts")
