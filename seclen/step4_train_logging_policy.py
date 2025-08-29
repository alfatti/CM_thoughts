# step4_train_logging_policy.py
from __future__ import annotations
from typing import List, Tuple
import random
import joblib
import numpy as np

from step1_clusters import ActionCatalog, Cluster
from step2_context_featurizer import RFQContext, RFQFeatureBuilder, FeatureSpec
from step3_datasets import LoggedExample
from step4_logging_policy import LoggingPolicyModel, Pi0Config

def build_synthetic_logs(n=3000) -> Tuple[list[LoggedExample], list[LoggedExample]]:
    # Same synthetic generator as before so you can run end-to-end
    spec = FeatureSpec(
        numeric=["request_qty","market_rate_bps","short_interest","inventory_avail","price_dollar"],
        categorical=["cusip_bucket","client_tier","venue"],
        hashed=[("issuer", 32)],
        poly2_for_pi0=["request_qty","market_rate_bps","short_interest","inventory_avail"],
    )
    fb = RFQFeatureBuilder(spec)

    cat = ActionCatalog()
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

    # simulate rewards and clusters (for logging policy we only need clusters)
    def sample_action_id():
        u = random.random()
        if u < 0.15:
            return cat.reject_action_id
        elif u < 0.50:
            return random.choice(cat.actions_in_cluster(Cluster.FULL))
        else:
            return random.choice(cat.actions_in_cluster(Cluster.PARTIAL))

    logs: list[LoggedExample] = []
    rfqs_tmp: list[RFQContext] = []
    for i in range(n):
        rfq = mk_rfqi(i)
        rfqs_tmp.append(rfq)
        aid = sample_action_id()
        # reward is not used for pi0; keep a dummy 0.0
        logs.append(LoggedExample(rfq=rfq, action_id=aid, reward=0.0))

    # fit featurizer on all rfqs (or train split)
    fb.fit(rfqs_tmp)

    random.shuffle(logs)
    split = int(0.8 * len(logs))
    return logs[:split], logs[split:], fb

def train_and_save_pi0(kind="logit", calibrate=False, save_prefix="./artifacts/pi0"):
    train_logs, dev_logs, fb = build_synthetic_logs(n=4000)

    cfg = Pi0Config(kind=kind, calibrate=calibrate)
    pi0 = LoggingPolicyModel(cfg, fb)
    metrics = pi0.fit(train_logs, dev_logs)
    print(f"[pi0/{kind}] dev:", metrics)

    # Save model + featurizer
    pi0.save(f"{save_prefix}.{kind}.joblib")
    joblib.dump(fb, f"{save_prefix}.featurizer.joblib")
    print("Saved pi0 + featurizer.")

if __name__ == "__main__":
    # Train both variants for a sanity check
    train_and_save_pi0(kind="logit", calibrate=False, save_prefix="./artifacts/pi0_logit")
    train_and_save_pi0(kind="hgb",   calibrate=True,  save_prefix="./artifacts/pi0_hgb")
