# step4_logging_policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
import joblib

from step1_clusters import ActionCatalog, Cluster
from step2_context_featurizer import RFQContext, RFQFeatureBuilder
from step3_datasets import LoggedExample

@dataclass
class Pi0Config:
    kind: Literal["logit", "hgb"] = "logit"
    # logistic regression (multinomial) on poly2 numeric features
    logit_C: float = 1.0
    logit_max_iter: int = 200
    logit_penalty: str = "l2"
    logit_class_weight: Optional[str] = "balanced"
    # boosted tree classifier on model features
    hgb_max_depth: Optional[int] = None
    hgb_learning_rate: float = 0.05
    hgb_max_iter: int = 400
    hgb_early_stopping: bool = True
    hgb_validation_fraction: float = 0.1
    hgb_l2_regularization: float = 0.0
    # common
    calibrate: bool = False              # wrap in CalibratedClassifierCV
    calibration_method: str = "isotonic" # or "sigmoid"
    propensity_floor: float = 1e-3
    propensity_clip: float = 0.999

class LoggingPolicyModel:
    """
    Unified interface for estimating pi0(c|x).
    • 'logit':  multinomial LogisticRegression on poly(2) numeric features from RFQFeatureBuilder
    • 'hgb':    HistGradientBoostingClassifier on the full model features (fb.transform_batch)
    """
    def __init__(self, cfg: Pi0Config, fb: RFQFeatureBuilder):
        self.cfg = cfg
        self.fb = fb
        self.kind = cfg.kind
        self._clf = None           # base estimator
        self._calibrated = False   # whether wrapped by CalibratedClassifierCV

    # ----- feature builders depending on kind -----

    def _build_X(self, rfqs: list[RFQContext]) -> np.ndarray:
        if self.kind == "logit":
            # poly2 numeric-only (as set in FeatureSpec.poly2_for_pi0)
            X = [self.fb.transform_poly2_for_pi0(r) for r in rfqs]
            if any(x is None for x in X):
                # fallback to full model features if poly2 wasn't configured
                return self.fb.transform_batch(rfqs)
            return np.vstack(X).astype(np.float32)
        else:
            # boosted tree uses the standard model features
            return self.fb.transform_batch(rfqs)

    # ----- training / prediction -----

    def fit(self, logs_train: list[LoggedExample], logs_dev: list[LoggedExample]) -> Dict[str, float]:
        y_tr = np.array([self._cluster_of(z) for z in logs_train], dtype=int)
        y_dv = np.array([self._cluster_of(z) for z in logs_dev], dtype=int)
        X_tr = self._build_X([z.rfq for z in logs_train])
        X_dv = self._build_X([z.rfq for z in logs_dev])

        if self.kind == "logit":
            base = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                C=self.cfg.logit_C,
                penalty=self.cfg.logit_penalty,
                max_iter=self.cfg.logit_max_iter,
                class_weight=self.cfg.logit_class_weight,
                n_jobs=None,
            )
        else:
            base = HistGradientBoostingClassifier(
                max_depth=self.cfg.hgb_max_depth,
                learning_rate=self.cfg.hgb_learning_rate,
                max_iter=self.cfg.hgb_max_iter,
                early_stopping=self.cfg.hgb_early_stopping,
                validation_fraction=self.cfg.hgb_validation_fraction,
                l2_regularization=self.cfg.hgb_l2_regularization,
            )

        if self.cfg.calibrate:
            clf = CalibratedClassifierCV(base, method=self.cfg.calibration_method, cv=5)
            clf.fit(X_tr, y_tr)
            self._calibrated = True
        else:
            base.fit(X_tr, y_tr)
            clf = base
            self._calibrated = False

        self._clf = clf

        # dev metrics
        p_dv = self.predict_proba_raw(X_dv)    # before flooring/clipping
        ll = float(log_loss(y_dv, p_dv))
        acc = float(accuracy_score(y_dv, np.argmax(p_dv, axis=1)))
        return {"dev_logloss": ll, "dev_acc": acc}

    def predict_proba_raw(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Model not fit/loaded.")
        P = self._clf.predict_proba(X)
        # Ensure class ordering [0,1,2] if estimator reorders — scikit keeps it aligned to classes_
        return np.asarray(P, dtype=np.float32)

    def predict_proba(self, rfqs: list[RFQContext]) -> np.ndarray:
        """Return clipped/floored probabilities over [REJECT,FULL,PARTIAL]."""
        X = self._build_X(rfqs)
        P = self.predict_proba_raw(X)

        eps = self.cfg.propensity_floor
        P = np.clip(P, eps, self.cfg.propensity_clip)
        P = P / P.sum(axis=1, keepdims=True)
        return P

    # ----- utils / persistence -----

    @staticmethod
    def _cluster_of(z: LoggedExample) -> int:
        from step1_clusters import ActionCatalog
        # We only need the mapping; create lightweight catalog if not passed
        # (IDs are deterministic from grids used during logging).
        c = ActionCatalog().cluster_of(z.action_id)
        return int(c)

    def save(self, path: str):
        if self._clf is None:
            raise RuntimeError("Nothing to save; fit first.")
        blob = {
            "cfg": self.cfg.__dict__,
            "kind": self.kind,
            "calibrated": self._calibrated,
            "clf": self._clf,
        }
        joblib.dump(blob, path)

    @staticmethod
    def load(path: str, fb: RFQFeatureBuilder) -> "LoggingPolicyModel":
        blob = joblib.load(path)
        cfg = Pi0Config(**blob["cfg"])
        m = LoggingPolicyModel(cfg, fb)
        m._clf = blob["clf"]
        m._calibrated = bool(blob["calibrated"])
        return m
