# step2_context_featurizer.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_extraction import FeatureHasher

# --- keep using the minimal Context for quoting math ---
from step0_action_space import Context as QuoteContext

# -------------------------
# Flexible RFQ context
# -------------------------

@dataclass
class RFQContext:
    """
    Rich, schema-less context container for learning.
    Put anything you'd like to use as model input into `raw`.
    """
    cusip: str
    request_qty: float
    market_rate_bps: int
    timestamp: Optional[datetime] = None
    client_id: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def as_quote_context(self) -> QuoteContext:
        """Adapter for step0 realize() math."""
        return QuoteContext(
            cusip=self.cusip,
            market_rate_bps=self.market_rate_bps,
            request_qty=self.request_qty,
        )

    def to_record(self) -> Dict[str, Any]:
        """Flat dict suitable for a DataFrame row."""
        base = {
            "cusip": self.cusip,
            "request_qty": self.request_qty,
            "market_rate_bps": self.market_rate_bps,
            "client_id": self.client_id,
            "ts_unix": None if self.timestamp is None else self.timestamp.timestamp(),
        }
        # raw keys override base only if you intentionally set them
        return {**base, **self.raw}

# -------------------------
# Feature specification
# -------------------------

@dataclass
class FeatureSpec:
    """
    Declare which fields are numeric vs categorical.
    - numeric: will be float32 + StandardScaler
    - categorical: OneHotEncoder(handle_unknown='ignore')
    - hashed: high-card categorical fields mapped via FeatureHasher
    """
    numeric: List[str]
    categorical: List[str]
    hashed: List[Tuple[str, int]] = field(default_factory=list)  # (field, n_features)

    # optional: which numeric cols to expand to poly2 for pi_0
    poly2_for_pi0: Optional[List[str]] = None

# -------------------------
# Feature builder (fit/transform)
# -------------------------

class RFQFeatureBuilder:
    """
    Fits a preprocessing pipeline on DataFrame rows created from RFQContext.to_record().
    Produces:
      - X_model: features for f_hat and pi^1st shallow NN
      - X_poly2: polynomial(deg=2) numeric-only for pi_0 logistic regression
    """
    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self._coltfm: Optional[ColumnTransformer] = None
        self._pipe: Optional[Pipeline] = None
        self._poly2: Optional[Pipeline] = None
        self._out_feature_names: Optional[List[str]] = None
        self._poly2_feature_names: Optional[List[str]] = None

    def _build_column_transformer(self) -> ColumnTransformer:
        transformers = []
        if self.spec.numeric:
            transformers.append(("num", StandardScaler(), self.spec.numeric))
        if self.spec.categorical:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            transformers.append(("cat", ohe, self.spec.categorical))
        for field, n_feats in self.spec.hashed:
            # FeatureHasher expects dict-like or iterable of tokens; we wrap to a single-column dict
            transformers.append(
                (f"hash__{field}",
                 Pipeline(steps=[("to_dict", _StringToDict(field)),
                                 ("hasher", FeatureHasher(n_features=n_feats, input_type="dict"))]),
                 [field])
            )
        # passthrough any columns not listed? safer to drop.
        return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)

    def fit(self, rfqs: Sequence[RFQContext]) -> "RFQFeatureBuilder":
        df = pd.DataFrame([r.to_record() for r in rfqs])

        self._coltfm = self._build_column_transformer()
        self._pipe = Pipeline(steps=[("cols", self._coltfm)])
        self._pipe.fit(df)

        # resolve feature names after fitting
        self._out_feature_names = self._resolve_output_feature_names()

        # poly2 for pi_0 on a numeric subset
        if self.spec.poly2_for_pi0:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            self._poly2 = Pipeline(steps=[
                ("selector", _ColumnSelector(self.spec.poly2_for_pi0)),
                ("scaler", StandardScaler()),
                ("poly2", poly),
            ])
            self._poly2.fit(df)
            # feature names for poly2
            poly_cols = self.spec.poly2_for_pi0
            self._poly2_feature_names = _poly_feature_names(poly_cols)
        return self

    def transform_context(self, rfq: RFQContext) -> np.ndarray:
        self._ensure_fit()
        df = pd.DataFrame([rfq.to_record()])
        X = self._pipe.transform(df).astype(np.float32)
        return X

    def transform_batch(self, rfqs: Sequence[RFQContext]) -> np.ndarray:
        self._ensure_fit()
        df = pd.DataFrame([r.to_record() for r in rfqs])
        X = self._pipe.transform(df).astype(np.float32)
        return X

    def transform_poly2_for_pi0(self, rfq: RFQContext) -> Optional[np.ndarray]:
        if self._poly2 is None:
            return None
        df = pd.DataFrame([rfq.to_record()])
        Xp = self._poly2.transform(df).astype(np.float32)
        return Xp

    def feature_names(self) -> List[str]:
        self._ensure_fit()
        return list(self._out_feature_names)

    def poly2_feature_names(self) -> Optional[List[str]]:
        return None if self._poly2_feature_names is None else list(self._poly2_feature_names)

    # ----- internals -----
    def _ensure_fit(self):
        if self._pipe is None:
            raise RuntimeError("RFQFeatureBuilder is not fitted yet. Call fit([...]) first.")

    def _resolve_output_feature_names(self) -> List[str]:
        names: List[str] = []
        # ColumnTransformer.get_feature_names_out is available in newer sklearn versions.
        try:
            names = list(self._pipe.named_steps["cols"].get_feature_names_out())
        except Exception:
            # Fallback: approximate names
            parts = []
            if self.spec.numeric:
                parts += [f"num__{c}" for c in self.spec.numeric]
            if self.spec.categorical:
                # OHE expands—we won't have exact dummy names here without get_feature_names_out
                parts += [f"cat__{c}_*" for c in self.spec.categorical]
            for field, n_feats in self.spec.hashed:
                parts += [f"hash__{field}_{i}" for i in range(n_feats)]
            names = parts
        return names

# -------------------------
# Small helper transformers
# -------------------------

class _StringToDict:
    """Wrap a single string column into a dict {key: value} for FeatureHasher."""
    def __init__(self, key: str):
        self.key = key
    def fit(self, X, y=None): return self
    def transform(self, X):
        # X is a 2D array-like with one column (the field name)
        # emit list[dict] for FeatureHasher(input_type="dict")
        vals = np.array(X).ravel().tolist()
        return [{self.key: str(v) if v is not None else ""} for v in vals]

class _ColumnSelector:
    """Select a subset of DataFrame columns by name (for pipelines)."""
    def __init__(self, cols: List[str]):
        self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.cols].to_numpy()
        # assume dict-like records
        return np.asarray([[row.get(c, np.nan) for c in self.cols] for row in X], dtype=float)

def _poly_feature_names(cols: List[str]) -> List[str]:
    # names like x1, x2, x1^2, x1*x2 ...
    names = list(cols)
    # quadratic terms
    names += [f"{c}^{2}" for c in cols]
    # cross terms
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            names.append(f"{cols[i]}*{cols[j]}")
    return names

# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    # 1) declare which raw keys you plan to use
    spec = FeatureSpec(
        numeric=[
            "request_qty",
            "market_rate_bps",
            "short_interest",
            "days_to_cover",
            "inventory_avail",
            "price_dollar",
        ],
        categorical=[
            "cusip_bucket",     # you can bucketize CUSIP by liquidity tier
            "client_tier",
            "venue",
        ],
        hashed=[("issuer", 32)],  # high-card issuer mapped via hashing
        poly2_for_pi0=[
            "request_qty",
            "market_rate_bps",
            "short_interest",
            "inventory_avail",
        ],
    )

    # 2) build some contexts
    rfqs = [
        RFQContext(
            cusip="123456AB7",
            request_qty=1_000_000,
            market_rate_bps=120,
            client_id="CLNT123",
            raw=dict(
                short_interest=8.2,
                days_to_cover=1.7,
                inventory_avail=2_500_000,
                price_dollar=37.45,
                cusip_bucket="liquid",
                client_tier="A",
                venue="POV",
                issuer="ACME INC",
            ),
        ),
        RFQContext(
            cusip="999999ZZZ",
            request_qty=300_000,
            market_rate_bps=450,
            client_id="CLNT777",
            raw=dict(
                short_interest=22.0,
                days_to_cover=4.1,
                inventory_avail=250_000,
                price_dollar=12.10,
                cusip_bucket="special",
                client_tier="B",
                venue="RFS",
                issuer="MEGA HOLDINGS",
            ),
        ),
    ]

    # 3) fit the featurizer
    fb = RFQFeatureBuilder(spec).fit(rfqs)

    # 4) transform to model features
    X = fb.transform_batch(rfqs)
    print("X shape:", X.shape)
    print("first few feature names:", fb.feature_names()[:10])

    # 5) poly2 features for pi_0 logistic regression
    Xp = fb.transform_poly2_for_pi0(rfqs[0])
    print("poly2 shape:", None if Xp is None else Xp.shape)
    print("poly2 names:", fb.poly2_feature_names())
