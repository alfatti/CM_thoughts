# step3_datasets.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict
import numpy as np

from step1_clusters import ActionCatalog, Cluster
from step2_context_featurizer import RFQContext, RFQFeatureBuilder

@dataclass
class LoggedExample:
    rfq: RFQContext
    action_id: int
    reward: float

class ActionPayloadEncoder:
    """
    One-hot encode (offset, fill) + a REJECT flag.
    • Non-REJECT: [onehot(offset)] + [onehot(fill)] + [0]
    • REJECT:     all zeros for offset/fill + [1]
    """
    def __init__(self, catalog: ActionCatalog):
        self.catalog = catalog

        # discover unique offsets/fills from non-REJECT actions
        offsets: set[int] = set()
        fills: set[int] = set()
        for a in catalog._base.iter_actions():  # OK to touch _base for discovery
            offsets.add(a.offset_bps)
            fills.add(a.fill_pct)
        self.offsets = sorted(offsets)
        self.fills = sorted(fills)

        self.offset_index: Dict[int,int] = {v:i for i,v in enumerate(self.offsets)}
        self.fill_index: Dict[int,int]   = {v:i for i,v in enumerate(self.fills)}

        self.n_offset = len(self.offsets)
        self.n_fill   = len(self.fills)
        self.dim = self.n_offset + self.n_fill + 1  # +1 for reject flag

    def encode_one(self, action_id: int) -> np.ndarray:
        x = np.zeros(self.dim, dtype=np.float32)
        if action_id == self.catalog.reject_action_id:
            x[-1] = 1.0
            return x
        off_bps, fill = self.catalog.tuple_from_id(action_id)
        x[self.offset_index[off_bps]] = 1.0
        x[self.n_offset + self.fill_index[fill]] = 1.0
        # reject flag stays 0
        return x

    def encode_many(self, action_ids: Sequence[int]) -> np.ndarray:
        return np.vstack([self.encode_one(aid) for aid in action_ids]).astype(np.float32)

    def payload_dim(self) -> int:
        return self.dim

class FhatDesignMatrixBuilder:
    """
    Creates (X_model, A_payload, y) matrices from LoggedExample[]
    using a fitted RFQFeatureBuilder and ActionPayloadEncoder.
    """
    def __init__(self, fb: RFQFeatureBuilder, ape: ActionPayloadEncoder):
        self.fb = fb
        self.ape = ape

    def build_supervised_arrays(self, logs: Sequence[LoggedExample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rfqs = [z.rfq for z in logs]
        X = self.fb.transform_batch(rfqs)                # [N, d_x]
        A = self.ape.encode_many([z.action_id for z in logs])  # [N, d_a]
        y = np.array([z.reward for z in logs], dtype=np.float32).reshape(-1, 1)  # [N, 1]
        return X, A, y

    # Utilities for cluster argmax/values at inference
    def tile_and_stack(self, X_row: np.ndarray, action_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given one RFQ's feature row [d_x], and a set of action_ids,
        return tiled X [K, d_x] and payloads [K, d_a] for scoring.
        """
        K = len(action_ids)
        X_tiled = np.repeat(X_row.reshape(1, -1), K, axis=0)
        A = self.ape.encode_many(action_ids)
        return X_tiled, A
