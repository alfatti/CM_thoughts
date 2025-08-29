# step3_action_payload.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from step1_clusters import ActionCatalog, Cluster

class ActionPayloadEncoder:
    """
    Encodes an action_id to a fixed-length one-hot vector:
      [one-hot(rate_offset_bps) || one-hot(fill_pct)]
    Reject is not encoded (2nd stage is not used for REJECT).
    """
    def __init__(self, catalog: ActionCatalog):
        self.catalog = catalog

        # Discover unique bins by scanning non-REJECT actions
        offsets = set()
        fills   = set()
        for a in catalog.iter_actions():
            if catalog.cluster_of(a.action_id) == Cluster.REJECT:
                continue
            offsets.add(a.offset_bps)
            fills.add(a.fill_pct)
        self.rate_offsets_bps: List[int] = sorted(offsets)
        self.fill_buckets_pct: List[int] = sorted(fills)

        self._offset2idx: Dict[int, int] = {v: i for i, v in enumerate(self.rate_offsets_bps)}
        self._fill2idx:   Dict[int, int] = {v: i for i, v in enumerate(self.fill_buckets_pct)}

        self.d_offset = len(self.rate_offsets_bps)
        self.d_fill   = len(self.fill_buckets_pct)
        self.dim      = self.d_offset + self.d_fill

    def encode(self, action_id: int) -> np.ndarray:
        """Return one-hot payload for FULL/PARTIAL actions. Raises on REJECT."""
        c = self.catalog.cluster_of(action_id)
        if c == Cluster.REJECT:
            raise ValueError("REJECT has no payload for second-stage.")
        offset_bps, fill_pct = self.catalog.tuple_from_id(action_id)

        v = np.zeros(self.dim, dtype=np.float32)
        v[self._offset2idx[offset_bps]] = 1.0
        v[self.d_offset + self._fill2idx[fill_pct]] = 1.0
        return v

    def batch_encode(self, action_ids: List[int]) -> np.ndarray:
        return np.stack([self.encode(aid) for aid in action_ids], axis=0)
