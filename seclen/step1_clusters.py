# step1_clusters.py
from enum import IntEnum
from dataclasses import dataclass
from typing import List, Iterable, Tuple, Optional

# reuse step0 types
from step0_action_space import ActionSpace as BaseActionSpace, Action, Context, Decision

class Cluster(IntEnum):
    REJECT  = 0
    FULL    = 1
    PARTIAL = 2

@dataclass(frozen=True)
class RejectAction(Action):
    """Special singleton action for REJECT cluster."""
    # For consistency, we keep offset_bps/fill_pct fields but ignore them.
    # Set them to 0 so they're harmless if someone inspects them.
    pass

class ActionCatalog:
    """
    Extends the base ActionSpace with:
      • a singleton REJECT action (its own cluster)
      • deterministic cluster mapping based on fill% (100 => FULL; <100 => PARTIAL)
    """
    def __init__(self,
                 rate_offsets_bps: Optional[List[int]] = None,
                 fill_buckets_pct: Optional[List[int]] = None):
        self._base = BaseActionSpace(rate_offsets_bps=rate_offsets_bps,
                                     fill_buckets_pct=fill_buckets_pct)

        # Append a singleton REJECT action at the end of the id range.
        self._reject_action_id = self._base.size  # last id
        self._size = self._base.size + 1

        # Precompute cluster partitions for speed
        self._full_ids: List[int] = []
        self._partial_ids: List[int] = []
        for a in self._base.iter_actions():
            if a.fill_pct == 100:
                self._full_ids.append(a.action_id)
            else:
                self._partial_ids.append(a.action_id)

    @property
    def size(self) -> int:
        """Total number of actions including REJECT."""
        return self._size

    @property
    def reject_action_id(self) -> int:
        return self._reject_action_id

    # ----- Action accessors -----

    def action_from_id(self, action_id: int) -> Action:
        if action_id == self._reject_action_id:
            return RejectAction(action_id=action_id, offset_bps=0, fill_pct=0)
        return self._base.action_from_id(action_id)

    def tuple_from_id(self, action_id: int) -> Tuple[int, int]:
        """(offset_bps, fill_pct) for non-REJECT; (0,0) for REJECT."""
        if action_id == self._reject_action_id:
            return (0, 0)
        return self._base.tuple_from_id(action_id)

    def action_from_tuple(self, offset_bps: int, fill_pct: int) -> Action:
        """Create FULL/PARTIAL actions by (offset,fill). Use reject() for REJECT."""
        a = self._base.action_from_tuple(offset_bps, fill_pct)
        return a

    def reject(self) -> RejectAction:
        return RejectAction(action_id=self._reject_action_id, offset_bps=0, fill_pct=0)

    def iter_actions(self) -> Iterable[Action]:
        """Iterate FULL & PARTIAL first, then REJECT last."""
        yield from self._base.iter_actions()
        yield self.reject()

    # ----- Clusters -----

    def cluster_of(self, action_id: int) -> Cluster:
        if action_id == self._reject_action_id:
            return Cluster.REJECT
        # FULL if fill == 100, else PARTIAL
        _, fill = self._base.tuple_from_id(action_id)
        return Cluster.FULL if fill == 100 else Cluster.PARTIAL

    def actions_in_cluster(self, c: Cluster) -> List[int]:
        if c == Cluster.REJECT:
            return [self._reject_action_id]
        elif c == Cluster.FULL:
            return self._full_ids
        else:
            return self._partial_ids

    # ----- Business helpers -----

    def realize(self, ctx: Context, action: Action) -> Decision:
        """Translate an action into a concrete quote/fill for the given RFQ context."""
        if action.action_id == self._reject_action_id:
            # REJECT: no quote, zero fill; set quote to market for logging convenience
            return Decision(quote_rate_bps=ctx.market_rate_bps, fill_qty=0.0, action=action)
        return self._base.realize(ctx, action)

# ---- Smoke test ----
if __name__ == "__main__":
    cat = ActionCatalog()
    print("Total actions (incl REJECT):", cat.size)  # 61*20 + 1 = 1221

    # Check clusters
    rid = cat.reject_action_id
    assert cat.cluster_of(rid) == Cluster.REJECT
    some_full = next(a for a in cat._base.iter_actions() if a.fill_pct == 100).action_id
    some_part = next(a for a in cat._base.iter_actions() if a.fill_pct < 100).action_id
    assert cat.cluster_of(some_full) == Cluster.FULL
    assert cat.cluster_of(some_part) == Cluster.PARTIAL

    # Example usage
    from step0_action_space import Context
    ctx = Context(cusip="123456AB7", market_rate_bps=120, request_qty=10_000)
    dec_r = cat.realize(ctx, cat.reject())
    print("REJECT → quote:", dec_r.quote_rate_bps, "bps; fill:", dec_r.fill_qty)

    a_full = cat.action_from_tuple(offset_bps=20, fill_pct=100)
    a_part = cat.action_from_tuple(offset_bps=-30, fill_pct=25)
    print("FULL id:", a_full.action_id, "cluster:", cat.cluster_of(a_full.action_id).name)
    print("PART id:", a_part.action_id, "cluster:", cat.cluster_of(a_part.action_id).name)
