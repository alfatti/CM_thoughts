from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional

# -------- Config (defaults from your note) --------

DEFAULT_RATE_MIN_BPS = -300
DEFAULT_RATE_MAX_BPS =  300
DEFAULT_RATE_STEP_BPS =  10

DEFAULT_FILL_MIN_PCT  =   5   # 5% bucket floor (20 buckets => 5..100)
DEFAULT_FILL_MAX_PCT  = 100
DEFAULT_FILL_STEP_PCT =   5

# -------- Bin builders --------

def make_rate_offsets_bps(
    min_bps: int = DEFAULT_RATE_MIN_BPS,
    max_bps: int = DEFAULT_RATE_MAX_BPS,
    step_bps: int = DEFAULT_RATE_STEP_BPS,
) -> List[int]:
    """Inclusive range in bps, e.g., [-300, -290, ..., +300]."""
    if step_bps <= 0:
        raise ValueError("step_bps must be > 0")
    if min_bps > max_bps:
        raise ValueError("min_bps must be <= max_bps")
    n = (max_bps - min_bps) // step_bps
    if min_bps + n * step_bps != max_bps:
        # Enforce exact divisibility so we know endpoints are included.
        raise ValueError("Range must be divisible by step_bps to include both endpoints.")
    return [min_bps + i * step_bps for i in range(n + 1)]

def make_fill_buckets_pct(
    min_pct: int = DEFAULT_FILL_MIN_PCT,
    max_pct: int = DEFAULT_FILL_MAX_PCT,
    step_pct: int = DEFAULT_FILL_STEP_PCT,
) -> List[int]:
    """Inclusive bucket centers/values in percent, e.g., [5,10,...,100] (20 buckets)."""
    if step_pct <= 0:
        raise ValueError("step_pct must be > 0")
    if min_pct <= 0 or max_pct > 100:
        raise ValueError("fill% must be in 1..100")
    if min_pct > max_pct:
        raise ValueError("min_pct must be <= max_pct")
    n = (max_pct - min_pct) // step_pct
    if min_pct + n * step_pct != max_pct:
        raise ValueError("Range must be divisible by step_pct to include both endpoints.")
    return [min_pct + i * step_pct for i in range(n + 1)]

# -------- Core dataclasses --------

@dataclass(frozen=True)
class Context:
    """Minimal context for pricing/allocating a borrow."""
    cusip: str
    market_rate_bps: int     # e.g., 120 = 1.20%
    request_qty: float       # client requested shares (or units)

@dataclass(frozen=True)
class Action:
    """An action = (rate offset in bps, fill %)."""
    action_id: int
    offset_bps: int
    fill_pct: int            # 5..100 inclusive by default

@dataclass(frozen=True)
class Decision:
    """Concrete quote & fill implied by an Action for a given Context."""
    quote_rate_bps: int      # market_rate_bps + offset_bps
    fill_qty: float          # fill_pct% of request_qty
    action: Action

# -------- Action space with encoders --------

class ActionSpace:
    def __init__(
        self,
        rate_offsets_bps: Optional[List[int]] = None,
        fill_buckets_pct: Optional[List[int]] = None,
    ):
        self.rate_offsets_bps = rate_offsets_bps or make_rate_offsets_bps()
        self.fill_buckets_pct = fill_buckets_pct or make_fill_buckets_pct()

        if not self.rate_offsets_bps:
            raise ValueError("rate_offsets_bps cannot be empty")
        if not self.fill_buckets_pct:
            raise ValueError("fill_buckets_pct cannot be empty")

        # For quick lookup/validation
        self._offset_index: Dict[int, int] = {v: i for i, v in enumerate(self.rate_offsets_bps)}
        self._fill_index:   Dict[int, int] = {v: i for i, v in enumerate(self.fill_buckets_pct)}

        self._n_offsets = len(self.rate_offsets_bps)
        self._n_fills   = len(self.fill_buckets_pct)
        self._size      = self._n_offsets * self._n_fills

    @property
    def size(self) -> int:
        return self._size

    def id_from_tuple(self, offset_bps: int, fill_pct: int) -> int:
        """Row-major encoding: id = offset_idx * n_fills + fill_idx."""
        try:
            oi = self._offset_index[offset_bps]
            fi = self._fill_index[fill_pct]
        except KeyError as e:
            raise ValueError(f"Unknown bin value {e.args[0]}") from None
        return oi * self._n_fills + fi

    def tuple_from_id(self, action_id: int) -> Tuple[int, int]:
        if not (0 <= action_id < self._size):
            raise IndexError(f"action_id out of range [0, {self._size})")
        oi, fi = divmod(action_id, self._n_fills)
        return self.rate_offsets_bps[oi], self.fill_buckets_pct[fi]

    def action_from_id(self, action_id: int) -> Action:
        offset, fill = self.tuple_from_id(action_id)
        return Action(action_id=action_id, offset_bps=offset, fill_pct=fill)

    def action_from_tuple(self, offset_bps: int, fill_pct: int) -> Action:
        aid = self.id_from_tuple(offset_bps, fill_pct)
        return Action(action_id=aid, offset_bps=offset_bps, fill_pct=fill_pct)

    def iter_actions(self) -> Iterable[Action]:
        for oi, offset in enumerate(self.rate_offsets_bps):
            base = oi * self._n_fills
            for fi, fill in enumerate(self.fill_buckets_pct):
                yield Action(action_id=base + fi, offset_bps=offset, fill_pct=fill)

    # ---- Business helpers ----

    @staticmethod
    def apply_offset_bps(market_rate_bps: int, offset_bps: int) -> int:
        """Return quoted rate in bps."""
        return market_rate_bps + offset_bps

    @staticmethod
    def compute_fill_qty(request_qty: float, fill_pct: int) -> float:
        """Return fill quantity given a request size and a fill% bucket."""
        return request_qty * (fill_pct / 100.0)

    def realize(self, ctx: Context, action: Action) -> Decision:
        quote = self.apply_offset_bps(ctx.market_rate_bps, action.offset_bps)
        fillq = self.compute_fill_qty(ctx.request_qty, action.fill_pct)
        return Decision(quote_rate_bps=quote, fill_qty=fillq, action=action)

# -------- Tiny smoke test (can delete) --------

if __name__ == "__main__":
    asp = ActionSpace()
    assert asp.size == 61 * 20  # 1220 actions

    # Example: +20 bps, 25% fill for a request of 10,000 shares at 120 bps market
    ctx = Context(cusip="123456AB7", market_rate_bps=120, request_qty=10_000)
    act = asp.action_from_tuple(offset_bps=20, fill_pct=25)
    dec = asp.realize(ctx, act)
    print(f"ActionID={act.action_id}, Quote={dec.quote_rate_bps} bps, Fill={dec.fill_qty:.0f}")
