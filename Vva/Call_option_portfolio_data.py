"""
DeepXVA (CVA + FVA) on a large Black-Scholes call option portfolio — everything except the BSDE solver.

Assumptions (as discussed):
- Risk factors X_t = (S_1(t),...,S_d(t)) are lognormal diffusions with correlated Brownian motion:
      dS_i(t) = (r(t) - q_i) * S_i(t) dt + sigma_i * S_i(t) dW_i(t)   under Q
- Deterministic OIS discount curve P(0,t) (or equivalently r(t) deterministic).
- Clean call option prices are Black-Scholes closed form using current spot S_i(t),
  discount factor P(t,T_i), and time-to-expiry (T_i - t).
- We solve only XVA = CVA + FVA (no DVA, no ColVA).
- Collateral is a simple threshold CSA: C_t = clip( Vbar_t, [-H, H] ) outside threshold style.

You plug in YOUR BSDE solver by implementing an object with:
  solve(simulator, driver, terminal_condition=0.0) -> dict with XVA0, (optionally) paths, etc.

The key objects provided here:
- ExpiryGrid, DiscountCurve
- CallOption, CallOptionPortfolio
- CollateralModel (threshold)
- CounterpartyModel (deterministic hazard + recovery)
- FundingModel (borrow/lend spreads)
- SpotSimulator (Euler lognormal with correlation under Q)
- Black-Scholes call pricer utilities
- XVA_Driver (the BSDE driver f(t, state, xva, vbar, collat))
- A "data adapter" that produces per-path arrays needed by a standard deepBSDE loop.

This is written to be clear and hackable, not micro-optimized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Dict, Any
import math
import numpy as np


# ----------------------------
# Utilities
# ----------------------------

SQRT_2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

def norm_cdf(x: np.ndarray) -> np.ndarray:
    # vectorized standard normal cdf
    return 0.5 * (1.0 + erf(x / SQRT_2))

def erf(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf as _erf
    return _erf(x)

def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)

def pos(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def neg(x: np.ndarray) -> np.ndarray:
    return np.maximum(-x, 0.0)

def piecewise_constant(t: float, knots: np.ndarray, values: np.ndarray) -> float:
    """
    knots: increasing array of knot times, e.g. [0,1,2,5,10]
    values: value on [knots[j], knots[j+1]) for j=0..len-2; last used for t>=knots[-1]
    """
    idx = np.searchsorted(knots, t, side="right") - 1
    idx = int(np.clip(idx, 0, len(values) - 1))
    return float(values[idx])


# ----------------------------
# Portfolio configuration
# ----------------------------

@dataclass
class PortfolioConfig:
    """
    Central configuration for all hard-coded parameters in example_build_problem
    and solve_with_your_bsde_solver.  Edit here instead of touching the build logic.
    """
    # Simulation / time grid
    d: int = 80                        # number of underlying assets / call options
    T_max: float = 5.0                 # horizon (years)
    dt: float = 1.0 / 52.0            # time-step (weekly)
    n_paths: int = 50_000              # Monte Carlo paths
    seed: int = 123                    # RNG seed

    # Expiry grid: options expire uniformly between T_min_expiry and T_max_expiry
    T_min_expiry: float = 0.25         # shortest expiry (years)
    T_max_expiry: float = 5.0          # longest  expiry (years)

    # Discount curve (flat continuous zero rate)
    r0: float = 0.03

    # Volatility (flat BS vol, one per underlying)
    sigma_const: float = 0.20

    # Dividend yield (flat, one per underlying)
    q_const: float = 0.00

    # Correlation (exponential decay indexed by asset number)
    corr_a: float = 0.2

    # Initial spot prices (linearly interpolated between these two values)
    S0_start: float = 90.0
    S0_end: float = 110.0

    # Call strikes: moneyness offsets around ATM spot (in percent of S0, e.g. -0.10 = 10% OTM)
    strike_moneyness: Tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10)

    # Notional per option (number of shares)
    notional: float = 1_000.0

    # Collateral (threshold CSA)
    threshold_H: float = 1e6

    # Counterparty hazard rate (flat, in decimal, e.g. 0.015 = 150 bp)
    lambda_C: float = 0.015
    recovery: float = 0.40

    # Funding spreads (decimal)
    spread_borrow: float = 0.010       # borrow spread over OIS
    spread_lend: float = 0.000         # lend  spread over OIS


# ----------------------------
# Expiry grid and discounting
# ----------------------------

@dataclass(frozen=True)
class ExpiryGrid:
    """
    Expiry dates T_1 < T_2 < ... < T_d, one per underlying asset.
    Each call option i expires at T_i.
    """
    T: np.ndarray  # shape (d,)

    def __post_init__(self):
        if self.T.ndim != 1 or self.T.size < 1:
            raise ValueError("ExpiryGrid.T must be 1D with length >= 1.")
        if not np.all(np.diff(self.T) > 0):
            raise ValueError("ExpiryGrid.T must be strictly increasing.")

    @property
    def d(self) -> int:
        return self.T.size

    def expiry(self, i: int) -> float:
        """Expiry time for asset i (1-indexed)."""
        return float(self.T[i - 1])


class DiscountCurve:
    """
    Deterministic discount curve P(0,t). Provide either:
      - zero_rate(t): continuous comp zero rate, or
      - discount(t): direct P(0,t) function.

    We also provide P(t,T) = P(0,T)/P(0,t) (deterministic).
    """
    def __init__(self, discount_0t: Callable[[float], float]):
        self._P0 = discount_0t

    def P0(self, t: float) -> float:
        return float(self._P0(t))

    def PtT(self, t: float, T: float) -> float:
        if T < t:
            raise ValueError("Need T >= t for PtT.")
        return self.P0(T) / self.P0(t)

    def r_inst(self, t: float, eps: float = 1e-5) -> float:
        # numerical instantaneous rate from log discount derivative
        t2 = t + eps
        P1 = self.P0(t)
        P2 = self.P0(t2)
        if P1 <= 0 or P2 <= 0:
            raise ValueError("Discount factors must be positive.")
        return -math.log(P2 / P1) / eps


# ----------------------------
# Call option portfolio
# ----------------------------

@dataclass(frozen=True)
class CallOption:
    """
    A single European call on asset i: payoff at T_i is N * (S_i(T_i) - K)^+.
    """
    i: int          # asset index (1-indexed), also determines expiry via ExpiryGrid
    K: float        # strike
    notional: float # number of shares
    is_long: bool = True  # if False, position is short


class CallOptionPortfolio:
    def __init__(self, expiry_grid: ExpiryGrid, options: List[CallOption]):
        self.expiry_grid = expiry_grid
        self.options = options
        # basic validation
        for o in options:
            if o.i < 1 or o.i > expiry_grid.d:
                raise ValueError(f"Option index i must be in 1..d where d={expiry_grid.d}. Got {o.i}.")
            if o.K <= 0 or o.notional <= 0:
                raise ValueError("Strike and notional must be positive.")

    @property
    def M(self) -> int:
        return len(self.options)


# ----------------------------
# Models: collateral, counterparty hazard, funding
# ----------------------------

class CollateralModel:
    """
    Simple threshold CSA:
        If Vbar > H:   C = Vbar - H
        If Vbar < -H:  C = Vbar + H
        Else:          C = 0
    This collateral is signed (received vs posted).
    """
    def __init__(self, threshold_H: float):
        if threshold_H < 0:
            raise ValueError("threshold_H must be nonnegative.")
        self.H = float(threshold_H)

    def C(self, Vbar: np.ndarray) -> np.ndarray:
        H = self.H
        return np.where(Vbar > H, Vbar - H, np.where(Vbar < -H, Vbar + H, 0.0))


class CounterpartyModel:
    """
    Deterministic hazard lambda_C^Q(t) (piecewise constant), recovery R_C.
    """
    def __init__(self, knots: np.ndarray, lambdas: np.ndarray, recovery: float):
        if not (0.0 <= recovery <= 1.0):
            raise ValueError("recovery must be in [0,1].")
        self.knots = np.asarray(knots, dtype=float)
        self.lambdas = np.asarray(lambdas, dtype=float)
        self.R = float(recovery)

    def lambda_C(self, t: float) -> float:
        return max(0.0, piecewise_constant(t, self.knots, self.lambdas))


class FundingModel:
    """
    Funding rates: r^f,b = r + s_b ; r^f,l = r + s_l (spreads may be time-dependent).
    Provide spread functions (or constants).
    """
    def __init__(self, spread_borrow: Callable[[float], float], spread_lend: Callable[[float], float]):
        self.sb = spread_borrow
        self.sl = spread_lend

    def r_fb(self, t: float, r: float) -> float:
        return r + float(self.sb(t))

    def r_fl(self, t: float, r: float) -> float:
        return r + float(self.sl(t))


# ----------------------------
# Volatility and correlation
# ----------------------------

class VolModel:
    """
    sigma_i(t) model for each asset. You can pass:
      - sigmas: shape (d,) constant per asset
      - or a function sigma(i, t)
    Also provides integrated variance to expiry:
        Int_t^{T_i} sigma_i(u)^2 du  (for Black-Scholes pricing)
    """
    def __init__(
        self,
        d: int,
        sigma_i_t: Optional[Callable[[int, float], float]] = None,
        sigmas_const: Optional[np.ndarray] = None,
        int_var_override: Optional[Callable[[int, float, float], float]] = None,
    ):
        self.d = int(d)
        self.sigma_i_t = sigma_i_t
        self.sigmas_const = None if sigmas_const is None else np.asarray(sigmas_const, dtype=float)
        if self.sigmas_const is not None and self.sigmas_const.shape != (self.d,):
            raise ValueError("sigmas_const must have shape (d,).")
        self.int_var_override = int_var_override

        if self.sigma_i_t is None and self.sigmas_const is None:
            raise ValueError("Provide either sigma_i_t or sigmas_const.")

    def sigma(self, i: int, t: float) -> float:
        # i in 1..d
        if self.sigma_i_t is not None:
            return float(self.sigma_i_t(i, t))
        return float(self.sigmas_const[i - 1])

    def int_var(self, i: int, t0: float, t1: float, n_steps: int = 32) -> float:
        if t1 <= t0:
            return 0.0
        if self.int_var_override is not None:
            return float(self.int_var_override(i, t0, t1))
        # crude midpoint quadrature
        dt = (t1 - t0) / n_steps
        acc = 0.0
        for k in range(n_steps):
            tm = t0 + (k + 0.5) * dt
            s = self.sigma(i, tm)
            acc += (s * s) * dt
        return float(acc)


def make_exp_corr(d: int, asset_indices: np.ndarray, a: float = 0.2) -> np.ndarray:
    """
    Exponential correlation indexed by asset number:
        rho_ij = exp(-a * |i - j|)
    Returns (d,d) SPD-ish matrix (should be SPD for a>0).
    """
    idx = asset_indices.astype(float)
    diff = np.abs(idx[:, None] - idx[None, :])
    rho = np.exp(-a * diff)
    # small nugget for numerical stability
    rho += 1e-10 * np.eye(d)
    return rho


# ----------------------------
# Spot price simulation
# ----------------------------

class SpotSimulator:
    """
    Simulate correlated lognormal spot prices under Q:
        S_{n+1} = S_n * exp((r - q_i - 0.5*sigma_i^2)*dt + sigma_i*dW_i)
    (Exact lognormal step per asset; sigma_i evaluated at t_n piecewise.)
    """
    def __init__(
        self,
        expiry_grid: ExpiryGrid,
        vol: VolModel,
        disc: DiscountCurve,
        div_yields: np.ndarray,     # shape (d,), continuous dividend yields q_i
        corr: np.ndarray,
        time_grid: np.ndarray,
        seed: Optional[int] = None,
    ):
        self.expiry_grid = expiry_grid
        self.vol = vol
        self.disc = disc
        self.d = expiry_grid.d
        self.q = np.asarray(div_yields, dtype=float)
        if self.q.shape != (self.d,):
            raise ValueError(f"div_yields must have shape (d,) = ({self.d},).")
        self.time_grid = np.asarray(time_grid, dtype=float)
        if self.time_grid.ndim != 1 or self.time_grid.size < 2:
            raise ValueError("time_grid must be 1D with at least 2 points.")
        if not np.all(np.diff(self.time_grid) > 0):
            raise ValueError("time_grid must be strictly increasing.")

        corr = np.asarray(corr, dtype=float)
        if corr.shape != (self.d, self.d):
            raise ValueError(f"corr must have shape (d,d)=({self.d},{self.d}).")
        # Cholesky for correlation
        self.L = np.linalg.cholesky(corr)
        self.rng = np.random.default_rng(seed)

    def simulate(self, n_paths: int, S0: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns dict with:
          - "t": time grid (N,)
          - "X": spot paths, shape (n_paths, N, d)
          - "dW": Brownian increments, shape (n_paths, N-1, d)
        """
        n_paths = int(n_paths)
        S0 = np.asarray(S0, dtype=float)
        if S0.shape != (self.d,):
            raise ValueError(f"S0 must have shape (d,) = ({self.d},).")

        t = self.time_grid
        N = t.size

        X = np.empty((n_paths, N, self.d), dtype=float)
        X[:, 0, :] = S0[None, :]
        dW = np.empty((n_paths, N - 1, self.d), dtype=float)

        for n in range(N - 1):
            dt = float(t[n + 1] - t[n])
            # iid normals -> correlate
            Z = self.rng.standard_normal(size=(n_paths, self.d))
            dWn = (Z @ self.L.T) * math.sqrt(dt)
            dW[:, n, :] = dWn

            # exact lognormal step: drift = (r - q_i - 0.5*sigma_i^2)*dt
            r_t = self.disc.r_inst(float(t[n]))
            sig = np.array([self.vol.sigma(i + 1, float(t[n])) for i in range(self.d)], dtype=float)  # (d,)
            drift = ((r_t - self.q - 0.5 * sig ** 2) * dt)[None, :]  # (1,d)
            diff = (sig[None, :] * dWn)  # (n_paths,d)
            X[:, n + 1, :] = X[:, n, :] * np.exp(drift + diff)

        return {"t": t, "X": X, "dW": dW}


# ----------------------------
# Black-Scholes call pricer (pathwise clean value)
# ----------------------------

def bs_call_price(
    S: np.ndarray,      # shape (...,)  current spot
    K: float,           # strike
    r: float,           # instantaneous risk-free rate (used as flat rate over tau)
    q: float,           # continuous dividend yield
    tau: float,         # time to expiry T - t
    sigma: float,       # implied vol (flat)
) -> np.ndarray:
    """
    Black-Scholes call value at time t, per 1 share:
      V = S * exp(-q*tau) * Phi(d1) - K * exp(-r*tau) * Phi(d2)
    Returns "per 1 notional" value; caller multiplies by notional and sign.
    """
    S = np.asarray(S, dtype=float)
    if tau <= 0.0:
        # expired: intrinsic only
        return np.maximum(S - K, 0.0)

    eps = 1e-14
    S_safe = np.maximum(S, eps)
    K_safe = max(K, eps)
    vol_sqrt_tau = sigma * math.sqrt(tau)
    d1 = (np.log(S_safe / K_safe) + (r - q + 0.5 * sigma * sigma) * tau) / vol_sqrt_tau
    d2 = d1 - vol_sqrt_tau
    Phi_d1 = norm_cdf(d1)
    Phi_d2 = norm_cdf(d2)
    return S * math.exp(-q * tau) * Phi_d1 - K_safe * math.exp(-r * tau) * Phi_d2


def portfolio_clean_value(
    t: float,
    X_t: np.ndarray,    # shape (n_paths, d) or (d,)
    portfolio: CallOptionPortfolio,
    disc: DiscountCurve,
    vol: VolModel,
    div_yields: np.ndarray,  # shape (d,)
) -> np.ndarray:
    """
    Compute portfolio clean value Vbar_t along paths at time t.
    X_t: spot vector(s) at time t.
    Returns array shape (n_paths,) (or scalar if input is (d,))
    """
    X_t = np.asarray(X_t, dtype=float)
    if X_t.ndim == 1:
        X_t = X_t[None, :]
    n_paths, d = X_t.shape
    if d != portfolio.expiry_grid.d:
        raise ValueError("X_t dimension mismatch with expiry_grid.d")

    r_t = disc.r_inst(t)
    V = np.zeros((n_paths,), dtype=float)

    for opt in portfolio.options:
        i = opt.i
        T_i = portfolio.expiry_grid.expiry(i)
        tau = T_i - t

        if tau < 0.0:
            # Option has already expired — contributes zero to portfolio value.
            continue

        sigma_i = vol.sigma(i, t)
        q_i = float(div_yields[i - 1])
        S_i = X_t[:, i - 1]

        v_per_notional = bs_call_price(S_i, opt.K, r_t, q_i, tau, sigma_i)
        sign = 1.0 if opt.is_long else -1.0
        V += sign * opt.notional * v_per_notional

    return V


# ----------------------------
# XVA driver (CVA + FVA only)
# ----------------------------

class XvaDriver:
    """
    Driver for reduced F-BSDE:
      -dXVA_t = f(t, X_t, XVA_t) dt - Z_t dW_t
    with XVA_T = 0.

    We implement:
      f = CVA_term + FVA_term - tilde_r(t) * XVA
    where
      CVA_term = -(1-R_C) * (Vbar - C)^- * lambda_C
      FVA_term = (r_fl - r) * (Vbar - XVA - C)^+ - (r_fb - r) * (Vbar - XVA - C)^-
      tilde_r  = r + lambda_C    (since no bank default)
    """
    def __init__(
        self,
        disc: DiscountCurve,
        cpty: CounterpartyModel,
        funding: FundingModel,
        collateral: CollateralModel,
    ):
        self.disc = disc
        self.cpty = cpty
        self.funding = funding
        self.collateral = collateral

    def r(self, t: float) -> float:
        return self.disc.r_inst(t)

    def lambda_C(self, t: float) -> float:
        return self.cpty.lambda_C(t)

    def tilde_r(self, t: float) -> float:
        return self.r(t) + self.lambda_C(t)

    def f(
        self,
        t: float,
        Vbar: np.ndarray,      # shape (n_paths,)
        XVA: np.ndarray,       # shape (n_paths,)
        C: np.ndarray,         # shape (n_paths,)
    ) -> np.ndarray:
        r = self.r(t)
        lam = self.lambda_C(t)
        R = self.cpty.R

        # CVA integrand
        exposure = Vbar - C
        cva_term = -(1.0 - R) * neg(exposure) * lam

        # funding requirement (recursive)
        F = Vbar - XVA - C
        rfl = self.funding.r_fl(t, r)
        rfb = self.funding.r_fb(t, r)
        fva_term = (rfl - r) * pos(F) - (rfb - r) * neg(F)

        # discount/killing on XVA
        return cva_term + fva_term - (r + lam) * XVA


# ----------------------------
# Adapter to produce BSDE training tensors
# ----------------------------

@dataclass
class XvaProblemData:
    t: np.ndarray        # (N,)
    X: np.ndarray        # (n_paths, N, d)
    dW: np.ndarray       # (n_paths, N-1, d)
    Vbar: np.ndarray     # (n_paths, N)
    C: np.ndarray        # (n_paths, N)
    # you may also want r(t), lambda(t) arrays for speed
    r: np.ndarray        # (N,)
    lam: np.ndarray      # (N,)
    tilde_r: np.ndarray  # (N,)


def build_xva_training_data(
    sim: Dict[str, np.ndarray],
    portfolio: CallOptionPortfolio,
    disc: DiscountCurve,
    vol: VolModel,
    div_yields: np.ndarray,
    collateral: CollateralModel,
    cpty: CounterpartyModel,
) -> XvaProblemData:
    """
    Given simulated paths (t, X, dW), compute:
      - Vbar(t_n) pathwise from Black-Scholes call formulas
      - C(t_n) from collateral rule on Vbar
      - deterministic r(t_n), lambda(t_n), tilde_r(t_n)

    This is the main "everything else" you need before calling your BSDE solver.
    """
    t = sim["t"]
    X = sim["X"]
    dW = sim["dW"]

    n_paths, N, d = X.shape
    Vbar = np.empty((n_paths, N), dtype=float)
    C = np.empty((n_paths, N), dtype=float)

    for n in range(N):
        tn = float(t[n])
        Vbar[:, n] = portfolio_clean_value(tn, X[:, n, :], portfolio, disc, vol, div_yields)
        C[:, n] = collateral.C(Vbar[:, n])

    r = np.array([disc.r_inst(float(tn)) for tn in t], dtype=float)
    lam = np.array([cpty.lambda_C(float(tn)) for tn in t], dtype=float)
    tilde_r = r + lam

    return XvaProblemData(t=t, X=X, dW=dW, Vbar=Vbar, C=C, r=r, lam=lam, tilde_r=tilde_r)


# ----------------------------
# Hooks for your BSDE solver
# ----------------------------

class XvaProblem:
    """
    High-level object your BSDE solver can consume.

    Many deepBSDE implementations expect:
      - time grid t_n
      - state paths X_{t_n}
      - Brownian increments dW_n
      - a driver function that maps (n, t_n, X_n, Y_n) -> f_n
      - terminal condition Y_T = 0

    Here Y is XVA. We also supply Vbar and C arrays so driver can be evaluated cheaply.
    """
    def __init__(self, data: XvaProblemData, driver: XvaDriver):
        self.data = data
        self.driver = driver

    def driver_f(self, n: int, Y: np.ndarray) -> np.ndarray:
        """
        Return f at time index n for all paths, given current Y=XVA at t_n.
        Shapes:
          Y: (n_paths,)
          returns f: (n_paths,)
        """
        tn = float(self.data.t[n])
        Vbar_n = self.data.Vbar[:, n]
        C_n = self.data.C[:, n]
        return self.driver.f(tn, Vbar_n, Y, C_n)

    @property
    def terminal(self) -> float:
        return 0.0


# ----------------------------
# Example: build everything and call your solver
# ----------------------------

def example_build_problem(
    cfg: PortfolioConfig = PortfolioConfig(),
) -> Tuple[XvaProblem, Dict[str, Any]]:
    """
    Returns:
      - XvaProblem (ready for a BSDE solver)
      - metadata dict with portfolio & model summary

    All tunable parameters are drawn from *cfg* (a PortfolioConfig instance).
    NOTE: n_paths=50k is heavy; adjust for your GPU/CPU.
    """
    # Expiry grid: d assets with uniformly spaced expiries in [T_min_expiry, T_max_expiry]
    T_expiries = np.linspace(cfg.T_min_expiry, cfg.T_max_expiry, cfg.d)
    expiry_grid = ExpiryGrid(T=T_expiries)

    # Deterministic discount curve: flat r=r0
    disc = DiscountCurve(discount_0t=lambda t: math.exp(-cfg.r0 * t))

    # Vols: flat sigma_const for all assets
    sigmas = cfg.sigma_const * np.ones((expiry_grid.d,), dtype=float)
    vol = VolModel(d=expiry_grid.d, sigmas_const=sigmas)

    # Dividend yields: flat q_const for all assets
    div_yields = cfg.q_const * np.ones((expiry_grid.d,), dtype=float)

    # Correlation: exponential decay by asset index
    asset_indices = np.arange(1, expiry_grid.d + 1, dtype=float)
    corr = make_exp_corr(expiry_grid.d, asset_indices, a=cfg.corr_a)

    # Time grid for BSDE/simulation (0..T_max)
    time_grid = np.arange(0.0, cfg.T_max + 1e-12, cfg.dt)
    time_grid = np.unique(np.clip(time_grid, 0.0, cfg.T_max))

    # Initial spot prices: linearly spaced between S0_start and S0_end
    S0 = np.linspace(cfg.S0_start, cfg.S0_end, expiry_grid.d)

    sim = SpotSimulator(
        expiry_grid=expiry_grid, vol=vol, disc=disc, div_yields=div_yields,
        corr=corr, time_grid=time_grid, seed=cfg.seed,
    ).simulate(n_paths=cfg.n_paths, S0=S0)

    # Portfolio: multiple strikes per asset, random long/short for netting richness
    rng = np.random.default_rng(cfg.seed + 1)
    options: List[CallOption] = []
    for i in range(1, expiry_grid.d + 1):
        S0_i = float(S0[i - 1])
        for m in cfg.strike_moneyness:
            K = max(1e-4, S0_i * (1.0 + float(m)))
            is_long = bool(rng.integers(0, 2))
            options.append(CallOption(i=i, K=K, notional=cfg.notional, is_long=is_long))
    portfolio = CallOptionPortfolio(expiry_grid=expiry_grid, options=options)

    # Collateral (threshold CSA)
    collateral = CollateralModel(threshold_H=cfg.threshold_H)

    # Counterparty hazard: flat lambda_C
    knots = np.array([0.0, cfg.T_max])
    lambdas = np.array([cfg.lambda_C])  # piecewise constant
    cpty = CounterpartyModel(knots=knots, lambdas=lambdas, recovery=cfg.recovery)

    # Funding spreads
    funding = FundingModel(
        spread_borrow=lambda t: cfg.spread_borrow,
        spread_lend=lambda t: cfg.spread_lend,
    )

    # Build driver and training data
    driver = XvaDriver(disc=disc, cpty=cpty, funding=funding, collateral=collateral)
    data = build_xva_training_data(sim, portfolio, disc, vol, div_yields, collateral, cpty)
    problem = XvaProblem(data=data, driver=driver)

    meta = {
        "d": expiry_grid.d,
        "M": portfolio.M,
        "T_max": cfg.T_max,
        "dt": cfg.dt,
        "n_paths": cfg.n_paths,
        "r0": cfg.r0,
        "sigma_const": cfg.sigma_const,
        "q_const": cfg.q_const,
        "lambda_C": float(lambdas[0]),
        "recovery_C": cpty.R,
        "funding_spread_b": cfg.spread_borrow,
        "funding_spread_l": cfg.spread_lend,
        "threshold_H": cfg.threshold_H,
    }
    return problem, meta


# ----------------------------
# Example usage with your solver
# ----------------------------

def solve_with_your_bsde_solver(bsde_solver: Any) -> Dict[str, Any]:
    """
    bsde_solver is YOUR object. Expected minimal interface:
        result = bsde_solver.solve(
            t=problem.data.t,
            X=problem.data.X,
            dW=problem.data.dW,
            driver_fn=lambda n, Y: problem.driver_f(n, Y),
            terminal_value=0.0,
        )
    """
    cfg = PortfolioConfig(n_paths=20_000)   # start smaller; override any field here
    problem, meta = example_build_problem(cfg)

    result = bsde_solver.solve(
        t=problem.data.t,
        X=problem.data.X,
        dW=problem.data.dW,
        driver_fn=lambda n, Y: problem.driver_f(n, Y),
        terminal_value=problem.terminal,
        # optionally pass Vbar/C for debugging/aux losses
        aux={"Vbar": problem.data.Vbar, "C": problem.data.C, "tilde_r": problem.data.tilde_r},
    )

    return {"meta": meta, "result": result}


if __name__ == "__main__":
    print("This module builds the XVA problem. Plug in your BSDE solver and call solve_with_your_bsde_solver().")
