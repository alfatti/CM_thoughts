"""
DeepXVA (CVA + FVA) on a large caplet portfolio — everything except the BSDE solver.

Assumptions (as discussed):
- Risk factors X_t = (F_1(t),...,F_d(t)) are lognormal diffusions with correlated Brownian motion:
      dF_i(t) = sigma_i(t) * F_i(t) dW_i(t)   under Q (stylized Black-forward dynamics)
- Deterministic OIS discount curve P(0,t) (or equivalently r(t) deterministic).
- Clean caplet prices are Black closed form using current forward F_i(t), discount factor P(t,T_{i+1}),
  and integrated vol to reset date T_i.
- We solve only XVA = CVA + FVA (no DVA, no ColVA).
- Collateral is a simple threshold CSA: C_t = clip( Vbar_t, [-H, H] ) outside threshold style.

You plug in YOUR BSDE solver by implementing an object with:
  solve(simulator, driver, terminal_condition=0.0) -> dict with XVA0, (optionally) paths, etc.

The key objects provided here:
- TenorGrid, DiscountCurve
- Caplet, CapletPortfolio
- CollateralModel (threshold)
- CounterpartyModel (deterministic hazard + recovery)
- FundingModel (borrow/lend spreads)
- ForwardRateSimulator (Euler lognormal with correlation)
- Black caplet pricer utilities
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
    # numpy has erf
    return np.erf(x)

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
# Tenor and discounting
# ----------------------------

@dataclass(frozen=True)
class TenorGrid:
    """Tenor dates T_0 < T_1 < ... < T_{d+1}. We use caplets on [T_i, T_{i+1}]."""
    T: np.ndarray  # shape (d+2,)
    def __post_init__(self):
        if self.T.ndim != 1 or self.T.size < 3:
            raise ValueError("TenorGrid.T must be 1D with length >= 3 (d+2).")
        if not np.all(np.diff(self.T) > 0):
            raise ValueError("TenorGrid.T must be strictly increasing.")

    @property
    def d(self) -> int:
        return self.T.size - 2

    def accrual(self, i: int) -> float:
        return float(self.T[i + 1] - self.T[i])

    def reset_time(self, i: int) -> float:
        return float(self.T[i])

    def pay_time(self, i: int) -> float:
        return float(self.T[i + 1])


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
# Caplet portfolio
# ----------------------------

@dataclass(frozen=True)
class Caplet:
    """
    A single caplet on forward index i: payoff at T_{i+1} is N*delta*(F_i(T_i)-K)^+.
    """
    i: int
    K: float
    notional: float
    is_long: bool = True  # if False, position is short


class CapletPortfolio:
    def __init__(self, tenor: TenorGrid, caplets: List[Caplet]):
        self.tenor = tenor
        self.caplets = caplets
        # basic validation
        for c in caplets:
            if c.i < 1 or c.i > tenor.d:
                raise ValueError(f"Caplet index i must be in 1..d where d={tenor.d}. Got {c.i}.")
            if c.K <= 0 or c.notional <= 0:
                raise ValueError("Strike and notional must be positive.")

    @property
    def M(self) -> int:
        return len(self.caplets)


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
    sigma_i(t) model. You can pass:
      - sigmas: shape (d,) constant per forward
      - or a function sigma(i, t)
    Also provides integrated variance to reset:
        Int_t^{T_i} sigma_i(u)^2 du  (for Black pricing)
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


def make_exp_corr(d: int, tenor_times: np.ndarray, a: float = 0.2) -> np.ndarray:
    """
    Exponential correlation in tenor time:
        rho_ij = exp(-a |T_i - T_j|)
    Returns (d,d) SPD-ish matrix (should be SPD for a>0).
    """
    T = tenor_times[1 : d + 1]  # T_1..T_d reset times
    diff = np.abs(T[:, None] - T[None, :])
    rho = np.exp(-a * diff)
    # small nugget for numerical stability
    rho += 1e-10 * np.eye(d)
    return rho


# ----------------------------
# Forward rate simulation
# ----------------------------

class ForwardRateSimulator:
    """
    Simulate correlated lognormal forwards:
        F_{n+1} = F_n * exp(-0.5 sigma^2 dt + sigma * dW)
    (Exact for constant sigma over dt; we use sigma_i(t_n) piecewise for each step.)
    """
    def __init__(
        self,
        tenor: TenorGrid,
        vol: VolModel,
        corr: np.ndarray,
        time_grid: np.ndarray,
        seed: Optional[int] = None,
    ):
        self.tenor = tenor
        self.vol = vol
        self.d = tenor.d
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

    def simulate(self, n_paths: int, F0: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns dict with:
          - "t": time grid (N,)
          - "X": forwards paths, shape (n_paths, N, d)
          - "dW": Brownian increments, shape (n_paths, N-1, d)
        """
        n_paths = int(n_paths)
        F0 = np.asarray(F0, dtype=float)
        if F0.shape != (self.d,):
            raise ValueError(f"F0 must have shape (d,) = ({self.d},).")

        t = self.time_grid
        N = t.size

        X = np.empty((n_paths, N, self.d), dtype=float)
        X[:, 0, :] = F0[None, :]
        dW = np.empty((n_paths, N - 1, self.d), dtype=float)

        for n in range(N - 1):
            dt = float(t[n + 1] - t[n])
            # iid normals -> correlate
            Z = self.rng.standard_normal(size=(n_paths, self.d))
            dWn = (Z @ self.L.T) * math.sqrt(dt)
            dW[:, n, :] = dWn

            # exact-ish lognormal step using sigma at t[n]
            sig = np.array([self.vol.sigma(i + 1, float(t[n])) for i in range(self.d)], dtype=float)  # (d,)
            drift = (-0.5 * (sig ** 2) * dt)[None, :]  # (1,d)
            diff = (sig[None, :] * dWn)  # (n_paths,d)
            X[:, n + 1, :] = X[:, n, :] * np.exp(drift + diff)

        return {"t": t, "X": X, "dW": dW}


# ----------------------------
# Black caplet pricer (pathwise clean value)
# ----------------------------

def black_caplet_price(
    F: np.ndarray,          # shape (...,)
    K: float,
    P_t_Tpay: float,
    delta: float,
    int_var: float,
) -> np.ndarray:
    """
    Black caplet value at time t for payoff at Tpay, reset at Treset:
      V = N*delta*P(t,Tpay)*(F*Phi(d1)-K*Phi(d2))
    Here we return "per 1 notional" value; caller multiplies by notional and sign.
    """
    F = np.asarray(F, dtype=float)
    if int_var <= 0.0:
        # zero vol limit
        return P_t_Tpay * delta * np.maximum(F - K, 0.0)

    vol = math.sqrt(int_var)
    # guard
    eps = 1e-14
    F_safe = np.maximum(F, eps)
    K_safe = max(K, eps)
    d1 = (np.log(F_safe / K_safe) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    Phi_d1 = norm_cdf(d1)
    Phi_d2 = norm_cdf(d2)
    return (P_t_Tpay * delta) * (F * Phi_d1 - K * Phi_d2)


def portfolio_clean_value(
    t: float,
    X_t: np.ndarray,  # shape (n_paths, d) or (d,)
    portfolio: CapletPortfolio,
    disc: DiscountCurve,
    vol: VolModel,
) -> np.ndarray:
    """
    Compute portfolio clean value \bar V_t along paths at time t.
    X_t: forwards vector(s) at time t.
    Returns array shape (n_paths,) (or scalar if input is (d,))
    """
    X_t = np.asarray(X_t, dtype=float)
    if X_t.ndim == 1:
        X_t = X_t[None, :]
    n_paths, d = X_t.shape
    if d != portfolio.tenor.d:
        raise ValueError("X_t dimension mismatch with tenor.d")

    V = np.zeros((n_paths,), dtype=float)
    for cap in portfolio.caplets:
        i = cap.i
        T_reset = portfolio.tenor.reset_time(i)
        T_pay = portfolio.tenor.pay_time(i)
        if t > T_reset:
            # after reset, clean value is known? For simplicity, we can treat it as discounted intrinsic
            # (You can refine if you model F_i beyond reset; most grids will include reset times anyway.)
            # payoff at T_pay: delta*(F_i(T_reset)-K)^+ ; value at t uses P(t,T_pay) deterministic
            # Here we approximate F_i(T_reset) by current X_t[:, i-1] if t ~ T_reset.
            pass

        delta = portfolio.tenor.accrual(i)
        P_t_Tpay = disc.PtT(t, T_pay)
        int_var = vol.int_var(i, t, T_reset)

        F_i = X_t[:, i - 1]
        v_per_notional = black_caplet_price(F_i, cap.K, P_t_Tpay, delta, int_var)
        sign = 1.0 if cap.is_long else -1.0
        V += sign * cap.notional * v_per_notional

    return V


# ----------------------------
# XVA driver (CVA + FVA only)
# ----------------------------

class XvaDriver:
    """
    Driver for reduced \mathbb{F}-BSDE:
      -dXVA_t = f(t, X_t, XVA_t) dt - Z_t dW_t
    with XVA_T = 0.

    We implement:
      f = CVA_term + FVA_term - \tilde r(t) * XVA
    where
      CVA_term = -(1-R_C) * (Vbar - C)^- * lambda_C
      FVA_term = (r_fl - r) * (Vbar - XVA - C)^+ - (r_fb - r) * (Vbar - XVA - C)^-
      \tilde r = r + lambda_C    (since no bank default)
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
    portfolio: CapletPortfolio,
    disc: DiscountCurve,
    vol: VolModel,
    collateral: CollateralModel,
    cpty: CounterpartyModel,
) -> XvaProblemData:
    """
    Given simulated paths (t, X, dW), compute:
      - Vbar(t_n) pathwise from Black caplet formulas
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
        Vbar[:, n] = portfolio_clean_value(tn, X[:, n, :], portfolio, disc, vol)
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
    *,
    d: int = 80,
    T_max: float = 20.0,
    dt: float = 1.0 / 52.0,      # weekly
    n_paths: int = 50_000,
    seed: int = 123,
    threshold_H: float = 1e6,
) -> Tuple[XvaProblem, Dict[str, Any]]:
    """
    Returns:
      - XvaProblem (ready for a BSDE solver)
      - metadata dict with portfolio & model summary

    NOTE: n_paths=50k is heavy; adjust for your GPU/CPU.
    """
    # Tenor: quarterly reset/pay for d caplets -> need d+2 tenor points
    # We'll create quarterly tenor regardless of dt
    tenor_dt = 0.25
    T = np.arange(0.0, (d + 2) * tenor_dt, tenor_dt)  # length d+2
    tenor = TenorGrid(T=T)

    # Deterministic discount curve: flat r=3%
    r0 = 0.03
    disc = DiscountCurve(discount_0t=lambda t: math.exp(-r0 * t))

    # Vols: simple hump or flat; start flat 20%
    sigmas = 0.20 * np.ones((tenor.d,), dtype=float)
    vol = VolModel(d=tenor.d, sigmas_const=sigmas)

    # Correlation: exp in tenor time
    corr = make_exp_corr(tenor.d, tenor.T, a=0.2)

    # Time grid for BSDE/simulation (0..T_max)
    time_grid = np.arange(0.0, T_max + 1e-12, dt)
    time_grid = np.unique(np.clip(time_grid, 0.0, T_max))

    # Initial forward curve: e.g. upward sloping 3%->4%
    F0 = np.linspace(0.03, 0.04, tenor.d)

    sim = ForwardRateSimulator(tenor=tenor, vol=vol, corr=corr, time_grid=time_grid, seed=seed).simulate(
        n_paths=n_paths, F0=F0
    )

    # Portfolio: many caplets across tenors & strikes
    # Example: 5 strikes per tenor, 1k notional each; random long/short for netting richness
    rng = np.random.default_rng(seed + 1)
    caplets: List[Caplet] = []
    strikes_bps = np.array([-100, -50, 0, 50, 100]) * 1e-4  # +/- 100bp around ATM proxy
    for i in range(1, tenor.d + 1):
        # proxy "ATM" strike as initial forward
        K_atm = float(F0[i - 1])
        for bump in strikes_bps:
            K = max(1e-4, K_atm + float(bump))
            notional = 1e6
            is_long = bool(rng.integers(0, 2))
            caplets.append(Caplet(i=i, K=K, notional=notional, is_long=is_long))
    portfolio = CapletPortfolio(tenor=tenor, caplets=caplets)

    # Collateral (threshold CSA)
    collateral = CollateralModel(threshold_H=threshold_H)

    # Counterparty hazard: flat 150bp
    knots = np.array([0.0, T_max])
    lambdas = np.array([0.015])  # piecewise constant
    cpty = CounterpartyModel(knots=knots, lambdas=lambdas, recovery=0.40)

    # Funding: borrow spread 100bp, lend spread 0bp
    funding = FundingModel(
        spread_borrow=lambda t: 0.010,
        spread_lend=lambda t: 0.000,
    )

    # Build driver and training data
    driver = XvaDriver(disc=disc, cpty=cpty, funding=funding, collateral=collateral)
    data = build_xva_training_data(sim, portfolio, disc, vol, collateral, cpty)
    problem = XvaProblem(data=data, driver=driver)

    meta = {
        "d": tenor.d,
        "M": portfolio.M,
        "T_max": T_max,
        "dt": dt,
        "n_paths": n_paths,
        "r0": r0,
        "lambda_C": float(lambdas[0]),
        "recovery_C": cpty.R,
        "funding_spread_b": 0.010,
        "funding_spread_l": 0.000,
        "threshold_H": threshold_H,
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
    problem, meta = example_build_problem(
        d=80,
        T_max=20.0,
        dt=1.0 / 52.0,
        n_paths=20_000,   # start smaller
        seed=123,
        threshold_H=1e6,
    )

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
