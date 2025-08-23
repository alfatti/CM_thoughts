params = rfq.ModelParams(lambda_b=0.8, lambda_a=1.0, z=1.0, sigma=0.02, kappa=0.0, gamma=5.0, T=5.0, qbar=10.0, n_time=1500)
sc_b = rfq.logistic_scurve(a=0.0, b=1.2, delta_min=-6, delta_max=6)
sc_a = rfq.logistic_scurve(a=0.2, b=1.0, delta_min=-6, delta_max=6)
engine, alpha, ABC = rfq.build_pipeline(params, sc_b, sc_a)

def simulate_reference_price_path(params: rfq.ModelParams, S0: float, t_grid: np.ndarray, seed: int = 42):
    rng = np.random.default_rng(seed)
    S = np.empty_like(t_grid, dtype=float)
    S[0] = S0
    drift = params.kappa * (params.lambda_a - params.lambda_b)
    for i in range(1, len(t_grid)):
        dt = t_grid[i] - t_grid[i-1]
        dW = np.sqrt(dt) * rng.standard_normal()
        S[i] = S[i-1] + params.sigma * dW + drift * dt
    return S

def plot_quotes_over_time(engine: rfq.QuoteEngine, S0: float, q: float = 0.0, seed: int = 42, n_delta: int = 2001):
    t_grid = engine.ABC["t"]
    bid_offsets = np.empty_like(t_grid, dtype=float)
    ask_offsets = np.empty_like(t_grid, dtype=float)
    for i, t in enumerate(t_grid):
        db, da = engine.quotes(t=float(t), q=float(q), n_delta=n_delta)
        bid_offsets[i] = np.nan if db is None else -float(db)
        ask_offsets[i] = np.nan if da is None else  float(da)
    S = simulate_reference_price_path(engine.params, S0=S0, t_grid=t_grid, seed=seed)
    bid_abs = S + bid_offsets
    ask_abs = S + ask_offsets
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_grid, S, label="Reference price $S_t$")
    ax.plot(t_grid, bid_abs, label="Bid quote $S_t - \\delta_b^*$")
    ax.plot(t_grid, ask_abs, label="Ask quote $S_t + \\delta_a^*$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price level")
    ax.set_title("Absolute Bid/Ask Quotes Over Time with Reference Price")
    ax.legend(loc="best")
    plt.show()

plot_quotes_over_time(engine, S0=100.0, q=0.0, seed=7, n_delta=2001)


+++++++++++++++++++++++++++++++++++++++++
# -------------------------------------------------------------------
# Simulation + plotting: reference price, bid/ask quotes, inventory, cash
# -------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def simulate_quotes_and_state(
    engine,
    S0: float,
    q0: float,
    X0: float = 0.0,
    n_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate reference price S_t, inventory q_t, cash X_t, and optimal quotes
    using the closed-form approximation (A,B,C) inside `engine`.

    Discrete-time Euler/thinning simulation over [0,T] with step dt.
    At each step:
      - Interpolate A(t), B(t), compute p^b, p^a -> δ_b*, δ_a* (with caps).
      - Update S via arithmetic Brownian: dS = sigma dW + kappa(λ_a-λ_b) dt.
      - Execute RFQ trades via thinning:
            buy (bid-side)   with prob λ_b dt * f^b(δ_b*), at price S - δ_b*
            sell (ask-side)  with prob λ_a dt * f^a(δ_a*), at price S + δ_a*
        (if a side is capped out, it's skipped automatically).
      - Update inventory q and cash X accordingly.

    Returns a dict of arrays for time, S, bid_quote, ask_quote, q, X, deltas.
    """
    rng = np.random.default_rng(seed)

    p = engine.params
    sc_b, sc_a = engine.scurve_b, engine.scurve_a

    # Time grid (match ABC grid length unless overridden)
    T = p.T
    if n_steps is None:
        # use the ABC grid resolution
        t_grid = engine.ABC["t"]
        # ensure starts at 0
        if t_grid[0] > 1e-12:
            t_grid = np.concatenate([[0.0], t_grid])
        if abs(t_grid[-1] - T) > 1e-12:
            t_grid[-1] = T
    else:
        t_grid = np.linspace(0.0, T, n_steps + 1)

    dt = float(t_grid[1] - t_grid[0])
    sqrt_dt = np.sqrt(dt)

    n = len(t_grid)
    S = np.empty(n); S[0] = S0
    q = np.empty(n); q[0] = q0
    X = np.empty(n); X[0] = X0

    # Quotes (absolute) and offsets (delta)
    bid = np.full(n, np.nan)
    ask = np.full(n, np.nan)
    delt_b = np.full(n, np.nan)
    delt_a = np.full(n, np.nan)

    for i in range(n - 1):
        t = t_grid[i]
        # Optimal offsets (respecting caps)
        db, da = engine.quotes(t=t, q=float(q[i]))
        delt_b[i] = db if db is not None else np.nan
        delt_a[i] = da if da is not None else np.nan

        # Absolute quotes
        if db is not None: bid[i] = S[i] - db
        if da is not None: ask[i] = S[i] + da

        # Reference price update: arithmetic Brownian motion with drift from flow imbalance
        dW = rng.normal(0.0, 1.0) * sqrt_dt
        drift = p.kappa * (p.lambda_a - p.lambda_b) * dt
        S[i + 1] = S[i] + p.sigma * dW + drift

        # Simulate RFQ conversions via thinning (at most one fill per side per dt)
        # Probabilities for a fill on each side within [t, t+dt]:
        #    P_bid = λ_b dt f^b(δ_b),   P_ask = λ_a dt f^a(δ_a)
        # Note: if a side is unavailable (cap) -> δ is NaN -> skip.
        qb, qa = q[i], q[i]  # current inventory for cap checks

        # Bid-side (buying, increases inventory)
        if db is not None and (qb + p.z) <= p.qbar:
            fb = float(sc_b.f(np.array([db]))[0])
            Pb = max(0.0, min(1.0, p.lambda_b * dt * fb))
            if rng.uniform() < Pb:
                # Execute buy at S - δ_b
                trade_price = S[i] - db
                q[i + 1] = qb + p.z
                X[i + 1] = X[i] - trade_price * p.z
            else:
                q[i + 1] = qb
                X[i + 1] = X[i]
        else:
            q[i + 1] = qb
            X[i + 1] = X[i]

        # Ask-side (selling, decreases inventory)
        if da is not None and (q[i + 1] - p.z) >= -p.qbar:
            fa = float(sc_a.f(np.array([da]))[0])
            Pa = max(0.0, min(1.0, p.lambda_a * dt * fa))
            if rng.uniform() < Pa:
                # Execute sell at S + δ_a
                trade_price = S[i] + da
                q[i + 1] = q[i + 1] - p.z
                X[i + 1] = X[i + 1] + trade_price * p.z

    # Last-step quotes at T (just for plotting continuity)
    dbT, daT = engine.quotes(t=t_grid[-1], q=float(q[-1]))
    delt_b[-1] = dbT if dbT is not None else np.nan
    delt_a[-1] = daT if daT is not None else np.nan
    if dbT is not None: bid[-1] = S[-1] - dbT
    if daT is not None: ask[-1] = S[-1] + daT

    return dict(
        t=t_grid, S=S, bid=bid, ask=ask, q=q, X=X, delta_b=delt_b, delta_a=delt_a
    )


def plot_quotes_inventory_cash(sim_out: dict, title: Optional[str] = None) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Plot S_t with bid/ask, plus inventory and cash in separate subplots.

    Parameters
    ----------
    sim_out : dict
        Output from simulate_quotes_and_state(...).
        Keys: 't','S','bid','ask','q','X','delta_b','delta_a'
    title : Optional[str]
        Figure title.

    Returns
    -------
    (fig, (ax_price, ax_q, ax_cash))
    """
    t = sim_out["t"]
    S = sim_out["S"]
    bid = sim_out["bid"]
    ask = sim_out["ask"]
    q = sim_out["q"]
    X = sim_out["X"]

    fig = plt.figure(figsize=(12, 8))

    # Panel 1: Reference price + quotes
    ax_price = fig.add_subplot(3, 1, 1)
    ax_price.plot(t, S, lw=1.6, label="Reference price $S_t$")
    ax_price.plot(t, bid, lw=1.1, linestyle="--", label="Bid quote $S_t - \\delta^*_b$")
    ax_price.plot(t, ask, lw=1.1, linestyle="--", label="Ask quote $S_t + \\delta^*_a$")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="best")
    ax_price.grid(True, alpha=0.25)

    # Panel 2: Inventory
    ax_q = fig.add_subplot(3, 1, 2, sharex=ax_price)
    ax_q.step(t, q, where="post", label="Inventory $q_t$")
    ax_q.set_ylabel("Inventory")
    ax_q.grid(True, alpha=0.25)
    ax_q.legend(loc="best")

    # Panel 3: Cash
    ax_cash = fig.add_subplot(3, 1, 3, sharex=ax_price)
    ax_cash.plot(t, X, lw=1.6, label="Cash $X_t$")
    ax_cash.set_xlabel("Time")
    ax_cash.set_ylabel("Cash")
    ax_cash.grid(True, alpha=0.25)
    ax_cash.legend(loc="best")

    if title:
        fig.suptitle(title, y=0.98)

    fig.tight_layout()
    return fig, (ax_price, ax_q, ax_cash)
+++++++++++++++++++++++++++++++++++++++++++++
# Assuming you already have `engine` from build_pipeline(...)

# 1) Simulate from S0, q0
sim = simulate_quotes_and_state(
    engine,
    S0=100.0,    # starting reference price
    q0=0.0,      # starting inventory
    X0=0.0,      # starting cash
    n_steps=1000,
    seed=42
)

# 2) Plot
fig, (ax_price, ax_q, ax_cash) = plot_quotes_inventory_cash(
    sim, title="RFQ Quotes, Reference Price, Inventory, and Cash"
)

# (Optional) Save the figure:
# fig.savefig("quotes_inventory_cash.png", dpi=150, bbox_inches="tight")
