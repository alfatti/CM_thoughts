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
def simulate_and_plot_quotes(engine, S0, n_steps=None, seed=None, show=True):
    """
    Simulate and plot:
      - Reference price S_t
      - Final bid/ask quotes S_t ± δ*_t (from engine)
      - Inventory q_t
      - Cash process X_t

    Assumptions (discrete-time Euler):
      dS_t = sigma * dW_t + kappa * (lambda_a - lambda_b) dt
      Trade arrivals at side s ∈ {bid, ask} within [t, t+dt):
          Bernoulli( λ_s * dt * f_s(δ_s*(t, q_t)) ).
      Inventory cap logic is enforced via engine.quotes (it returns None on a side if the next trade would breach the cap).

    Parameters
    ----------
    engine : QuoteEngine
        Built from build_pipeline(...). Provides quotes(t, q) and S-curves.
    S0 : float
        Initial reference price.
    n_steps : int, optional
        Number of time steps between 0 and T. Defaults to max(params.n_time, 1000).
    seed : int, optional
        Random seed for reproducibility.
    show : bool, default True
        If True, calls plt.show().

    Returns
    -------
    sim : dict
        Dictionary with arrays: t, S, bid_quote, ask_quote, q, X, delta_b, delta_a, traded_bid, traded_ask.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Unpack model
    m = engine.params
    sc_b, sc_a = engine.scurve_b, engine.scurve_a
    Lb, La = m.lambda_b, m.lambda_a
    z = m.z
    sigma = m.sigma
    kappa = m.kappa
    T = m.T

    # Time grid
    if n_steps is None:
        n_steps = max(m.n_time, 1000)
    t = np.linspace(0.0, T, n_steps + 1)
    dt = T / n_steps
    sdt = np.sqrt(dt)

    # State arrays
    S = np.empty(n_steps + 1)
    q = np.empty(n_steps + 1)
    X = np.empty(n_steps + 1)  # cash

    delta_b = np.full(n_steps + 1, np.nan)
    delta_a = np.full(n_steps + 1, np.nan)
    bid_quote = np.full(n_steps + 1, np.nan)
    ask_quote = np.full(n_steps + 1, np.nan)

    traded_bid = np.zeros(n_steps + 1, dtype=int)  # 1 if trade at bid occurred at step i
    traded_ask = np.zeros(n_steps + 1, dtype=int)  # 1 if trade at ask occurred at step i

    # Init
    rng = np.random.default_rng(seed)
    S[0] = float(S0)
    q[0] = 0.0
    X[0] = 0.0

    # Simulate
    for i in range(n_steps):
        ti = t[i]
        qi = q[i]
        Si = S[i]

        # Optimal offsets (enforces inventory caps)
        db, da = engine.quotes(t=ti, q=qi)
        delta_b[i] = db if db is not None else np.nan
        delta_a[i] = da if da is not None else np.nan

        # Final quotes (absolute)
        bid_quote[i] = Si - db if db is not None else np.nan
        ask_quote[i] = Si + da if da is not None else np.nan

        # Trade probabilities this step
        # If a side is disabled (None), its probability is zero
        pb = 0.0
        pa = 0.0
        if db is not None:
            fb = sc_b.f(np.array([db], dtype=float))[0]
            pb = max(0.0, min(1.0, Lb * dt * float(fb)))
        if da is not None:
            fa = sc_a.f(np.array([da], dtype=float))[0]
            pa = max(0.0, min(1.0, La * dt * float(fa)))

        # Realizations: at most one trade per side per step
        trade_b = 1 if rng.random() < pb else 0
        trade_a = 1 if rng.random() < pa else 0

        # Apply trades (cash and inventory)
        # Bid trade: we BUY z at price S - δ_b  -> inventory +z, cash -z*(S - δ_b)
        if trade_b and db is not None and (qi + z) <= m.qbar:
            q[i+1] = qi + z
            X[i+1] = X[i] - z * (Si - db)
            traded_bid[i] = 1
        else:
            q[i+1] = qi
            X[i+1] = X[i]

        # Ask trade: we SELL z at price S + δ_a -> inventory -z, cash +z*(S + δ_a)
        # Use q[i+1] since bid could have filled in this step first (conservative sequencing).
        if trade_a and da is not None and (q[i+1] - z) >= -m.qbar:
            q[i+1] = q[i+1] - z
            X[i+1] = X[i+1] + z * (Si + da)
            traded_ask[i] = 1

        # Reference price evolution (Euler)
        dW = sdt * rng.standard_normal()
        S[i+1] = Si + sigma * dW + kappa * (La - Lb) * dt

    # Last-step quotes for plotting completeness
    delta_b[-1], delta_a[-1] = engine.quotes(t=t[-1], q=q[-1])
    bid_quote[-1] = S[-1] - delta_b[-1] if not np.isnan(delta_b[-1]) else np.nan
    ask_quote[-1] = S[-1] + delta_a[-1] if not np.isnan(delta_a[-1]) else np.nan

    # Package results
    sim = dict(
        t=t, S=S,
        bid_quote=bid_quote, ask_quote=ask_quote,
        q=q, X=X,
        delta_b=delta_b, delta_a=delta_a,
        traded_bid=traded_bid, traded_ask=traded_ask,
    )

    # ---------- Plot ----------
    fig = plt.figure(figsize=(11, 8))

    # Panel 1: S and quotes
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, S, label="Reference price $S_t$")
    ax1.plot(t, bid_quote, label="Bid quote $S_t - \\delta^{b,*}_t$")
    ax1.plot(t, ask_quote, label="Ask quote $S_t + \\delta^{a,*}_t$")
    ax1.set_ylabel("Price")
    ax1.set_title("Reference Price and Final Bid/Ask Quotes")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Inventory
    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, q, label="Inventory $q_t$")
    ax2.axhline(m.qbar, linestyle="--")
    ax2.axhline(-m.qbar, linestyle="--")
    ax2.set_ylabel("Inventory")
    ax2.set_title("Inventory Path (caps shown)")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Cash
    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t, X, label="Cash $X_t$")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Cash")
    ax3.set_title("Cash Process")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    if show:
        plt.show()

    return sim

