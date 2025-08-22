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
