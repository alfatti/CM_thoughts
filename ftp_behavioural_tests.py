# Create the behavioral testing module and a tiny smoke test.
from textwrap import dedent

code = dedent(r'''
# rfq_behavioral_tests.py
# -------------------------------------------------------------
# Behavioral testing framework for RFQ optimal quoting policies
# built on top of rfq_quotes_pipeline.py
#
# Produces structured metrics for:
#   - Scenario analysis over (sigma, mu) where mu = kappa*(lambda_a - lambda_b)
#   - Risk aversion gamma sweep
#   - Horizon T sweep
#   - Inventory limit qbar sweep
#
# Optional: saves CSV artifacts and simple matplotlib figures.
# -------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from rfq_quotes_pipeline import (
    ModelParams, Scurve, logistic_scurve,
    build_pipeline
)

# --------------------------
# Utility: metrics from engine on a (t,q) grid
# --------------------------

@dataclass
class MetricsConfig:
    n_t: int = 51
    n_q: int = 41
    t0: float = 0.0      # evaluate from t0 to T (inclusive)
    save_dir: Optional[str] = None
    n_delta_quotes: int = 1201  # resolution of delta-search when quoting


def evaluate_grid_metrics(engine, t_grid: np.ndarray, q_grid: np.ndarray, n_delta: int = 1201) -> Dict[str, np.ndarray]:
    """
    Compute δ_b*, δ_a*, spread, ftp, and cap-activity mask over the (t,q) grid.
    Returns dict of arrays with shape (len(t_grid), len(q_grid)).
    """
    Nt = len(t_grid)
    Nq = len(q_grid)
    db = np.full((Nt, Nq), np.nan, dtype=float)
    da = np.full((Nt, Nq), np.nan, dtype=float)
    spread = np.full((Nt, Nq), np.nan, dtype=float)
    ftp = np.full((Nt, Nq), np.nan, dtype=float)
    cap_mask_bid_off = np.zeros((Nt, Nq), dtype=bool)
    cap_mask_ask_off = np.zeros((Nt, Nq), dtype=bool)

    for i, t in enumerate(t_grid):
        for j, q in enumerate(q_grid):
            b, a = engine.quotes(t=float(t), q=float(q), n_delta=n_delta)
            # Detect caps via None
            if b is None:
                cap_mask_bid_off[i, j] = True
            else:
                db[i, j] = b
            if a is None:
                cap_mask_ask_off[i, j] = True
            else:
                da[i, j] = a
            if np.isfinite(db[i, j]) and np.isfinite(da[i, j]):
                spread[i, j] = da[i, j] + db[i, j]
                ftp[i, j] = 0.5*(da[i, j] - db[i, j])

    return dict(db=db, da=da, spread=spread, ftp=ftp,
                cap_bid_off=cap_mask_bid_off, cap_ask_off=cap_mask_ask_off)


def inventory_slopes_at_origin(db: np.ndarray, da: np.ndarray, q_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Central-difference slopes ∂q δ_b*, ∂q δ_a* at q≈0 for each time.
    Expects arrays of shape (Nt, Nq). Returns vectors (Nt,).
    """
    Nt, Nq = db.shape
    # find closest indices around q=0
    j0 = np.argmin(np.abs(q_grid - 0.0))
    # pick neighbors for central diff; if j0 at edge, fallback to one-sided
    if 0 < j0 < Nq-1:
        dq = q_grid[j0+1] - q_grid[j0-1]
        slope_b = (db[:, j0+1] - db[:, j0-1]) / dq
        slope_a = (da[:, j0+1] - da[:, j0-1]) / dq
    elif j0 == 0:
        dq = q_grid[1] - q_grid[0]
        slope_b = (db[:, 1] - db[:, 0]) / dq
        slope_a = (da[:, 1] - da[:, 0]) / dq
    else:
        dq = q_grid[-1] - q_grid[-2]
        slope_b = (db[:, -1] - db[:, -2]) / dq
        slope_a = (da[:, -1] - da[:, -2]) / dq
    return slope_b, slope_a


def summarize_metrics(t_grid: np.ndarray, q_grid: np.ndarray, grids: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Produce scalar summaries:
      - spread_at_q0_t0, ftp_at_q0_t0
      - mean_spread_over_grid, mean_ftp_over_grid
      - cap_activity_fraction (either bid or ask disabled)
      - mean inventory slopes at origin (time-avg)
    """
    Nt, Nq = grids['spread'].shape
    # locate q0, t0
    i0 = 0  # assume t_grid starts at t0 per config
    j0 = np.argmin(np.abs(q_grid - 0.0))

    spread_at_q0_t0 = grids['spread'][i0, j0]
    ftp_at_q0_t0 = grids['ftp'][i0, j0]

    mask_valid = np.isfinite(grids['spread'])
    mean_spread = np.nanmean(grids['spread'][mask_valid])

    mask_valid_ftp = np.isfinite(grids['ftp'])
    mean_ftp = np.nanmean(grids['ftp'][mask_valid_ftp])

    cap_any = grids['cap_bid_off'] | grids['cap_ask_off']
    cap_fraction = cap_any.mean()

    slope_b, slope_a = inventory_slopes_at_origin(grids['db'], grids['da'], q_grid)
    mean_slope_b = np.nanmean(slope_b)
    mean_slope_a = np.nanmean(slope_a)

    return dict(
        spread_at_q0_t0=float(spread_at_q0_t0),
        ftp_at_q0_t0=float(ftp_at_q0_t0),
        mean_spread=float(mean_spread),
        mean_ftp=float(mean_ftp),
        cap_fraction=float(cap_fraction),
        mean_slope_b=float(mean_slope_b),
        mean_slope_a=float(mean_slope_a)
    )


def ensure_dir(path: Optional[str]):
    if path is not None and len(path) > 0:
        os.makedirs(path, exist_ok=True)


# --------------------------
# Experiment runners
# --------------------------

def _build_engine(params: ModelParams, sc_b: Scurve, sc_a: Scurve):
    engine, alpha, ABC = build_pipeline(params, sc_b, sc_a)
    return engine, alpha, ABC

def _default_grids(T: float, qbar: float, cfg: MetricsConfig):
    t_grid = np.linspace(cfg.t0, T, cfg.n_t)
    q_grid = np.linspace(-qbar, qbar, cfg.n_q)
    return t_grid, q_grid

def run_sigma_mu_grid(params_base: ModelParams,
                      sc_b: Scurve, sc_a: Scurve,
                      sigma_list: List[float], kappa_list: List[float],
                      cfg: MetricsConfig) -> pd.DataFrame:
    """
    Scenario analysis over (sigma, mu). We vary sigma directly and kappa which generates mu = kappa*(lambda_a - lambda_b).
    """
    ensure_dir(cfg.save_dir)
    rows = []
    for sigma in sigma_list:
        for kappa in kappa_list:
            params = ModelParams(
                lambda_b=params_base.lambda_b, lambda_a=params_base.lambda_a,
                z=params_base.z, sigma=sigma, kappa=kappa, gamma=params_base.gamma,
                T=params_base.T, qbar=params_base.qbar, n_time=params_base.n_time
            )
            engine, alpha, ABC = _build_engine(params, sc_b, sc_a)
            t_grid, q_grid = _default_grids(params.T, params.qbar, cfg)
            grids = evaluate_grid_metrics(engine, t_grid, q_grid, n_delta=cfg.n_delta_quotes)
            summ = summarize_metrics(t_grid, q_grid, grids)
            summ.update(dict(sigma=sigma, kappa=kappa, mu=kappa*(params.lambda_a-params.lambda_b)))
            rows.append(summ)

    df = pd.DataFrame(rows)
    if cfg.save_dir:
        df.to_csv(os.path.join(cfg.save_dir, "scenario_sigma_mu.csv"), index=False)
    return df


def run_gamma_sweep(params_base: ModelParams,
                    sc_b: Scurve, sc_a: Scurve,
                    gamma_list: List[float],
                    cfg: MetricsConfig) -> pd.DataFrame:
    ensure_dir(cfg.save_dir)
    rows = []
    for gamma in gamma_list:
        params = ModelParams(
            lambda_b=params_base.lambda_b, lambda_a=params_base.lambda_a,
            z=params_base.z, sigma=params_base.sigma, kappa=params_base.kappa, gamma=gamma,
            T=params_base.T, qbar=params_base.qbar, n_time=params_base.n_time
        )
        engine, alpha, ABC = _build_engine(params, sc_b, sc_a)
        t_grid, q_grid = _default_grids(params.T, params.qbar, cfg)
        grids = evaluate_grid_metrics(engine, t_grid, q_grid, n_delta=cfg.n_delta_quotes)
        summ = summarize_metrics(t_grid, q_grid, grids)
        summ.update(dict(gamma=gamma))
        rows.append(summ)

    df = pd.DataFrame(rows)
    if cfg.save_dir:
        df.to_csv(os.path.join(cfg.save_dir, "sweep_gamma.csv"), index=False)
    return df


def run_T_sweep(params_base: ModelParams,
                sc_b: Scurve, sc_a: Scurve,
                T_list: List[float],
                cfg: MetricsConfig) -> pd.DataFrame:
    ensure_dir(cfg.save_dir)
    rows = []
    for T in T_list:
        params = ModelParams(
            lambda_b=params_base.lambda_b, lambda_a=params_base.lambda_a,
            z=params_base.z, sigma=params_base.sigma, kappa=params_base.kappa, gamma=params_base.gamma,
            T=T, qbar=params_base.qbar, n_time=params_base.n_time
        )
        engine, alpha, ABC = _build_engine(params, sc_b, sc_a)
        t_grid, q_grid = _default_grids(params.T, params.qbar, cfg)
        grids = evaluate_grid_metrics(engine, t_grid, q_grid, n_delta=cfg.n_delta_quotes)
        summ = summarize_metrics(t_grid, q_grid, grids)
        summ.update(dict(T=T))
        rows.append(summ)

    df = pd.DataFrame(rows)
    if cfg.save_dir:
        df.to_csv(os.path.join(cfg.save_dir, "sweep_T.csv"), index=False)
    return df


def run_qbar_sweep(params_base: ModelParams,
                   sc_b: Scurve, sc_a: Scurve,
                   qbar_list: List[float],
                   cfg: MetricsConfig) -> pd.DataFrame:
    ensure_dir(cfg.save_dir)
    rows = []
    for qbar in qbar_list:
        params = ModelParams(
            lambda_b=params_base.lambda_b, lambda_a=params_base.lambda_a,
            z=params_base.z, sigma=params_base.sigma, kappa=params_base.kappa, gamma=params_base.gamma,
            T=params_base.T, qbar=qbar, n_time=params_base.n_time
        )
        engine, alpha, ABC = _build_engine(params, sc_b, sc_a)
        t_grid, q_grid = _default_grids(params.T, params.qbar, cfg)
        grids = evaluate_grid_metrics(engine, t_grid, q_grid, n_delta=cfg.n_delta_quotes)
        summ = summarize_metrics(t_grid, q_grid, grids)
        summ.update(dict(qbar=qbar))
        rows.append(summ)

    df = pd.DataFrame(rows)
    if cfg.save_dir:
        df.to_csv(os.path.join(cfg.save_dir, "sweep_qbar.csv"), index=False)
    return df


# --------------------------
# Optional plotting helpers (matplotlib; one plot per figure)
# --------------------------

def plot_time_profile(t_grid: np.ndarray, y: np.ndarray, title: str, path: Optional[str] = None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t_grid, y)
    plt.xlabel("t")
    plt.ylabel(title)
    plt.title(title)
    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_vs_param(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, path: Optional[str] = None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if path:
        plt.savefig(path, bbox_inches='tight')
    plt.close()


# --------------------------
# Minimal demo
# --------------------------

def _demo():
    outdir = "/mnt/data/behavioral_artifacts"
    os.makedirs(outdir, exist_ok=True)

    params = ModelParams(
        lambda_b=0.8, lambda_a=1.0,
        z=1.0, sigma=0.02, kappa=0.0, gamma=5.0,
        T=4.0, qbar=8.0, n_time=1500
    )
    sc_b = logistic_scurve(a=0.0, b=1.2, delta_min=-6, delta_max=6)
    sc_a = logistic_scurve(a=0.2, b=1.0, delta_min=-6, delta_max=6)

    cfg = MetricsConfig(n_t=31, n_q=31, t0=0.0, save_dir=outdir, n_delta_quotes=801)

    # Scenario: sigma–kappa grid
    df_sigmamu = run_sigma_mu_grid(params, sc_b, sc_a,
                                   sigma_list=[0.01, 0.02, 0.05],
                                   kappa_list=[-0.5, 0.0, 0.5],
                                   cfg=cfg)

    # Risk aversion sweep
    df_gamma = run_gamma_sweep(params, sc_b, sc_a,
                               gamma_list=[1.0, 3.0, 5.0, 8.0],
                               cfg=cfg)

    # Horizon sweep
    df_T = run_T_sweep(params, sc_b, sc_a,
                       T_list=[2.0, 4.0, 8.0],
                       cfg=cfg)

    # Inventory cap sweep
    df_qbar = run_qbar_sweep(params, sc_b, sc_a,
                             qbar_list=[4.0, 8.0, 16.0],
                             cfg=cfg)

    # Save quick plots for a few summaries
    plot_vs_param(df_gamma["gamma"].to_numpy(), df_gamma["mean_spread"].to_numpy(),
                  xlabel="gamma", ylabel="mean spread", title="Mean Spread vs Gamma",
                  path=os.path.join(outdir, "mean_spread_vs_gamma.png"))

    plot_vs_param(df_T["T"].to_numpy(), df_T["spread_at_q0_t0"].to_numpy(),
                  xlabel="T", ylabel="spread at (t0,q0)", title="Spread at (t0,q0) vs T",
                  path=os.path.join(outdir, "spread_at_q0_t0_vs_T.png"))

    # Return paths
    return dict(
        sigmamu_csv=os.path.join(outdir, "scenario_sigma_mu.csv"),
        gamma_csv=os.path.join(outdir, "sweep_gamma.csv"),
        T_csv=os.path.join(outdir, "sweep_T.csv"),
        qbar_csv=os.path.join(outdir, "sweep_qbar.csv"),
        plot1=os.path.join(outdir, "mean_spread_vs_gamma.png"),
        plot2=os.path.join(outdir, "spread_at_q0_t0_vs_T.png")
    )


if __name__ == "__main__":
    paths = _demo()
    for k, v in paths.items():
        print(f"{k}: {v}")
''')

with open('/mnt/data/rfq_behavioral_tests.py', 'w') as f:
    f.write(code)

print("Saved to /mnt/data/rfq_behavioral_tests.py")

# Run a tiny smoke test to produce artifacts quickly
import subprocess, sys, os, json
try:
    out = subprocess.check_output([sys.executable, "/mnt/data/rfq_behavioral_tests.py"], stderr=subprocess.STDOUT, timeout=180)
    print(out.decode('utf-8'))
    print("Artifacts created under /mnt/data/behavioral_artifacts")
except subprocess.CalledProcessError as e:
    print("Script error:\n", e.output.decode("utf-8"))
except Exception as e:
    print("Could not run demo:", repr(e))

#======================================================================
$Quick start

from rfq_behavioral_tests import (
    ModelParams, logistic_scurve, MetricsConfig,
    run_sigma_mu_grid, run_gamma_sweep, run_T_sweep, run_qbar_sweep
)

# Base model
params = ModelParams(
    lambda_b=0.8, lambda_a=1.0,
    z=1.0, sigma=0.02, kappa=0.0, gamma=5.0,
    T=4.0, qbar=8.0, n_time=1500
)

# S-curves
sc_b = logistic_scurve(a=0.0, b=1.2, delta_min=-6, delta_max=6)
sc_a = logistic_scurve(a=0.2, b=1.0, delta_min=-6, delta_max=6)

cfg = MetricsConfig(n_t=31, n_q=31, t0=0.0, save_dir="/tmp/behavioral_artifacts", n_delta_quotes=801)

# 1) Volatility–drift grid (vary sigma and kappa -> mu = kappa*(λa-λb))
df_sigmamu = run_sigma_mu_grid(params, sc_b, sc_a,
                               sigma_list=[0.01, 0.02, 0.05],
                               kappa_list=[-0.5, 0.0, 0.5],
                               cfg=cfg)

# 2) Risk aversion sweep
df_gamma = run_gamma_sweep(params, sc_b, sc_a,
                           gamma_list=[1.0, 3.0, 5.0, 8.0],
                           cfg=cfg)

# 3) Horizon sweep
df_T = run_T_sweep(params, sc_b, sc_a,
                   T_list=[2.0, 4.0, 8.0],
                   cfg=cfg)

# 4) Inventory cap sweep
df_qbar = run_qbar_sweep(params, sc_b, sc_a,
                         qbar_list=[4.0, 8.0, 16.0],
                         cfg=cfg)





