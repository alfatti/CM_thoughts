# Create three separate scripts as requested: 
# 1) rfq_export_surface.py
# 2) rfq_heatmaps.py
# 3) rfq_sensitivity_tables.py
# They depend on rfq_quotes_pipeline.py and rfq_behavioral_tests.py created earlier.

from textwrap import dedent
import os, json, pathlib

scripts = {}

scripts["/mnt/data/rfq_export_surface.py"] = dedent(r'''
# rfq_export_surface.py
# -------------------------------------------------------------
# Export quote surfaces δ_b*, δ_a*, spread, ftp on (t,q) grids.
# Requires: rfq_quotes_pipeline.py and rfq_behavioral_tests.py
# -------------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from rfq_quotes_pipeline import ModelParams, logistic_scurve, build_pipeline
from rfq_behavioral_tests import MetricsConfig, evaluate_grid_metrics

@dataclass
class SurfaceConfig:
    save_dir: str = "./quotes_surface"
    n_t: int = 61
    n_q: int = 61
    t0: float = 0.0
    n_delta_quotes: int = 1201

def main():
    # --- Base model params (edit as needed) ---
    params = ModelParams(
        lambda_b=0.8, lambda_a=1.0,
        z=1.0, sigma=0.02, kappa=0.0, gamma=5.0,
        T=6.0, qbar=10.0, n_time=2000
    )
    # S-curves (example: logistics; replace with your own if desired)
    sc_b = logistic_scurve(a=0.0, b=1.2, delta_min=-6, delta_max=6)
    sc_a = logistic_scurve(a=0.2, b=1.0, delta_min=-6, delta_max=6)

    cfg = SurfaceConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)
    # Build engine
    engine, alpha, ABC = build_pipeline(params, sc_b, sc_a)

    # Grids
    t_grid = np.linspace(cfg.t0, params.T, cfg.n_t)
    q_grid = np.linspace(-params.qbar, params.qbar, cfg.n_q)

    # Evaluate
    grids = evaluate_grid_metrics(engine, t_grid, q_grid, n_delta=cfg.n_delta_quotes)

    # Save tidy CSVs (long format)
    def to_long(name: str, arr: np.ndarray):
        Nt, Nq = arr.shape
        df = pd.DataFrame({
            "t": np.repeat(t_grid, Nq),
            "q": np.tile(q_grid, Nt),
            name: arr.reshape(-1)
        })
        return df

    tidy_db = to_long("delta_b", grids["db"])
    tidy_da = to_long("delta_a", grids["da"])
    tidy_spread = to_long("spread", grids["spread"])
    tidy_ftp = to_long("ftp", grids["ftp"])

    tidy_db.to_csv(os.path.join(cfg.save_dir, "surface_delta_b.csv"), index=False)
    tidy_da.to_csv(os.path.join(cfg.save_dir, "surface_delta_a.csv"), index=False)
    tidy_spread.to_csv(os.path.join(cfg.save_dir, "surface_spread.csv"), index=False)
    tidy_ftp.to_csv(os.path.join(cfg.save_dir, "surface_ftp.csv"), index=False)

    # Also save compact NPZ with all arrays + grids
    np.savez_compressed(
        os.path.join(cfg.save_dir, "surfaces.npz"),
        t_grid=t_grid, q_grid=q_grid,
        delta_b=grids["db"], delta_a=grids["da"],
        spread=grids["spread"], ftp=grids["ftp"],
        cap_bid_off=grids["cap_bid_off"], cap_ask_off=grids["cap_ask_off"]
    )

    # Save run metadata
    with open(os.path.join(cfg.save_dir, "run_meta.json"), "w") as f:
        import json
        json.dump({
            "params": params.__dict__,
            "alpha_b": getattr(engine, "scurve_b").name,
            "alpha_a": getattr(engine, "scurve_a").name,
            "config": cfg.__dict__
        }, f, indent=2)

    print("Saved surfaces to:", os.path.abspath(cfg.save_dir))

if __name__ == "__main__":
    main()
''')

scripts["/mnt/data/rfq_heatmaps.py"] = dedent(r'''
# rfq_heatmaps.py
# -------------------------------------------------------------
# Generate heatmaps (PNG) for δ_b*, δ_a*, spread, ftp at selected times.
# Requires: rfq_quotes_pipeline.py and rfq_behavioral_tests.py
# -------------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass
from typing import List

from rfq_quotes_pipeline import ModelParams, logistic_scurve, build_pipeline
from rfq_behavioral_tests import MetricsConfig, evaluate_grid_metrics

@dataclass
class HeatmapConfig:
    save_dir: str = "./quotes_heatmaps"
    n_t: int = 61
    n_q: int = 61
    t0: float = 0.0
    n_delta_quotes: int = 801
    times_to_plot: int = 3  # number of evenly-spaced time slices (including endpoints)

def _plot_heatmap(Z: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, path: str):
    import matplotlib.pyplot as plt
    plt.figure()
    # imshow expects [rows, cols] -> y, x
    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(Z, origin="lower", extent=extent, aspect="auto")
    plt.xlabel("q")
    plt.ylabel("t")
    plt.title(title)
    plt.colorbar()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def _plot_slice(y: np.ndarray, x: np.ndarray, title: str, path: str, xlabel: str, ylabel: str):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def main():
    # --- Base model params (edit as needed) ---
    params = ModelParams(
        lambda_b=0.8, lambda_a=1.0,
        z=1.0, sigma=0.02, kappa=0.0, gamma=5.0,
        T=6.0, qbar=10.0, n_time=2000
    )
    sc_b = logistic_scurve(a=0.0, b=1.2, delta_min=-6, delta_max=6)
    sc_a = logistic_scurve(a=0.2, b=1.0, delta_min=-6, delta_max=6)

    cfg = HeatmapConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    engine, alpha, ABC = build_pipeline(params, sc_b, sc_a)
    t_grid = np.linspace(cfg.t0, params.T, cfg.n_t)
    q_grid = np.linspace(-params.qbar, params.qbar, cfg.n_q)
    grids = evaluate_grid_metrics(engine, t_grid, q_grid, n_delta=cfg.n_delta_quotes)

    # Full heatmaps (t vs q)
    _plot_heatmap(grids["spread"], q_grid, t_grid, "Spread heatmap", os.path.join(cfg.save_dir, "spread_heatmap.png"))
    _plot_heatmap(grids["ftp"], q_grid, t_grid, "FTP heatmap", os.path.join(cfg.save_dir, "ftp_heatmap.png"))
    _plot_heatmap(grids["db"], q_grid, t_grid, "Delta_b heatmap", os.path.join(cfg.save_dir, "delta_b_heatmap.png"))
    _plot_heatmap(grids["da"], q_grid, t_grid, "Delta_a heatmap", os.path.join(cfg.save_dir, "delta_a_heatmap.png"))

    # Time-slice profiles at q=0
    j0 = int(np.argmin(np.abs(q_grid - 0.0)))
    spread_q0 = grids["spread"][:, j0]
    ftp_q0 = grids["ftp"][:, j0]
    _plot_slice(spread_q0, t_grid, "Spread at q=0 over time", os.path.join(cfg.save_dir, "spread_q0_time.png"), "t", "spread")
    _plot_slice(ftp_q0, t_grid, "FTP at q=0 over time", os.path.join(cfg.save_dir, "ftp_q0_time.png"), "t", "ftp")

    # q-slice profiles at t = {start, mid, end}
    t_slices = np.linspace(cfg.t0, params.T, cfg.times_to_plot)
    for k, t in enumerate(t_slices):
        i = int(np.argmin(np.abs(t_grid - t)))
        _plot_slice(grids["spread"][i, :], q_grid, f"Spread vs q at t≈{t_grid[i]:.2f}", os.path.join(cfg.save_dir, f"spread_vs_q_t{i}.png"), "q", "spread")
        _plot_slice(grids["ftp"][i, :], q_grid, f"FTP vs q at t≈{t_grid[i]:.2f}", os.path.join(cfg.save_dir, f"ftp_vs_q_t{i}.png"), "q", "ftp")

    print("Saved heatmaps to:", os.path.abspath(cfg.save_dir))

if __name__ == "__main__":
    main()
''')

scripts["/mnt/data/rfq_sensitivity_tables.py"] = dedent(r'''
# rfq_sensitivity_tables.py
# -------------------------------------------------------------
# Build elasticity/sensitivity tables for key metrics with respect to
# sigma, gamma, and T (finite differences around a base config).
# Requires: rfq_quotes_pipeline.py and rfq_behavioral_tests.py
# -------------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from rfq_quotes_pipeline import ModelParams, logistic_scurve, build_pipeline
from rfq_behavioral_tests import MetricsConfig, evaluate_grid_metrics, summarize_metrics

@dataclass
class SensitivityConfig:
    save_dir: str = "./sensitivity_tables"
    n_t: int = 31
    n_q: int = 31
    t0: float = 0.0
    n_delta_quotes: int = 801
    # step sizes for finite differences
    d_sigma: float = 0.002
    d_gamma: float = 0.5
    d_T: float = 0.5

def _metrics_for(params: ModelParams, sc_b, sc_a, cfg: SensitivityConfig):
    engine, alpha, ABC = build_pipeline(params, sc_b, sc_a)
    t_grid = np.linspace(cfg.t0, params.T, cfg.n_t)
    q_grid = np.linspace(-params.qbar, params.qbar, cfg.n_q)
    grids = evaluate_grid_metrics(engine, t_grid, q_grid, n_delta=cfg.n_delta_quotes)
    return summarize_metrics(t_grid, q_grid, grids)

def _central_diff(base, low, high, step):
    return (high - low) / (2.0 * step)

def main():
    cfg = SensitivityConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Base model
    base = ModelParams(
        lambda_b=0.8, lambda_a=1.0,
        z=1.0, sigma=0.02, kappa=0.0, gamma=5.0,
        T=6.0, qbar=10.0, n_time=1800
    )
    sc_b = logistic_scurve(a=0.0, b=1.2, delta_min=-6, delta_max=6)
    sc_a = logistic_scurve(a=0.2, b=1.0, delta_min=-6, delta_max=6)

    # --- Base metrics ---
    m0 = _metrics_for(base, sc_b, sc_a, cfg)

    # --- Sigma sensitivity ---
    p_low = ModelParams(**{**base.__dict__, "sigma": base.sigma - cfg.d_sigma})
    p_high = ModelParams(**{**base.__dict__, "sigma": base.sigma + cfg.d_sigma})
    m_sigma_low = _metrics_for(p_low, sc_b, sc_a, cfg)
    m_sigma_high = _metrics_for(p_high, sc_b, sc_a, cfg)

    # --- Gamma sensitivity ---
    p_low = ModelParams(**{**base.__dict__, "gamma": base.gamma - cfg.d_gamma})
    p_high = ModelParams(**{**base.__dict__, "gamma": base.gamma + cfg.d_gamma})
    m_gamma_low = _metrics_for(p_low, sc_b, sc_a, cfg)
    m_gamma_high = _metrics_for(p_high, sc_b, sc_a, cfg)

    # --- Horizon T sensitivity ---
    p_low = ModelParams(**{**base.__dict__, "T": base.T - cfg.d_T})
    p_high = ModelParams(**{**base.__dict__, "T": base.T + cfg.d_T})
    m_T_low = _metrics_for(p_low, sc_b, sc_a, cfg)
    m_T_high = _metrics_for(p_high, sc_b, sc_a, cfg)

    # Build table
    metrics = ["spread_at_q0_t0", "ftp_at_q0_t0", "mean_spread", "mean_ftp", "cap_fraction", "mean_slope_b", "mean_slope_a"]
    rows = []
    # Sigma
    rows.append({
        "param": "sigma",
        **{f"d{m}/d sigma": _central_diff(m0[m], m_sigma_low[m], m_sigma_high[m], cfg.d_sigma) for m in metrics}
    })
    # Gamma
    rows.append({
        "param": "gamma",
        **{f"d{m}/d gamma": _central_diff(m0[m], m_gamma_low[m], m_gamma_high[m], cfg.d_gamma) for m in metrics}
    })
    # T
    rows.append({
        "param": "T",
        **{f"d{m}/d T": _central_diff(m0[m], m_T_low[m], m_T_high[m], cfg.d_T) for m in metrics}
    })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(cfg.save_dir, "sensitivity_elasticities.csv")
    df.to_csv(out_csv, index=False)

    # Also save base metrics for reference
    pd.DataFrame([m0]).to_csv(os.path.join(cfg.save_dir, "base_metrics.csv"), index=False)

    print("Saved sensitivity tables to:", os.path.abspath(cfg.save_dir))

if __name__ == "__main__":
    main()
''')

# Write scripts to disk
for path, code in scripts.items():
    with open(path, "w") as f:
        f.write(code)

# List the created files for the user
print(json.dumps({"created": list(scripts.keys())}, indent=2))
