"""
XVA BSDE equation for dim-fbsde solver.

Maps the CVA+FVA BSDE from the caplet portfolio problem into the dim-fbsde
FBSDE interface. The forward process is the vector of forward rates F(t),
and the backward process Y = XVA_t satisfies:

    -dXVA_t = f(t, F(t), XVA_t) dt - Z_t · dW_t,   XVA_T = 0

where f aggregates CVA cost, FVA funding recursion, and intensity discounting.

Usage:
    problem, meta = example_build_problem(d=10, ...)   # from Caplet_portfolio_data
    result = solve_xva(problem, meta, device="cpu")
"""

import sys
import os
import math
import numpy as np
import torch

# Import dim-fbsde from sibling repo
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DIM_FBSDE_SRC = os.path.join(os.path.dirname(_REPO_ROOT), "dim-fbsde", "src")
if _DIM_FBSDE_SRC not in sys.path:
    sys.path.insert(0, _DIM_FBSDE_SRC)

from dim_fbsde.equations.base import FBSDE
from dim_fbsde.solvers.uncoupled import UncoupledFBSDESolver
from dim_fbsde.config import SolverConfig, TrainingConfig
from dim_fbsde.nets import MLP

# Import market data infrastructure
sys.path.insert(0, _REPO_ROOT)
from Caplet_portfolio_data import example_build_problem, build_xva_training_data


class XVACapletEquation(FBSDE):
    """
    FBSDE formulation for CVA+FVA on a caplet portfolio.

    Forward:  dF_i(t) = sigma_i F_i(t) dW_i^Q(t)   (lognormal, zero drift)
    Backward: -dXVA_t = f(t, F, XVA) dt - Z_t dW_t^Q,  XVA_T = 0

    The driver f (Eq. 6 in the PDF) is:
        f = -(1-R_C)(V̄-C)^- λ_C
            + s_l (V̄ - XVA - C)^+
            - s_b (V̄ - XVA - C)^-
            - (r + λ_C) XVA

    V̄ and C are precomputed for all paths and time steps.
    """

    def __init__(
        self,
        dim_x: int,
        F0: np.ndarray,
        sigma_vec: np.ndarray,
        Vbar: np.ndarray,
        C: np.ndarray,
        r_array: np.ndarray,
        lam_array: np.ndarray,
        recovery: float,
        spread_borrow: float,
        spread_lend: float,
        T: float,
        device: str = "cpu",
    ):
        """
        Args:
            dim_x: Number of forward rates (d).
            F0: Initial forward curve, shape (d,).
            sigma_vec: Volatilities per forward, shape (d,).
            Vbar: Precomputed clean portfolio value, shape (M, N+1).
            C: Precomputed collateral, shape (M, N+1).
            r_array: Deterministic short rate at each time step, shape (N+1,).
            lam_array: Counterparty hazard rate at each time step, shape (N+1,).
            recovery: Counterparty recovery rate R_C in [0, 1].
            spread_borrow: Funding borrow spread s_b (constant).
            spread_lend: Funding lend spread s_l (constant).
            T: Terminal time.
            device: Torch device.
        """
        super().__init__(
            dim_x=dim_x,
            dim_y=1,
            dim_w=dim_x,
            x0=torch.tensor(F0, dtype=torch.float32),
            device=device,
        )

        self._sigma = torch.tensor(sigma_vec, dtype=torch.float32, device=device)
        self._Vbar = torch.tensor(Vbar, dtype=torch.float32, device=device)
        self._C = torch.tensor(C, dtype=torch.float32, device=device)
        self._r = torch.tensor(r_array, dtype=torch.float32, device=device)
        self._lam = torch.tensor(lam_array, dtype=torch.float32, device=device)
        self._R_C = recovery
        self._s_b = spread_borrow
        self._s_l = spread_lend
        self._T = T
        self._N = Vbar.shape[1] - 1

    def _time_to_idx(self, t):
        """Convert scalar time to nearest time grid index."""
        if isinstance(t, torch.Tensor):
            t_val = t.item()
        else:
            t_val = float(t)
        return min(int(round(t_val * self._N / self._T)), self._N)

    def drift(self, t, x, y, z, **kwargs):
        """Zero drift (driftless lognormal dynamics under Q)."""
        return torch.zeros_like(x)

    def diffusion(self, t, x, y, z, **kwargs):
        """Diagonal diffusion: sigma_i * F_i(t). Shape [M, d, d]."""
        return torch.diag_embed(self._sigma.unsqueeze(0) * x)

    def driver(self, t, x, y, z, **kwargs):
        """
        XVA driver: CVA + FVA - (r + lambda) * XVA.

        Args:
            t: Scalar time.
            x: Forward rates [M, d].
            y: XVA value [M, 1].
            z: Control [M, 1, d].

        Returns:
            Driver value [M, 1].
        """
        idx = self._time_to_idx(t)
        M = x.shape[0]

        # Look up precomputed values
        Vbar = self._Vbar[:M, idx].unsqueeze(1)  # [M, 1]
        C_val = self._C[:M, idx].unsqueeze(1)     # [M, 1]
        r = self._r[idx].item()
        lam = self._lam[idx].item()

        # CVA term: -(1 - R_C) * (V̄ - C)^- * lambda_C
        exposure = Vbar - C_val
        cva = -(1.0 - self._R_C) * torch.clamp(-exposure, min=0.0) * lam

        # FVA term: s_l * (V̄ - XVA - C)^+ - s_b * (V̄ - XVA - C)^-
        funding_req = Vbar - y - C_val
        fva = self._s_l * torch.clamp(funding_req, min=0.0) \
            - self._s_b * torch.clamp(-funding_req, min=0.0)

        # Intensity discount: -(r + lambda) * XVA
        discount = -(r + lam) * y

        return cva + fva + discount

    def terminal_condition(self, x, **kwargs):
        """XVA_T = 0."""
        return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)


def solve_xva(
    d: int = 10,
    T_max: float = 5.0,
    dt: float = 1.0 / 12.0,
    n_paths: int = 5000,
    picard_iterations: int = 10,
    epochs: int = 5,
    learning_rate: float = 1e-4,
    hidden_dims: list = None,
    threshold_H: float = 1e6,
    seed: int = 123,
    device: str = "cpu",
):
    """
    End-to-end: build market data, simulate, create equation, solve XVA BSDE.

    Args:
        d: Number of forward rates (state dimension).
        T_max: Time horizon in years.
        dt: Simulation time step (e.g. 1/12 for monthly, 1/52 for weekly).
        n_paths: Number of Monte Carlo paths.
        picard_iterations: Number of Picard fixed-point iterations.
        epochs: Training epochs per Picard iteration.
        learning_rate: Adam learning rate.
        hidden_dims: MLP hidden layer sizes (default: [256, 256]).
        threshold_H: Collateral threshold.
        seed: Random seed.
        device: Torch device ('cpu' or 'cuda').

    Returns:
        dict with keys: 'XVA_0', 'result', 'meta', 'problem_data'
    """
    if hidden_dims is None:
        hidden_dims = [256, 256]

    # --- 1. Build market data and simulate ---
    problem, meta = example_build_problem(
        d=d,
        T_max=T_max,
        dt=dt,
        n_paths=n_paths,
        seed=seed,
        threshold_H=threshold_H,
    )
    data = problem.data

    # Extract sigma vector (constant vols from example)
    sigma_vec = np.full(d, 0.20)

    # --- 2. Create equation ---
    equation = XVACapletEquation(
        dim_x=d,
        F0=data.X[0, 0, :],  # initial forward curve from simulated paths
        sigma_vec=sigma_vec,
        Vbar=data.Vbar,
        C=data.C,
        r_array=data.r,
        lam_array=data.lam,
        recovery=meta["recovery_C"],
        spread_borrow=meta["funding_spread_b"],
        spread_lend=meta["funding_spread_l"],
        T=T_max,
        device=device,
    )

    # --- 3. Configure solver ---
    N = data.X.shape[1] - 1
    solver_cfg = SolverConfig(
        T=T_max,
        N=N,
        num_paths=n_paths,
        picard_iterations=picard_iterations,
        z_method="gradient",
        device=device,
    )
    train_cfg = TrainingConfig(
        batch_size=512,
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=True,
    )

    # --- 4. Create networks ---
    input_dim = 1 + d  # (t, F_1, ..., F_d)
    nn_Y = MLP(input_dim=input_dim, output_dim=1, hidden_dims=hidden_dims)

    # --- 5. Create solver and inject paths ---
    solver = UncoupledFBSDESolver(equation, solver_cfg, train_cfg, nn_Y)
    solver.set_external_paths(data.X, data.dW)

    # --- 6. Solve ---
    result = solver.solve()

    # Extract XVA_0 (mean across paths at t=0)
    xva_0 = float(result["Y"][:, 0, 0].mean())

    return {
        "XVA_0": xva_0,
        "result": result,
        "meta": meta,
        "problem_data": data,
    }


if __name__ == "__main__":
    print("Solving XVA BSDE with d=10, T=5y, monthly steps...")
    out = solve_xva(d=10, T_max=5.0, dt=1.0 / 12.0, n_paths=2000, picard_iterations=5)
    print(f"XVA_0 = {out['XVA_0']:.4f}")
