# deep_rfq_control_train.py
# Learn time-dependent bid/ask spreads (delta_a_t, delta_b_t) via a deep stochastic control graph.
# Objective: maximize E[P&L_T - (gamma/2) <P&L>_T], trained end-to-end through dynamics.
# Han & E (2016): controls-as-networks, stacked by time; the control objective is the NN loss.  (see paper)
# -----------------------------------------------------------------------------------------------

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Configuration / Environment
# ----------------------------

@dataclass
class RFQConfig:
    T: int = 60                 # number of time steps
    dt: float = 1.0/60.0        # time step (e.g., 1 minute horizon -> 60 steps of 1/60 day)
    batch_size: int = 1024
    device: str = "cpu"

    # Market / flow parameters (single-state)
    sigma: float = 0.02         # daily vol of reference price (in price units)
    lambda_a: float = 0.2       # ask RFQ arrival intensity per time unit
    lambda_b: float = 0.2       # bid RFQ arrival intensity per time unit
    kappa: float = 0.0          # price-drift sensitivity to flow imbalance
    z: float = 1.0              # trade size per fill (units of inventory)
    S0: float = 100.0           # initial reference price
    q0: float = 0.0             # initial inventory

    # S-curve parameters: p_fill = 1 / (1 + exp( k * (delta - m) ))
    scurve_k: float = 1.0       # slope (>0); larger => steeper decline with delta
    scurve_m: float = 0.0       # midpoint shift (in price units)

    # Risk and penalties
    gamma: float = 5.0          # risk aversion on quadratic variation of P&L
    phi_terminal_abs_q: float = 0.0   # optional |q_T| penalty (0 to disable)

    # Spread output constraints
    delta_min: float = 0.0
    delta_max: float = 2.0

    # Training
    epochs: int = 5000
    lr: float = 3e-4
    weight_decay: float = 0.0
    log_every: int = 200

    # Expected-flow vs stochastic fills
    MODE_EXPECTED_FLOW: bool = True
    seed: int = 42

# Set reproducibility
def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------------
# Utilities
# ----------------------------

def logistic_scurve(delta: torch.Tensor, k: float, m: float) -> torch.Tensor:
    """
    Monotone decreasing fill probability vs spread.
    p = 1 / (1 + exp(k * (delta - m)))  with k>0
    """
    return torch.sigmoid(-(k * (delta - m)))

def project_spreads(delta_a: torch.Tensor, delta_b: torch.Tensor, cfg: RFQConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    # Hard clipping (also possible to use softplus to ensure positivity then clip the top)
    da = torch.clamp(delta_a, cfg.delta_min, cfg.delta_max)
    db = torch.clamp(delta_b, cfg.delta_min, cfg.delta_max)
    return da, db

# ----------------------------
# Policy: Time-stacked heads
# ----------------------------

class TimeHead(nn.Module):
    """
    A small MLP that maps state -> (delta_a, delta_b) for a specific time step.
    State here is minimal: (q, tau), optionally you can expand with context later.
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),  # [q, tau]
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)   # outputs: [u_a, u_b] unconstrained
        )

    def forward(self, q: torch.Tensor, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, tau shapes: [B]
        x = torch.stack([q, tau], dim=-1)
        u = self.net(x)                    # [B,2]
        # map to positive spreads using softplus, then we'll clip to [delta_min, delta_max]
        delta = F.softplus(u)              # strictly >=0
        delta_a, delta_b = delta[..., 0], delta[..., 1]
        return delta_a, delta_b

class PolicyStack(nn.Module):
    """
    A stack of T TimeHeads: one head per time step.
    """
    def __init__(self, T: int, hidden=64):
        super().__init__()
        self.T = T
        self.heads = nn.ModuleList([TimeHead(hidden=hidden) for _ in range(T)])

    def forward(self, t: int, q: torch.Tensor, tau: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce (delta_a, delta_b) at time index t for batch of states.
        """
        return self.heads[t](q, tau)

    def save_per_time(self, path_dir: str):
        os.makedirs(path_dir, exist_ok=True)
        for t, head in enumerate(self.heads):
            torch.save(head.state_dict(), os.path.join(path_dir, f"policy_t{t:03d}.pt"))
        # also save whole stack
        torch.save(self.state_dict(), os.path.join(path_dir, "policy_stack.pt"))

# ----------------------------
# Differentiable RFQ Simulator
# ----------------------------

class RFQSimulator(nn.Module):
    """
    Batch simulator with differentiable expected-flow fills.
    Implements P&L bookkeeping and quadratic-variation penalty exactly as in FTP-style objective:
       Loss = -E[ PnL_T - (gamma/2) * <PnL>_T ]
    where <PnL>_T = sum q_k^2 * sigma^2 * dt   (discrete version of quadratic variation).
    """
    def __init__(self, cfg: RFQConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("sigma", torch.tensor(cfg.sigma, dtype=torch.float32))
        self.register_buffer("lambda_a", torch.tensor(cfg.lambda_a, dtype=torch.float32))
        self.register_buffer("lambda_b", torch.tensor(cfg.lambda_b, dtype=torch.float32))
        self.register_buffer("mu", torch.tensor(cfg.kappa * (cfg.lambda_a - cfg.lambda_b), dtype=torch.float32))

    def step_expected_flow(
        self,
        S: torch.Tensor,
        q: torch.Tensor,
        delta_a: torch.Tensor,
        delta_b: torch.Tensor,
        eps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One step with expected-flow relaxation (fully differentiable).
        Returns: (S_next, q_next, cash_inc, dS, risk_increment)
        """
        cfg = self.cfg
        # Price step: dS = mu*dt + sigma*sqrt(dt)*eps
        dS = self.mu * cfg.dt + self.sigma * math.sqrt(cfg.dt) * eps
        S_next = S + dS

        # Fill probabilities from S-curves
        p_a = logistic_scurve(delta_a, cfg.scurve_k, cfg.scurve_m)  # in (0,1)
        p_b = logistic_scurve(delta_b, cfg.scurve_k, cfg.scurve_m)

        # Expected number of fills in [t, t+dt): lambda * dt * p
        exp_fills_a = self.lambda_a * cfg.dt * p_a
        exp_fills_b = self.lambda_b * cfg.dt * p_b
        # Expected filled volume (units)
        vol_a = cfg.z * exp_fills_a
        vol_b = cfg.z * exp_fills_b

        # Cash increment from trades:
        cash_inc = (S + delta_a) * vol_a - (S - delta_b) * vol_b

        # Inventory update (bid adds inventory, ask reduces):
        q_next = q + (vol_b - vol_a)

        # Quadratic-variation increment (discrete): q^2 * sigma^2 * dt
        risk_inc = (q ** 2) * (self.sigma ** 2) * cfg.dt

        return S_next, q_next, cash_inc, dS, risk_inc

    def step_sampled(
        self,
        S: torch.Tensor,
        q: torch.Tensor,
        delta_a: torch.Tensor,
        delta_b: torch.Tensor,
        eps: torch.Tensor,
        rng: torch.Generator
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optional stochastic fills (non-differentiable w.r.t. action through sampling).
        Useful for evaluation/MC; training is recommended with expected-flow.
        """
        cfg = self.cfg
        dS = self.mu * cfg.dt + self.sigma * math.sqrt(cfg.dt) * eps
        S_next = S + dS

        p_a = logistic_scurve(delta_a, cfg.scurve_k, cfg.scurve_m)
        p_b = logistic_scurve(delta_b, cfg.scurve_k, cfg.scurve_m)

        # Arrival probabilities in dt (Poisson thinning)
        p_arr_a = 1.0 - torch.exp(-self.lambda_a * cfg.dt)
        p_arr_b = 1.0 - torch.exp(-self.lambda_b * cfg.dt)
        p_fill_a = torch.clamp(p_arr_a * p_a, 0.0, 1.0)
        p_fill_b = torch.clamp(p_arr_b * p_b, 0.0, 1.0)

        # Sample fills
        u_a = torch.rand_like(p_fill_a, generator=rng)
        u_b = torch.rand_like(p_fill_b, generator=rng)
        fill_a = (u_a < p_fill_a).float()
        fill_b = (u_b < p_fill_b).float()

        vol_a = cfg.z * fill_a
        vol_b = cfg.z * fill_b

        cash_inc = (S + delta_a) * vol_a - (S - delta_b) * vol_b
        q_next = q + (vol_b - vol_a)
        risk_inc = (q ** 2) * (self.sigma ** 2) * cfg.dt

        return S_next, q_next, cash_inc, dS, risk_inc

    def simulate_batch(
        self,
        policy: PolicyStack,
        batch_size: int,
        expected_flow: bool = True,
        return_traces: bool = False,
        rng: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate a batch of trajectories and compute:
           - terminal P&L
           - quadratic-variation penalty
           - loss = -E[P&L_T - (gamma/2) <P&L>_T]
        """
        cfg = self.cfg
        device = cfg.device

        # Initialize batch state
        S = torch.full((batch_size,), cfg.S0, device=device)
        q = torch.full((batch_size,), cfg.q0, device=device)
        X = torch.zeros(batch_size, device=device)  # cash
        # For value function we use P&L = X + q*S, but bookkeeping is clearer via increments

        eps_all = torch.randn(cfg.T, batch_size, device=device)

        risk_cum = torch.zeros(batch_size, device=device)

        # Optional traces
        traces = {
            "delta_a": [],
            "delta_b": [],
            "q": [],
            "S": [],
            "cash_inc": [],
            "dS": []
        } if return_traces else None

        for t in range(cfg.T):
            tau = torch.full((batch_size,), float(cfg.T - t), device=device)  # time-to-go in steps
            delta_a, delta_b = policy(t, q, tau)
            delta_a, delta_b = project_spreads(delta_a, delta_b, cfg)

            if expected_flow:
                S_next, q_next, cash_inc, dS, risk_inc = self.step_expected_flow(
                    S, q, delta_a, delta_b, eps_all[t]
                )
            else:
                if rng is None:
                    rng = torch.Generator(device=device)
                    rng.manual_seed(cfg.seed + 123)
                S_next, q_next, cash_inc, dS, risk_inc = self.step_sampled(
                    S, q, delta_a, delta_b, eps_all[t], rng
                )

            X = X + cash_inc
            risk_cum = risk_cum + risk_inc

            if return_traces:
                traces["delta_a"].append(delta_a.detach())
                traces["delta_b"].append(delta_b.detach())
                traces["q"].append(q.detach())
                traces["S"].append(S.detach())
                traces["cash_inc"].append(cash_inc.detach())
                traces["dS"].append(dS.detach())

            S, q = S_next, q_next

        # Terminal marked-to-market P&L
        PnL_T = X + q * S

        # Optional terminal inventory penalty (L1)
        if cfg.phi_terminal_abs_q > 0:
            PnL_T = PnL_T - cfg.phi_terminal_abs_q * torch.abs(q)

        # Quadratic variation penalty term
        qv_penalty = (cfg.gamma / 2.0) * risk_cum

        # Objective and loss
        objective = PnL_T - qv_penalty
        loss = -objective.mean()

        out = {
            "loss": loss,
            "objective_mean": objective.mean().detach(),
            "PnL_mean": PnL_T.mean().detach(),
            "risk_mean": risk_cum.mean().detach(),
            "final_q_abs_mean": torch.abs(q).mean().detach(),
        }
        if return_traces:
            # stack traces as [T, B]
            for k in traces:
                traces[k] = torch.stack(traces[k], dim=0)
            out["traces"] = traces
        return out

# ----------------------------
# Value function estimator
# ----------------------------

class ValueEstimator:
    """
    Monte Carlo estimator of V(t0, state) under trained policy:
       V = E[ P&L_T - (gamma/2) <P&L>_T | s_{t0} ]
    Allows starting mid-horizon (t0).
    """
    def __init__(self, cfg: RFQConfig):
        self.cfg = cfg
        self.sim = RFQSimulator(cfg).to(cfg.device)

    @torch.no_grad()
    def estimate(
        self,
        policy: PolicyStack,
        t0: int,
        q0: float,
        S0: float,
        n_paths: int = 10000,
        expected_flow_eval: bool = True
    ) -> float:
        cfg = self.cfg
        device = cfg.device

        # Temporarily adjust horizon by shifting T and keep dt fixed
        T_saved = cfg.T
        cfg.T = max(0, T_saved - t0)

        # Build a wrapper that seeds the simulator at given (q0, S0)
        # We reuse simulate_batch but override initial conditions by temporarily hacking the config.
        sim = RFQSimulator(cfg).to(device)
        # Monkey patch initial conditions
        cfg_backup = (cfg.S0, cfg.q0)
        cfg.S0, cfg.q0 = S0, q0

        out = sim.simulate_batch(policy, n_paths, expected_flow=expected_flow_eval)
        val = (out["PnL_mean"] - (cfg.gamma/2.0) * out["risk_mean"] * 0.0 + out["objective_mean"] - out["objective_mean"] + out["objective_mean"]).item()
        # The above expression simply returns objective_mean; it's left verbose to make it explicit.

        # Restore config
        cfg.S0, cfg.q0 = cfg_backup
        cfg.T = T_saved

        return val

# ----------------------------
# Training Loop
# ----------------------------

def train(cfg: RFQConfig) -> Tuple[PolicyStack, Dict[str, float]]:
    set_seed(cfg.seed)
    device = cfg.device

    policy = PolicyStack(T=cfg.T, hidden=128).to(device)
    sim = RFQSimulator(cfg).to(device)

    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Learning rate schedule (optional)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    for epoch in range(1, cfg.epochs + 1):
        outs = sim.simulate_batch(
            policy=policy,
            batch_size=cfg.batch_size,
            expected_flow=cfg.MODE_EXPECTED_FLOW,
            return_traces=False
        )
        loss = outs["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
        scheduler.step()

        if epoch % cfg.log_every == 0 or epoch == 1 or epoch == cfg.epochs:
            print(f"[{epoch:5d}] "
                  f"loss={outs['loss']:.5f}  "
                  f"obj={outs['objective_mean']:.5f}  "
                  f"PnL={outs['PnL_mean']:.5f}  "
                  f"risk={outs['risk_mean']:.5f}  "
                  f"|q_T|={outs['final_q_abs_mean']:.5f}")

    return policy, {"final_loss": outs["loss"].item(), "final_objective": outs["objective_mean"].item()}

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    cfg = RFQConfig(
        T=60,
        dt=1.0/60.0,
        batch_size=2048,
        device="cuda" if torch.cuda.is_available() else "cpu",
        sigma=0.02,
        lambda_a=0.25,
        lambda_b=0.20,
        kappa=0.10,          # enables flow-driven drift mu = kappa*(lambda_a - lambda_b)
        z=1.0,
        S0=100.0,
        q0=0.0,
        scurve_k=1.5,
        scurve_m=0.0,
        gamma=5.0,
        phi_terminal_abs_q=0.0,   # set >0 if you want terminal flatness penalty
        delta_min=0.0,
        delta_max=2.0,
        epochs=4000,
        lr=3e-4,
        log_every=200,
        MODE_EXPECTED_FLOW=True,
        seed=123
    )

    policy, stats = train(cfg)

    # Save trained heads (one file per time) and full stack
    save_dir = "./trained_policies"
    policy.save_per_time(save_dir)
    print(f"Saved per-time heads and full stack to: {save_dir}")

    # Example: evaluate value function mid-horizon at t0=30 with custom state
    val_est = ValueEstimator(cfg)
    V_mid = val_est.estimate(policy, t0=30, q0=0.0, S0=100.0, n_paths=5000, expected_flow_eval=True)
    print(f"Estimated value at t0=30, q0=0, S0=100: {V_mid:.5f}")
