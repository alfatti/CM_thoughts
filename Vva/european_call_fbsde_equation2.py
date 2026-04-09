"""
European Call Option FBSDE.

Characterizes the FBSDE system corresponding to the Black-Scholes pricing 
of a European call option via the Feynman-Kac representation. The backward
component Y_t recovers the option price and Z_t recovers the delta-hedge.

System Dynamics (risk-neutral measure):
    Forward:  dX_t = r * X_t dt + sigma * X_t dW_t,   X_0 = S0
    Backward: -dY_t = f(t, X_t, Y_t, Z_t) dt - Z_t dW_t
    Terminal: Y_T = max(X_T - K, 0)

Driver (from Feynman-Kac under risk-neutral measure Q):
    f(t, x, y, z) = -r * y

This is the same nonlinear structure as BSBEquation but adapted to:
  - A single underlying asset (dim_x = 1, dim_w = 1)
  - Risk-neutral GBM drift in the forward SDE
  - Standard call payoff at terminal time
  - Full Black-Scholes analytical solution (price + delta)

Analytical Solution:
    Y_t = V_BS(t, X_t)            (Black-Scholes call price)
    Z_t = sigma * X_t * Delta(t, X_t)   (scaled delta)

where:
    d1 = (log(X_t/K) + (r + sigma^2/2)(T-t)) / (sigma * sqrt(T-t))
    d2 = d1 - sigma * sqrt(T-t)
    V_BS = X_t * N(d1) - K * exp(-r*(T-t)) * N(d2)
    Delta = N(d1)
"""

import torch
import math
from typing import Optional, Union

from dim_fbsde.equations.base import FBSDE


class EuropeanCallEquation(FBSDE):
    """
    FBSDE characterization of a European call option under Black-Scholes.

    The Feynman-Kac theorem links the Black-Scholes PDE to this FBSDE:
        - Y_t is the option price V(t, X_t)
        - Z_t is sigma * X_t * Delta(t, X_t), the diffusion-scaled delta hedge

    Args:
        S0      : Initial asset price (scalar float).
        K       : Strike price.
        r       : Risk-free rate.
        sigma   : Volatility of the underlying asset.
        T       : Option maturity (time to expiry from t=0).
        device  : Torch device string or object.
        dtype   : Torch floating-point dtype.
    """

    def __init__(self,
                 S0: float = 1.0,
                 K: float = 1.0,
                 r: float = 0.05,
                 sigma: float = 0.2,
                 T: float = 1.0,
                 device: Union[str, torch.device] = "cpu",
                 dtype: torch.dtype = torch.float32):

        # European call is inherently 1-dimensional:
        #   dim_x = 1  (single asset price)
        #   dim_y = 1  (scalar option price)
        #   dim_w = 1  (single Brownian driver)
        super().__init__(
            dim_x=1,
            dim_y=1,
            dim_w=1,
            x0=torch.tensor([S0]),
            device=device,
            dtype=dtype,
        )

        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    # ------------------------------------------------------------------
    # Forward SDE coefficients
    # ------------------------------------------------------------------

    def drift(self, t: torch.Tensor, x: torch.Tensor,
              y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Risk-neutral GBM drift: mu(t, x) = r * X_t.

        Shape: [Batch, dim_x] = [Batch, 1]
        """
        return self.r * x

    def diffusion(self, t: torch.Tensor, x: torch.Tensor,
                  y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        GBM diffusion matrix: sigma(t, x) = sigma * diag(X_t).

        For dim_x = dim_w = 1 this is simply [[sigma * X_t]].
        Shape: [Batch, dim_x, dim_w] = [Batch, 1, 1]
        """
        return (self.sigma * x).unsqueeze(-1)   # [Batch, 1] -> [Batch, 1, 1]

    # ------------------------------------------------------------------
    # Backward SDE driver
    # ------------------------------------------------------------------

    def driver(self, t: torch.Tensor, x: torch.Tensor,
               y: torch.Tensor, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Black-Scholes driver under the risk-neutral measure Q:
            f(t, x, y, z) = -r * Y_t

        Under Q the Brownian motion already absorbs the market price of risk,
        so the drift of the discounted price process is zero. The backward SDE
        becomes simply:
            -dV̂_t = -r_t * V̂_t dt - Z_t dW_t^Q

        meaning the driver is pure discounting with no z-dependence. A
        (r/sigma)*z correction would only appear if the forward SDE were
        stated under the physical measure P and a Girsanov change of measure
        were required.

        returns : [Batch, dim_y] = [Batch, 1]
        """
        return -self.r * y

    # ------------------------------------------------------------------
    # Terminal condition
    # ------------------------------------------------------------------

    def terminal_condition(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        European call payoff: g(X_T) = max(X_T - K, 0).

        Shape: [Batch, dim_y] = [Batch, 1]
        """
        return torch.clamp(x - self.K, min=0.0)

    # ------------------------------------------------------------------
    # Analytical solutions (Black-Scholes closed form)
    # ------------------------------------------------------------------

    def analytical_y(self, t: torch.Tensor, x: torch.Tensor,
                     **kwargs) -> Optional[torch.Tensor]:
        """
        Black-Scholes call price V(t, X_t).

        V = X_t * N(d1) - K * exp(-r*(T-t)) * N(d2)

        d1 = [log(X_t/K) + (r + sigma^2/2)*(T-t)] / [sigma*sqrt(T-t)]
        d2 = d1 - sigma * sqrt(T-t)

        At expiry (T-t = 0) the formula reduces to the payoff directly.

        Returns: [Batch, 1]
        """
        T = kwargs.get('T_terminal', self.T)
        tau = T - t                                       # time to expiry, scalar or [Batch,1]

        # Clamp to avoid division by zero at terminal time
        tau_clamped = torch.clamp(tau, min=1e-8)

        log_moneyness = torch.log(x / self.K)             # [Batch, 1]
        sigma_sqrt_tau = self.sigma * torch.sqrt(tau_clamped)

        d1 = (log_moneyness + (self.r + 0.5 * self.sigma**2) * tau_clamped) / sigma_sqrt_tau
        d2 = d1 - sigma_sqrt_tau

        # Standard normal CDF via torch
        N_d1 = self._standard_normal_cdf(d1)
        N_d2 = self._standard_normal_cdf(d2)

        price = x * N_d1 - self.K * torch.exp(-self.r * tau_clamped) * N_d2

        # At tau=0 return the payoff directly
        if isinstance(tau, torch.Tensor):
            at_expiry = (tau <= 1e-8)
            if at_expiry.any():
                payoff = torch.clamp(x - self.K, min=0.0)
                price = torch.where(at_expiry, payoff, price)

        return price   # [Batch, 1]

    def analytical_z(self, t: torch.Tensor, x: torch.Tensor,
                     **kwargs) -> Optional[torch.Tensor]:
        """
        Diffusion-scaled delta: Z_t = sigma * X_t * Delta(t, X_t).

        Delta = dV/dX = N(d1)   (Black-Scholes delta for a call)
        Z_t   = sigma * X_t * N(d1)

        Returns: [Batch, dim_y, dim_w] = [Batch, 1, 1]
        """
        T = kwargs.get('T_terminal', self.T)
        tau = T - t
        tau_clamped = torch.clamp(tau, min=1e-8)

        log_moneyness = torch.log(x / self.K)
        sigma_sqrt_tau = self.sigma * torch.sqrt(tau_clamped)

        d1 = (log_moneyness + (self.r + 0.5 * self.sigma**2) * tau_clamped) / sigma_sqrt_tau

        delta = self._standard_normal_cdf(d1)             # [Batch, 1]
        z = self.sigma * x * delta                        # [Batch, 1]

        return z.unsqueeze(1)    # [Batch, 1, 1]

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable standard normal CDF using the error function:
            N(x) = 0.5 * (1 + erf(x / sqrt(2)))
        """
        return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
