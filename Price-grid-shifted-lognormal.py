\section*{Shifted Log-Normal Model: Simulation \& Pricing}

\subsection*{Model Specification}

Under the risk-neutral measure $\mathbb{Q}$, the underlying asset follows a shifted log-normal process:
\begin{equation}
dS_t = (r - q)(S_t + \alpha)\,dt + \sigma(S_t + \alpha)\,dW_t^{\mathbb{Q}}
\end{equation}
where $S_t$ is the spot price, $r$ is the risk-free rate, $q$ is the dividend yield, $\sigma$ is the volatility, $\alpha$ is the shift parameter, and $W_t^{\mathbb{Q}}$ is a standard Brownian motion under $\mathbb{Q}$.

\subsection*{Analytical Solution}

The shifted process admits an exact solution. Define $Y_t = S_t + \alpha$, then:
\begin{equation}
Y_t = Y_0 \exp\left[\left(r - q - \frac{1}{2}\sigma^2\right)t + \sigma W_t\right]
\end{equation}

Thus, the spot price at time $t$ is:
\begin{equation}
S_t = (S_0 + \alpha)\exp\left[\left(r - q - \frac{1}{2}\sigma^2\right)t + \sigma W_t\right] - \alpha
\end{equation}

\subsection*{European Call Option Pricing}

The price of a European call option with strike $K$ and maturity $T$ is given by the shifted Black-Scholes formula:
\begin{equation}
C(S_0, K, T) = (S_0 + \alpha)e^{-qT}\Phi(d_1) - (K + \alpha)e^{-rT}\Phi(d_2)
\end{equation}
where:
\begin{align}
d_1 &= \frac{\ln\left(\frac{(S_0 + \alpha)e^{(r-q)T}}{K + \alpha}\right) + \frac{1}{2}\sigma^2 T}{\sigma\sqrt{T}} \\
d_2 &= d_1 - \sigma\sqrt{T}
\end{align}
and $\Phi(\cdot)$ is the cumulative distribution function of the standard normal distribution.

\subsection*{Greeks}

The option sensitivities (Greeks) are modified by the shift parameter:

\begin{align}
\Delta &= e^{-qT}\Phi(d_1) \\
\Gamma &= \frac{e^{-qT}\phi(d_1)}{(S_0 + \alpha)\sigma\sqrt{T}} \\
\mathcal{V} &= (S_0 + \alpha)e^{-qT}\phi(d_1)\sqrt{T} \\
\Theta &= -\frac{(S_0 + \alpha)e^{-qT}\phi(d_1)\sigma}{2\sqrt{T}} - r(K + \alpha)e^{-rT}\Phi(d_2) + q(S_0 + \alpha)e^{-qT}\Phi(d_1) \\
\rho &= (K + \alpha)Te^{-rT}\Phi(d_2)
\end{align}
where $\phi(\cdot)$ is the probability density function of the standard normal distribution.

\subsection*{Implementation Details}

\subsubsection*{Path Simulation}
The code simulates $N$ paths over $M$ time steps using the exact solution:
\begin{equation}
S_{t_{i+1}} = (S_{t_i} + \alpha)\exp\left[\left(r - q - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\,Z_i\right] - \alpha
\end{equation}
where $Z_i \sim \mathcal{N}(0,1)$ are i.i.d. standard normal random variables and $\Delta t = T/M$.

\subsubsection*{Monte Carlo Pricing}
The call option price is estimated via:
\begin{equation}
\hat{C}_{\text{MC}} = e^{-rT}\frac{1}{N}\sum_{i=1}^{N}\max(S_T^{(i)} - K, 0)
\end{equation}
where $S_T^{(i)}$ is the terminal price of the $i$-th simulated path.

\subsubsection*{Parameter Grid}
The function generates a full factorial grid over seven parameters:
\begin{equation}
\mathcal{P} = \{S_0, T, r, q, K, \sigma, \alpha\} \in \mathbb{R}^7
\end{equation}
For each parameter $p \in \mathcal{P}$, we specify $(p_{\min}, p_{\max}, n_p)$ to create $n_p$ linearly spaced values. The total number of configurations is $\prod_{p \in \mathcal{P}} n_p$.

\subsection*{Validation \& Error Analysis}

The code computes the pricing error between Monte Carlo and analytical prices:
\begin{equation}
\epsilon = \hat{C}_{\text{MC}} - C_{\text{analytical}}
\end{equation}

Key error metrics reported:
\begin{itemize}
    \item Mean error: $\bar{\epsilon} = \frac{1}{N_{\text{config}}}\sum_{i=1}^{N_{\text{config}}} \epsilon_i$
    \item Root mean squared error: $\text{RMSE} = \sqrt{\frac{1}{N_{\text{config}}}\sum_{i=1}^{N_{\text{config}}} \epsilon_i^2}$
    \item Maximum absolute error: $\max_i |\epsilon_i|$
\end{itemize}

\subsection*{Constraints}

The model requires the following constraints to ensure well-posedness:
\begin{enumerate}
    \item $S_0 + \alpha > 0$ (positivity of shifted spot)
    \item $K + \alpha > 0$ (positivity of shifted strike)
    \item $\sigma > 0$ (positive volatility)
    \item $T > 0$ (positive time to maturity)
\end{enumerate}

\subsection*{Special Cases}

\begin{itemize}
    \item \textbf{Standard Black-Scholes:} $\alpha = 0$
    \item \textbf{Bachelier (Normal) Model:} $\alpha \to \infty$ with $\sigma(S_0 + \alpha) = \sigma_{\text{normal}}$ held constant
    \item \textbf{At-the-Money:} $S_0 = K \implies d_1 = \frac{(r-q)T + \frac{1}{2}\sigma^2T}{\sigma\sqrt{T}}$
\end{itemize}






import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple
import matplotlib.pyplot as plt

def simulate_shifted_lognormal_and_price(
    S0_range: Tuple[float, float, int],
    T_range: Tuple[float, float, int],
    r_range: Tuple[float, float, int],
    q_range: Tuple[float, float, int],
    K_range: Tuple[float, float, int],
    sigma_range: Tuple[float, float, int],
    alpha_range: Tuple[float, float, int],
    n_steps: int = 252,
    n_paths_per_config: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Simulate paths under shifted log-normal model and price European calls.
    
    Parameters:
    -----------
    S0_range : (min, max, n_points) for spot price
    T_range : (min, max, n_points) for maturity in years
    r_range : (min, max, n_points) for risk-free rate
    q_range : (min, max, n_points) for dividend yield
    K_range : (min, max, n_points) for strike price
    sigma_range : (min, max, n_points) for volatility
    alpha_range : (min, max, n_points) for shift parameter
    n_steps : number of time steps per path
    n_paths_per_config : number of Monte Carlo paths per parameter configuration
    seed : random seed
    
    Returns:
    --------
    Dictionary containing:
        - 'parameters': array of parameter combinations
        - 'simulated_paths': simulated price paths
        - 'terminal_prices': terminal prices from each path
        - 'call_prices_mc': call prices from Monte Carlo
        - 'call_prices_analytical': analytical call prices
        - 'pricing_errors': difference between MC and analytical
    """
    
    np.random.seed(seed)
    
    # Generate parameter grids
    S0_vals = np.linspace(*S0_range)
    T_vals = np.linspace(*T_range)
    r_vals = np.linspace(*r_range)
    q_vals = np.linspace(*q_range)
    K_vals = np.linspace(*K_range)
    sigma_vals = np.linspace(*sigma_range)
    alpha_vals = np.linspace(*alpha_range)
    
    # Create parameter combinations (using meshgrid for full factorial)
    # For computational efficiency, you might want to use random sampling instead
    param_grid = np.array(np.meshgrid(
        S0_vals, T_vals, r_vals, q_vals, K_vals, sigma_vals, alpha_vals
    )).T.reshape(-1, 7)
    
    n_configs = param_grid.shape[0]
    
    print(f"Total parameter configurations: {n_configs}")
    print(f"Total paths to simulate: {n_configs * n_paths_per_config:,}")
    
    # Storage arrays
    all_paths = []
    all_terminal_prices = []
    call_prices_mc = np.zeros(n_configs)
    call_prices_analytical = np.zeros(n_configs)
    
    # Process each configuration
    for i, (S0, T, r, q, K, sigma, alpha) in enumerate(param_grid):
        
        # Validation
        if S0 + alpha <= 0:
            print(f"Warning: Config {i} - S0 + alpha <= 0, skipping")
            continue
        if K + alpha <= 0:
            print(f"Warning: Config {i} - K + alpha <= 0, skipping")
            continue
        if T <= 0 or sigma <= 0:
            print(f"Warning: Config {i} - Invalid T or sigma, skipping")
            continue
            
        # Simulate paths under risk-neutral measure
        # dS_t = (r - q)(S_t + alpha)dt + sigma(S_t + alpha)dW_t
        # Equivalently: d(S_t + alpha) = (r - q)(S_t + alpha)dt + sigma(S_t + alpha)dW_t
        
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        # Initialize paths
        paths = np.zeros((n_paths_per_config, n_steps + 1))
        paths[:, 0] = S0
        
        # Simulate using exact solution for shifted process
        # (S_t + alpha) = (S_0 + alpha) * exp((r - q - 0.5*sigma^2)*t + sigma*W_t)
        for step in range(1, n_steps + 1):
            t = step * dt
            Z = np.random.standard_normal(n_paths_per_config)
            
            # Exact solution at time t
            shifted_S = (S0 + alpha) * np.exp(
                (r - q - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * Z
            )
            paths[:, step] = shifted_S - alpha
        
        # Terminal prices
        S_T = paths[:, -1]
        
        # Monte Carlo call price
        payoffs = np.maximum(S_T - K, 0)
        call_mc = np.exp(-r * T) * np.mean(payoffs)
        
        # Analytical call price (shifted Black-Scholes)
        call_analytical = shifted_bs_call(S0, K, T, r, q, sigma, alpha)
        
        # Store results
        all_paths.append(paths)
        all_terminal_prices.append(S_T)
        call_prices_mc[i] = call_mc
        call_prices_analytical[i] = call_analytical
        
        if (i + 1) % max(1, n_configs // 10) == 0:
            print(f"Progress: {i+1}/{n_configs} configurations processed")
    
    results = {
        'parameters': param_grid,
        'parameter_names': ['S0', 'T', 'r', 'q', 'K', 'sigma', 'alpha'],
        'simulated_paths': all_paths,
        'terminal_prices': all_terminal_prices,
        'call_prices_mc': call_prices_mc,
        'call_prices_analytical': call_prices_analytical,
        'pricing_errors': call_prices_mc - call_prices_analytical,
        'n_steps': n_steps,
        'n_paths_per_config': n_paths_per_config
    }
    
    return results


def shifted_bs_call(S0: float, K: float, T: float, r: float, q: float, 
                   sigma: float, alpha: float) -> float:
    """
    Analytical European call price under shifted log-normal model.
    
    Parameters:
    -----------
    S0 : spot price
    K : strike price
    T : time to maturity
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    alpha : shift parameter
    
    Returns:
    --------
    Call option price
    """
    
    if T <= 0:
        return max(S0 - K, 0)
    
    if S0 + alpha <= 0 or K + alpha <= 0:
        raise ValueError("S0 + alpha and K + alpha must be positive")
    
    # Shifted moneyness
    shifted_forward = (S0 + alpha) * np.exp((r - q) * T)
    shifted_strike = K + alpha
    
    # d1 and d2
    d1 = (np.log(shifted_forward / shifted_strike) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Call price
    call_price = (
        shifted_forward * np.exp(-r * T) * norm.cdf(d1) - 
        shifted_strike * np.exp(-r * T) * norm.cdf(d2)
    )
    
    return call_price


def shifted_bs_greeks(S0: float, K: float, T: float, r: float, q: float,
                     sigma: float, alpha: float) -> Dict[str, float]:
    """
    Calculate Greeks for European call under shifted log-normal model.
    """
    
    if T <= 0:
        return {'delta': 1.0 if S0 > K else 0.0, 'gamma': 0.0, 'vega': 0.0, 
                'theta': 0.0, 'rho': 0.0}
    
    shifted_forward = (S0 + alpha) * np.exp((r - q) * T)
    shifted_strike = K + alpha
    
    sqrt_T = np.sqrt(T)
    d1 = (np.log(shifted_forward / shifted_strike) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Greeks
    delta = np.exp(-q * T) * norm.cdf(d1)
    
    gamma = np.exp(-q * T) * norm.pdf(d1) / ((S0 + alpha) * sigma * sqrt_T)
    
    vega = (S0 + alpha) * np.exp(-q * T) * norm.pdf(d1) * sqrt_T / 100  # per 1% vol
    
    theta = (
        -(S0 + alpha) * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * sqrt_T)
        - r * shifted_strike * np.exp(-r * T) * norm.cdf(d2)
        + q * (S0 + alpha) * np.exp(-q * T) * norm.cdf(d1)
    ) / 365  # per day
    
    rho = shifted_strike * T * np.exp(-r * T) * norm.cdf(d2) / 100  # per 1% rate
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }


# Example usage
if __name__ == "__main__":
    
    # Define parameter ranges (min, max, n_points)
    # Using smaller ranges for demonstration
    results = simulate_shifted_lognormal_and_price(
        S0_range=(95, 105, 3),      # Spot: 95 to 105
        T_range=(0.25, 1.0, 3),     # Maturity: 3M to 1Y
        r_range=(0.03, 0.05, 2),    # Rate: 3% to 5%
        q_range=(0.0, 0.02, 2),     # Dividend: 0% to 2%
        K_range=(95, 105, 3),       # Strike: 95 to 105
        sigma_range=(0.15, 0.30, 3), # Vol: 15% to 30%
        alpha_range=(-5, 5, 3),     # Shift: -5 to 5
        n_steps=252,                # Daily steps
        n_paths_per_config=10000,   # 10k paths per config
        seed=42
    )
    
    # Summary statistics
    print("\n" + "="*60)
    print("SIMULATION RESULTS SUMMARY")
    print("="*60)
    print(f"Total configurations: {len(results['call_prices_mc'])}")
    print(f"\nAnalytical Call Prices:")
    print(f"  Mean: {np.mean(results['call_prices_analytical']):.4f}")
    print(f"  Std:  {np.std(results['call_prices_analytical']):.4f}")
    print(f"  Min:  {np.min(results['call_prices_analytical']):.4f}")
    print(f"  Max:  {np.max(results['call_prices_analytical']):.4f}")
    
    print(f"\nMonte Carlo Call Prices:")
    print(f"  Mean: {np.mean(results['call_prices_mc']):.4f}")
    print(f"  Std:  {np.std(results['call_prices_mc']):.4f}")
    
    print(f"\nPricing Errors (MC - Analytical):")
    print(f"  Mean:     {np.mean(results['pricing_errors']):.6f}")
    print(f"  Std:      {np.std(results['pricing_errors']):.6f}")
    print(f"  RMSE:     {np.sqrt(np.mean(results['pricing_errors']**2)):.6f}")
    print(f"  Max Abs:  {np.max(np.abs(results['pricing_errors'])):.6f}")
    
    # Example: Plot a single path
    idx = 0  # First configuration
    params = results['parameters'][idx]
    path = results['simulated_paths'][idx]
    
    print(f"\nExample Configuration {idx}:")
    for name, val in zip(results['parameter_names'], params):
        print(f"  {name}: {val:.4f}")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(path[:10].T, alpha=0.3)  # Plot first 10 paths
    plt.axhline(y=params[4], color='r', linestyle='--', label=f'Strike = {params[4]:.2f}')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.title('Sample Paths (First 10)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(results['terminal_prices'][idx], bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=params[4], color='r', linestyle='--', label=f'Strike = {params[4]:.2f}')
    plt.xlabel('Terminal Price')
    plt.ylabel('Frequency')
    plt.title('Terminal Price Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('shifted_lognormal_simulation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved as 'shifted_lognormal_simulation.png'")
    
    # Test Greeks calculation
    print("\n" + "="*60)
    print("GREEKS CALCULATION (First Configuration)")
    print("="*60)
    greeks = shifted_bs_greeks(*params)
    for greek, value in greeks.items():
        print(f"  {greek.capitalize():8s}: {value:.6f}")

