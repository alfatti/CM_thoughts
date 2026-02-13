import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict

def compute_analytical_call_prices(
    S0_range: Tuple[float, float, int],
    T_range: Tuple[float, float, int],
    r_range: Tuple[float, float, int],
    q_range: Tuple[float, float, int],
    K_range: Tuple[float, float, int],
    sigma_range: Tuple[float, float, int],
    alpha_range: Tuple[float, float, int],
    return_format: str = 'array'
) -> Dict:
    """
    Compute analytical European call prices under shifted log-normal model
    for a grid of parameter combinations.
    
    Parameters:
    -----------
    S0_range : (min, max, n_points) for spot price
    T_range : (min, max, n_points) for maturity in years
    r_range : (min, max, n_points) for risk-free rate
    q_range : (min, max, n_points) for dividend yield
    K_range : (min, max, n_points) for strike price
    sigma_range : (min, max, n_points) for volatility
    alpha_range : (min, max, n_points) for shift parameter
    return_format : 'array' for flat array or 'grid' for 7D array
    
    Returns:
    --------
    Dictionary containing:
        - 'call_prices': call option prices
        - 'parameters': parameter combinations
        - 'parameter_names': names of parameters
        - 'valid_mask': boolean mask for valid configurations
        - 'shape': shape of output (if return_format='grid')
    """
    
    # Generate parameter grids
    S0_vals = np.linspace(*S0_range)
    T_vals = np.linspace(*T_range)
    r_vals = np.linspace(*r_range)
    q_vals = np.linspace(*q_range)
    K_vals = np.linspace(*K_range)
    sigma_vals = np.linspace(*sigma_range)
    alpha_vals = np.linspace(*alpha_range)
    
    # Create full parameter grid
    param_grid = np.array(np.meshgrid(
        S0_vals, T_vals, r_vals, q_vals, K_vals, sigma_vals, alpha_vals,
        indexing='ij'
    ))
    
    grid_shape = param_grid.shape[1:]  # Shape of the 7D grid
    n_configs = np.prod(grid_shape)
    
    # Reshape to (7, n_configs) for vectorized computation
    params_flat = param_grid.reshape(7, -1)
    
    S0_flat = params_flat[0]
    T_flat = params_flat[1]
    r_flat = params_flat[2]
    q_flat = params_flat[3]
    K_flat = params_flat[4]
    sigma_flat = params_flat[5]
    alpha_flat = params_flat[6]
    
    print(f"Computing analytical prices for {n_configs:,} configurations...")
    
    # Validate constraints
    valid_mask = (
        (S0_flat + alpha_flat > 0) &
        (K_flat + alpha_flat > 0) &
        (T_flat > 0) &
        (sigma_flat > 0)
    )
    
    n_valid = np.sum(valid_mask)
    n_invalid = n_configs - n_valid
    
    if n_invalid > 0:
        print(f"Warning: {n_invalid} invalid configurations (will be set to NaN)")
    
    # Initialize call prices with NaN
    call_prices = np.full(n_configs, np.nan)
    
    # Compute only for valid configurations
    if n_valid > 0:
        # Extract valid parameters
        S0_v = S0_flat[valid_mask]
        T_v = T_flat[valid_mask]
        r_v = r_flat[valid_mask]
        q_v = q_flat[valid_mask]
        K_v = K_flat[valid_mask]
        sigma_v = sigma_flat[valid_mask]
        alpha_v = alpha_flat[valid_mask]
        
        # Vectorized computation of d1 and d2
        shifted_forward = (S0_v + alpha_v) * np.exp((r_v - q_v) * T_v)
        shifted_strike = K_v + alpha_v
        
        sqrt_T = np.sqrt(T_v)
        sigma_sqrt_T = sigma_v * sqrt_T
        
        d1 = (np.log(shifted_forward / shifted_strike) + 
              0.5 * sigma_v**2 * T_v) / sigma_sqrt_T
        d2 = d1 - sigma_sqrt_T
        
        # Vectorized call price computation
        call_prices_valid = (
            shifted_forward * np.exp(-r_v * T_v) * norm.cdf(d1) -
            shifted_strike * np.exp(-r_v * T_v) * norm.cdf(d2)
        )
        
        # Assign to output array
        call_prices[valid_mask] = call_prices_valid
    
    # Prepare parameter array for output
    param_combinations = params_flat.T  # Shape: (n_configs, 7)
    
    results = {
        'call_prices': call_prices if return_format == 'array' else call_prices.reshape(grid_shape),
        'parameters': param_combinations if return_format == 'array' else param_grid,
        'parameter_names': ['S0', 'T', 'r', 'q', 'K', 'sigma', 'alpha'],
        'valid_mask': valid_mask if return_format == 'array' else valid_mask.reshape(grid_shape),
        'n_total': n_configs,
        'n_valid': n_valid,
        'n_invalid': n_invalid
    }
    
    if return_format == 'grid':
        results['shape'] = grid_shape
        results['axes_values'] = {
            'S0': S0_vals,
            'T': T_vals,
            'r': r_vals,
            'q': q_vals,
            'K': K_vals,
            'sigma': sigma_vals,
            'alpha': alpha_vals
        }
    
    print(f"Computation complete: {n_valid}/{n_configs} valid prices computed")
    
    return results


def shifted_bs_call_single(
    S0: float, 
    K: float, 
    T: float, 
    r: float, 
    q: float, 
    sigma: float, 
    alpha: float
) -> float:
    """
    Compute single analytical European call price under shifted log-normal model.
    
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
    call_price : European call option price
    """
    
    # Handle edge cases
    if T <= 0:
        return max(S0 - K, 0)
    
    if S0 + alpha <= 0 or K + alpha <= 0:
        raise ValueError(f"Invalid parameters: S0+alpha={S0+alpha}, K+alpha={K+alpha} must be positive")
    
    if sigma <= 0:
        raise ValueError(f"Volatility must be positive: sigma={sigma}")
    
    # Shifted forward and strike
    shifted_forward = (S0 + alpha) * np.exp((r - q) * T)
    shifted_strike = K + alpha
    
    # d1 and d2
    sqrt_T = np.sqrt(T)
    d1 = (np.log(shifted_forward / shifted_strike) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    # Call price
    call_price = (
        shifted_forward * np.exp(-r * T) * norm.cdf(d1) -
        shifted_strike * np.exp(-r * T) * norm.cdf(d2)
    )
    
    return call_price


# Example usage and testing
if __name__ == "__main__":
    
    # Example 1: Small grid for demonstration
    print("="*70)
    print("EXAMPLE 1: Small Parameter Grid (Array Format)")
    print("="*70)
    
    results_array = compute_analytical_call_prices(
        S0_range=(95, 105, 3),
        T_range=(0.25, 1.0, 3),
        r_range=(0.03, 0.05, 2),
        q_range=(0.0, 0.02, 2),
        K_range=(95, 105, 3),
        sigma_range=(0.15, 0.30, 3),
        alpha_range=(-5, 5, 3),
        return_format='array'
    )
    
    print(f"\nResults Summary:")
    print(f"  Shape of call_prices: {results_array['call_prices'].shape}")
    print(f"  Valid prices: {results_array['n_valid']}/{results_array['n_total']}")
    print(f"\nCall Price Statistics (valid only):")
    valid_prices = results_array['call_prices'][results_array['valid_mask']]
    print(f"  Mean:   {np.mean(valid_prices):.4f}")
    print(f"  Std:    {np.std(valid_prices):.4f}")
    print(f"  Min:    {np.min(valid_prices):.4f}")
    print(f"  Max:    {np.max(valid_prices):.4f}")
    print(f"  Median: {np.median(valid_prices):.4f}")
    
    # Show first 5 configurations
    print(f"\nFirst 5 configurations:")
    print(f"{'S0':>8} {'T':>8} {'r':>8} {'q':>8} {'K':>8} {'sigma':>8} {'alpha':>8} {'Call':>10}")
    print("-"*70)
    for i in range(min(5, len(results_array['parameters']))):
        params = results_array['parameters'][i]
        price = results_array['call_prices'][i]
        print(f"{params[0]:8.2f} {params[1]:8.2f} {params[2]:8.4f} {params[3]:8.4f} "
              f"{params[4]:8.2f} {params[5]:8.4f} {params[6]:8.2f} {price:10.4f}")
    
    # Example 2: Grid format for sensitivity analysis
    print("\n" + "="*70)
    print("EXAMPLE 2: Grid Format (for sensitivity analysis)")
    print("="*70)
    
    results_grid = compute_analytical_call_prices(
        S0_range=(100, 100, 1),      # Fix S0 = 100
        T_range=(0.5, 0.5, 1),       # Fix T = 0.5
        r_range=(0.05, 0.05, 1),     # Fix r = 0.05
        q_range=(0.01, 0.01, 1),     # Fix q = 0.01
        K_range=(90, 110, 5),        # Vary K: 90, 95, 100, 105, 110
        sigma_range=(0.1, 0.5, 5),   # Vary sigma: 0.1 to 0.5
        alpha_range=(-10, 10, 5),    # Vary alpha: -10 to 10
        return_format='grid'
    )
    
    print(f"\nGrid shape: {results_grid['shape']}")
    print(f"Dimensions: (S0, T, r, q, K, sigma, alpha)")
    
    # Extract 2D slice: K vs sigma (with other params fixed)
    call_price_grid = results_grid['call_prices'][0, 0, 0, 0, :, :, 2]  # alpha=0 (middle value)
    K_values = results_grid['axes_values']['K']
    sigma_values = results_grid['axes_values']['sigma']
    
    print(f"\n2D Slice: Call Price vs (K, sigma) with alpha=0")
    print(f"{'K \\ sigma':>10}", end='')
    for sig in sigma_values:
        print(f"{sig:>10.2f}", end='')
    print()
    print("-"*70)
    for i, k in enumerate(K_values):
        print(f"{k:>10.2f}", end='')
        for j, sig in enumerate(sigma_values):
            print(f"{call_price_grid[i, j]:>10.4f}", end='')
        print()
    
    # Example 3: Single call price computation
    print("\n" + "="*70)
    print("EXAMPLE 3: Single Call Price Computation")
    print("="*70)
    
    single_price = shifted_bs_call_single(
        S0=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.25, alpha=0
    )
    print(f"Call price (at-the-money, no shift): {single_price:.6f}")
    
    single_price_shifted = shifted_bs_call_single(
        S0=100, K=100, T=1.0, r=0.05, q=0.02, sigma=0.25, alpha=-5
    )
    print(f"Call price (at-the-money, alpha=-5):  {single_price_shifted:.6f}")
    
    # Demonstrate impact of shift on ATM option
    print(f"\nImpact of negative shift: {single_price_shifted - single_price:+.6f}")
    print("(Negative shift increases call value for ATM options)")
