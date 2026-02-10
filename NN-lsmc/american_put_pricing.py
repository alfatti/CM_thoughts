"""
Deep Learning for Pricing and Hedging American Put Options
Based on: Becker, Cheridito, Jentzen (2019)
"Pricing and hedging American-style options with deep learning"

Implementation for 1-dimensional American Put option
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


class ContinuationValueNetwork(nn.Module):
    """Neural network to approximate continuation values"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 50, depth: int = 2):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class HedgingNetwork(nn.Module):
    """Neural network to approximate hedging positions"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 50, depth: int = 2):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Tanh())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class AmericanPutPricer:
    """
    Deep learning method for pricing and hedging American Put options
    """
    
    def __init__(
        self,
        S0: float = 100.0,      # Initial stock price
        K: float = 100.0,       # Strike price
        r: float = 0.05,        # Risk-free rate
        sigma: float = 0.2,     # Volatility
        T: float = 1.0,         # Maturity
        N: int = 9,             # Number of exercise dates
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.dt = T / N
        self.device = device
        
        # Storage for trained continuation value networks
        self.cont_networks = []
        
        print(f"Using device: {device}")
    
    def payoff(self, S: torch.Tensor) -> torch.Tensor:
        """Payoff function for American Put: max(K - S, 0)"""
        return torch.maximum(self.K - S, torch.tensor(0.0, device=self.device))
    
    def discounted_payoff(self, n: int, S: torch.Tensor) -> torch.Tensor:
        """Discounted payoff at time step n"""
        discount = np.exp(-self.r * n * self.dt)
        return discount * self.payoff(S)
    
    def simulate_paths(self, K: int, seed: int = None) -> torch.Tensor:
        """
        Simulate K paths of the underlying stock price
        Returns: tensor of shape (K, N+1)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate Brownian increments
        dW = torch.randn(K, self.N, device=self.device) * np.sqrt(self.dt)
        
        # Initialize paths
        S = torch.zeros(K, self.N + 1, device=self.device)
        S[:, 0] = self.S0
        
        # Generate paths using geometric Brownian motion
        for i in range(self.N):
            S[:, i + 1] = S[:, i] * torch.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * dW[:, i]
            )
        
        return S
    
    def train_continuation_values(
        self,
        K_train: int = 100000,
        batch_size: int = 8192,
        epochs_first: int = 6000,
        epochs_other: int = 3500,
        learning_rate: float = 1e-3,
        seed: int = 42
    ):
        """
        Train continuation value networks using Longstaff-Schwartz with neural networks
        """
        print("\n=== Training Continuation Value Networks ===")
        
        # Simulate training paths
        print(f"Simulating {K_train} training paths...")
        S_paths = self.simulate_paths(K_train, seed=seed)
        
        # Storage for optimal stopping times along training paths
        stopping_times = torch.full((K_train,), self.N, dtype=torch.long, device=self.device)
        
        # Initialize continuation value networks
        self.cont_networks = []
        
        # Work backwards from N-1 to 0
        for n in range(self.N - 1, -1, -1):
            print(f"\nTraining network for time step {n}...")
            
            # Create network with augmented state (S, discounted_payoff)
            net = ContinuationValueNetwork(input_dim=2, hidden_dim=50, depth=2).to(self.device)
            
            if n == self.N - 1:
                # First network: train from scratch
                optimizer = optim.Adam(net.parameters(), lr=learning_rate)
                epochs = epochs_first
            else:
                # Subsequent networks: warm start from previous network
                net.load_state_dict(self.cont_networks[0].state_dict())
                optimizer = optim.Adam(net.parameters(), lr=learning_rate)
                epochs = epochs_other
            
            # Prepare training data
            S_n = S_paths[:, n].unsqueeze(1)  # Stock price at time n
            payoff_n = self.discounted_payoff(n, S_paths[:, n]).unsqueeze(1)
            
            # Augmented state
            X_n = torch.cat([S_n, payoff_n], dim=1)
            
            # Target: discounted payoff at stopping time
            S_stop = S_paths[torch.arange(K_train), stopping_times]
            G_stop = self.discounted_payoff(stopping_times.cpu().numpy(), S_stop)
            
            # Training loop
            num_batches = (K_train + batch_size - 1) // batch_size
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                perm = torch.randperm(K_train, device=self.device)
                
                for i in range(num_batches):
                    idx = perm[i * batch_size:(i + 1) * batch_size]
                    
                    optimizer.zero_grad()
                    pred = net(X_n[idx]).squeeze()
                    target = G_stop[idx]
                    
                    loss = nn.MSELoss()(pred, target)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                if (epoch + 1) % 1000 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Update stopping times for next iteration
            with torch.no_grad():
                cont_value = net(X_n).squeeze()
                immediate_payoff = self.payoff(S_paths[:, n])
                exercise = immediate_payoff >= cont_value
                stopping_times = torch.where(exercise, n, stopping_times)
            
            # Store network (insert at beginning for proper ordering)
            self.cont_networks.insert(0, net)
        
        # Compute c_theta_0 as average of first exercise payoffs
        with torch.no_grad():
            S_stop = S_paths[torch.arange(K_train), stopping_times]
            G_stop = self.discounted_payoff(stopping_times.cpu().numpy(), S_stop)
            c_0 = G_stop.mean().item()
        
        # Create constant network for time 0
        const_net = ContinuationValueNetwork(input_dim=2, hidden_dim=50, depth=2).to(self.device)
        
        # Set all parameters to produce constant output c_0
        with torch.no_grad():
            for param in const_net.parameters():
                param.zero_()
            # Set final bias to c_0
            const_net.network[-1].bias[0] = c_0
        
        self.cont_networks.insert(0, const_net)
        
        print("\n=== Continuation Value Training Complete ===")
    
    def compute_lower_bound(self, K_L: int = 4096000, seed: int = 100) -> Tuple[float, float]:
        """
        Compute lower bound estimate using learned stopping strategy
        Returns: (lower_bound_estimate, standard_error)
        """
        print("\n=== Computing Lower Bound ===")
        
        S_paths = self.simulate_paths(K_L, seed=seed)
        payoffs = []
        
        with torch.no_grad():
            for k in range(K_L):
                for n in range(self.N + 1):
                    if n == self.N:
                        # Must exercise at final time
                        payoff = self.discounted_payoff(n, S_paths[k, n])
                        payoffs.append(payoff.item())
                        break
                    else:
                        # Check exercise condition
                        S_n = S_paths[k, n].unsqueeze(0).unsqueeze(0)
                        payoff_n = self.discounted_payoff(n, S_paths[k, n]).unsqueeze(0).unsqueeze(0)
                        X_n = torch.cat([S_n, payoff_n], dim=1)
                        
                        cont_value = self.cont_networks[n](X_n).item()
                        immediate_payoff = self.payoff(S_paths[k, n]).item()
                        
                        if immediate_payoff >= cont_value:
                            # Exercise now
                            payoff = self.discounted_payoff(n, S_paths[k, n])
                            payoffs.append(payoff.item())
                            break
        
        payoffs = np.array(payoffs)
        L_hat = payoffs.mean()
        sigma_L = payoffs.std(ddof=1)
        
        print(f"Lower bound estimate: {L_hat:.6f}")
        print(f"Standard error: {sigma_L / np.sqrt(K_L):.6f}")
        
        return L_hat, sigma_L
    
    def compute_upper_bound(
        self,
        K_U_outer: int = 2048,
        K_U_inner: int = 2048,
        seed: int = 200
    ) -> Tuple[float, float]:
        """
        Compute upper bound using dual approach with nested simulation
        Returns: (upper_bound_estimate, standard_error)
        """
        print("\n=== Computing Upper Bound (this may take a while) ===")
        
        # Outer paths
        S_outer = self.simulate_paths(K_U_outer, seed=seed)
        max_values = []
        
        with torch.no_grad():
            for k in range(K_U_outer):
                if (k + 1) % 500 == 0:
                    print(f"  Processing outer path {k + 1}/{K_U_outer}")
                
                # Compute martingale increments using nested simulation
                M = torch.zeros(self.N + 1, device=self.device)
                
                for n in range(self.N):
                    # Inner simulation for martingale increment
                    S_n = S_outer[k, n]
                    
                    # Simulate from current state
                    S_inner = torch.zeros(K_U_inner, self.N + 1 - n, device=self.device)
                    S_inner[:, 0] = S_n
                    
                    for i in range(self.N - n):
                        dW = torch.randn(K_U_inner, device=self.device) * np.sqrt(self.dt)
                        S_inner[:, i + 1] = S_inner[:, i] * torch.exp(
                            (self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * dW
                        )
                    
                    # Compute continuation value via nested Monte Carlo
                    inner_values = []
                    for j in range(K_U_inner):
                        for m in range(n + 1, self.N + 1):
                            if m == self.N:
                                val = self.discounted_payoff(m, S_inner[j, m - n])
                                inner_values.append(val.item())
                                break
                            else:
                                S_m = S_inner[j, m - n].unsqueeze(0).unsqueeze(0)
                                payoff_m = self.discounted_payoff(m, S_inner[j, m - n]).unsqueeze(0).unsqueeze(0)
                                X_m = torch.cat([S_m, payoff_m], dim=1)
                                
                                cont_val = self.cont_networks[m](X_m).item()
                                imm_payoff = self.payoff(S_inner[j, m - n]).item()
                                
                                if imm_payoff >= cont_val:
                                    val = self.discounted_payoff(m, S_inner[j, m - n])
                                    inner_values.append(val.item())
                                    break
                    
                    # Martingale increment
                    M[n + 1] = M[n] + (np.mean(inner_values) - self.discounted_payoff(n, S_n).item())
                
                # Compute max(G_n - M_n)
                max_val = -float('inf')
                for n in range(self.N + 1):
                    val = self.discounted_payoff(n, S_outer[k, n]).item() - M[n].item()
                    max_val = max(max_val, val)
                
                max_values.append(max_val)
        
        max_values = np.array(max_values)
        U_hat = max_values.mean()
        sigma_U = max_values.std(ddof=1)
        
        print(f"Upper bound estimate: {U_hat:.6f}")
        print(f"Standard error: {sigma_U / np.sqrt(K_U_outer):.6f}")
        
        return U_hat, sigma_U
    
    def compute_confidence_interval(
        self,
        L_hat: float,
        sigma_L: float,
        K_L: int,
        U_hat: float,
        sigma_U: float,
        K_U: int,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Compute 95% confidence interval for the option price"""
        from scipy.stats import norm
        
        z = norm.ppf(1 - alpha / 2)
        
        lower = L_hat - z * sigma_L / np.sqrt(K_L)
        upper = U_hat + z * sigma_U / np.sqrt(K_U)
        
        return lower, upper
    
    def train_hedging_strategy(
        self,
        M: int = 24,  # Number of rehedging times between exercise dates
        K_H: int = 100000,
        batch_size: int = 8192,
        epochs: int = 10000,
        learning_rate: float = 1e-2,
        seed: int = 300
    ):
        """
        Train hedging strategy networks
        M: number of rebalancing times between consecutive exercise dates
        """
        print("\n=== Training Hedging Strategy ===")
        
        # Time grid for hedging
        total_steps = self.N * M
        dt_hedge = self.T / total_steps
        
        # Simulate paths on fine grid
        print(f"Simulating {K_H} paths on fine grid...")
        torch.manual_seed(seed)
        
        dW = torch.randn(K_H, total_steps, device=self.device) * np.sqrt(dt_hedge)
        S_fine = torch.zeros(K_H, total_steps + 1, device=self.device)
        S_fine[:, 0] = self.S0
        
        for i in range(total_steps):
            S_fine[:, i + 1] = S_fine[:, i] * torch.exp(
                (self.r - 0.5 * self.sigma**2) * dt_hedge + self.sigma * dW[:, i]
            )
        
        # Discounted stock prices for hedging
        P_fine = torch.zeros_like(S_fine)
        for i in range(total_steps + 1):
            P_fine[:, i] = S_fine[:, i] * np.exp(-self.r * i * dt_hedge)
        
        # Hedging networks (one for each time step)
        self.hedge_networks = []
        
        # Train networks sequentially
        for m in range(total_steps):
            print(f"\nTraining hedging network for time step {m}/{total_steps}...")
            
            net = HedgingNetwork(input_dim=1, output_dim=1, hidden_dim=50, depth=2).to(self.device)
            
            if m >= M:
                # Warm start from network M steps ago
                net.load_state_dict(self.hedge_networks[m - M].state_dict())
            
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
            
            # Determine epochs based on whether this is a fresh network
            n_epochs = epochs if m < M else 3000
            
            # Prepare features and targets
            S_m = P_fine[:, m].unsqueeze(1)
            
            # Compute target value at next exercise date or current step
            if m < total_steps:
                # Target is value at next step
                next_idx = min(m + M, total_steps)
                n_next = next_idx // M
                
                with torch.no_grad():
                    target_values = []
                    for k in range(K_H):
                        S_next = S_fine[k, next_idx]
                        payoff_next = self.discounted_payoff(n_next, S_next)
                        
                        if n_next < self.N:
                            S_aug = S_next.unsqueeze(0).unsqueeze(0)
                            payoff_aug = payoff_next.unsqueeze(0).unsqueeze(0)
                            X_aug = torch.cat([S_aug, payoff_aug], dim=1)
                            cont_val = self.cont_networks[n_next](X_aug).item()
                            val = max(payoff_next.item(), cont_val)
                        else:
                            val = payoff_next.item()
                        
                        target_values.append(val)
                    
                    targets = torch.tensor(target_values, device=self.device)
            
            # Training loop
            num_batches = (K_H + batch_size - 1) // batch_size
            
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                perm = torch.randperm(K_H, device=self.device)
                
                for i in range(num_batches):
                    idx = perm[i * batch_size:(i + 1) * batch_size]
                    
                    optimizer.zero_grad()
                    
                    # Compute cumulative gains
                    cum_gains = torch.zeros(len(idx), device=self.device)
                    for j in range(m + 1):
                        if j < len(self.hedge_networks):
                            h = self.hedge_networks[j](P_fine[idx, j].unsqueeze(1)).squeeze()
                            price_change = P_fine[idx, j + 1] - P_fine[idx, j]
                            cum_gains += h * price_change
                    
                    # For current step
                    h_current = net(S_m[idx]).squeeze()
                    if m < total_steps:
                        price_change = P_fine[idx, m + 1] - P_fine[idx, m]
                        cum_gains += h_current * price_change
                    
                    # Loss: squared difference from target
                    pred = cum_gains
                    loss = nn.MSELoss()(pred, targets[idx])
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                if (epoch + 1) % 2000 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")
            
            self.hedge_networks.append(net)
        
        print("\n=== Hedging Strategy Training Complete ===")


def main():
    """Example usage"""
    
    # American Put parameters
    pricer = AmericanPutPricer(
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        N=9
    )
    
    # Step 1: Train continuation value networks
    pricer.train_continuation_values(
        K_train=100000,
        epochs_first=6000,
        epochs_other=3500
    )
    
    # Step 2: Compute lower bound
    L_hat, sigma_L = pricer.compute_lower_bound(K_L=100000)
    
    # Step 3: Compute upper bound (reduced sample size for faster computation)
    U_hat, sigma_U = pricer.compute_upper_bound(
        K_U_outer=512,
        K_U_inner=512
    )
    
    # Step 4: Compute point estimate and confidence interval
    V_hat = (L_hat + U_hat) / 2
    ci_lower, ci_upper = pricer.compute_confidence_interval(
        L_hat, sigma_L, 100000,
        U_hat, sigma_U, 512
    )
    
    print("\n" + "="*50)
    print("PRICING RESULTS")
    print("="*50)
    print(f"Lower bound:      {L_hat:.6f}")
    print(f"Upper bound:      {U_hat:.6f}")
    print(f"Point estimate:   {V_hat:.6f}")
    print(f"95% CI:          [{ci_lower:.6f}, {ci_upper:.6f}]")
    print("="*50)
    
    # Step 5: Train hedging strategy (optional, can be slow)
    # pricer.train_hedging_strategy(M=24, K_H=50000, epochs=5000)


if __name__ == "__main__":
    main()
