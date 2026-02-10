"""
Simplified version for quick testing and demonstration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.stats import norm


class ContinuationNet(nn.Module):
    """Neural network for continuation value approximation"""
    
    def __init__(self, input_dim=2, hidden_dim=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class AmericanPutPricer:
    """Simplified American Put pricer using deep learning"""
    
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=9):
        self.S0, self.K, self.r, self.sigma, self.T, self.N = S0, K, r, sigma, T, N
        self.dt = T / N
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cont_nets = []
        print(f"Device: {self.device}")
    
    def payoff(self, S):
        """Put payoff: max(K - S, 0)"""
        return torch.clamp(self.K - S, min=0)
    
    def disc_payoff(self, n, S):
        """Discounted payoff"""
        return np.exp(-self.r * n * self.dt) * self.payoff(S)
    
    def simulate_paths(self, K, seed=None):
        """Simulate K stock price paths"""
        if seed is not None:
            torch.manual_seed(seed)
        
        S = torch.zeros(K, self.N + 1, device=self.device)
        S[:, 0] = self.S0
        
        for i in range(self.N):
            Z = torch.randn(K, device=self.device)
            S[:, i+1] = S[:, i] * torch.exp(
                (self.r - 0.5*self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*Z
            )
        return S
    
    def train_continuation_values(self, K=50000, batch_size=4096, lr=1e-3):
        """Train continuation value networks (Longstaff-Schwartz with NN)"""
        print("\n=== Training Continuation Values ===")
        
        S_paths = self.simulate_paths(K, seed=42)
        stop_times = torch.full((K,), self.N, dtype=torch.long, device=self.device)
        
        for n in range(self.N-1, -1, -1):
            print(f"Time step {n}...")
            
            net = ContinuationNet().to(self.device)
            
            # Warm start from previous network
            if n < self.N-1:
                net.load_state_dict(self.cont_nets[0].state_dict())
                epochs = 2000
            else:
                epochs = 4000
            
            optimizer = optim.Adam(net.parameters(), lr=lr)
            
            # Training data
            S_n = S_paths[:, n].unsqueeze(1)
            payoff_n = self.disc_payoff(n, S_paths[:, n]).unsqueeze(1)
            X = torch.cat([S_n, payoff_n], dim=1)
            
            # Target: payoff at stopping time
            S_stop = S_paths[torch.arange(K), stop_times]
            y = self.disc_payoff(stop_times.cpu().numpy(), S_stop)
            
            # Train
            for epoch in range(epochs):
                perm = torch.randperm(K, device=self.device)
                epoch_loss = 0
                
                for i in range(0, K, batch_size):
                    idx = perm[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    pred = net(X[idx])
                    loss = nn.MSELoss()(pred, y[idx])
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                if (epoch+1) % 1000 == 0:
                    print(f"  Epoch {epoch+1}, Loss: {epoch_loss/(K/batch_size):.6f}")
            
            # Update stopping times
            with torch.no_grad():
                cont_val = net(X)
                exercise = self.payoff(S_paths[:, n]) >= cont_val
                stop_times = torch.where(exercise, n, stop_times)
            
            self.cont_nets.insert(0, net)
        
        # Add constant network for t=0
        with torch.no_grad():
            S_stop = S_paths[torch.arange(K), stop_times]
            c0 = self.disc_payoff(stop_times.cpu().numpy(), S_stop).mean().item()
        
        const_net = ContinuationNet().to(self.device)
        with torch.no_grad():
            for p in const_net.parameters():
                p.zero_()
            const_net.net[-1].bias[0] = c0
        
        self.cont_nets.insert(0, const_net)
        print("Training complete!")
    
    def compute_lower_bound(self, K=100000):
        """Compute lower bound using learned strategy"""
        print("\n=== Computing Lower Bound ===")
        
        S_paths = self.simulate_paths(K, seed=100)
        payoffs = []
        
        with torch.no_grad():
            for k in range(K):
                for n in range(self.N + 1):
                    if n == self.N:
                        payoff = self.disc_payoff(n, S_paths[k, n])
                        payoffs.append(payoff.item())
                        break
                    
                    S_n = S_paths[k, n].view(1, 1)
                    p_n = self.disc_payoff(n, S_paths[k, n]).view(1, 1)
                    X = torch.cat([S_n, p_n], dim=1)
                    
                    cont = self.cont_nets[n](X).item()
                    imm = self.payoff(S_paths[k, n]).item()
                    
                    if imm >= cont:
                        payoff = self.disc_payoff(n, S_paths[k, n])
                        payoffs.append(payoff.item())
                        break
        
        payoffs = np.array(payoffs)
        L = payoffs.mean()
        std_L = payoffs.std(ddof=1)
        
        print(f"Lower bound: {L:.6f} ± {1.96*std_L/np.sqrt(K):.6f}")
        return L, std_L
    
    def compute_upper_bound(self, K_outer=1024, K_inner=1024):
        """Compute upper bound using dual approach"""
        print("\n=== Computing Upper Bound ===")
        print("(Using nested simulation - this may take a minute)")
        
        S_outer = self.simulate_paths(K_outer, seed=200)
        max_vals = []
        
        with torch.no_grad():
            for k in range(K_outer):
                if (k+1) % 256 == 0:
                    print(f"  Path {k+1}/{K_outer}")
                
                M = torch.zeros(self.N + 1, device=self.device)
                
                for n in range(self.N):
                    # Nested simulation from S[k,n]
                    S_n = S_outer[k, n]
                    S_inner = torch.zeros(K_inner, self.N+1-n, device=self.device)
                    S_inner[:, 0] = S_n
                    
                    for i in range(self.N - n):
                        Z = torch.randn(K_inner, device=self.device)
                        S_inner[:, i+1] = S_inner[:, i] * torch.exp(
                            (self.r - 0.5*self.sigma**2)*self.dt + 
                            self.sigma*np.sqrt(self.dt)*Z
                        )
                    
                    # Evaluate continuation value via inner paths
                    inner_vals = []
                    for j in range(K_inner):
                        for m in range(n+1, self.N+1):
                            if m == self.N:
                                val = self.disc_payoff(m, S_inner[j, m-n])
                                inner_vals.append(val.item())
                                break
                            
                            S_m = S_inner[j, m-n].view(1, 1)
                            p_m = self.disc_payoff(m, S_inner[j, m-n]).view(1, 1)
                            X = torch.cat([S_m, p_m], dim=1)
                            
                            cont = self.cont_nets[m](X).item()
                            imm = self.payoff(S_inner[j, m-n]).item()
                            
                            if imm >= cont:
                                val = self.disc_payoff(m, S_inner[j, m-n])
                                inner_vals.append(val.item())
                                break
                    
                    M[n+1] = M[n] + (np.mean(inner_vals) - self.disc_payoff(n, S_n).item())
                
                # Compute max(G_n - M_n)
                max_val = max([
                    self.disc_payoff(n, S_outer[k, n]).item() - M[n].item()
                    for n in range(self.N + 1)
                ])
                max_vals.append(max_val)
        
        max_vals = np.array(max_vals)
        U = max_vals.mean()
        std_U = max_vals.std(ddof=1)
        
        print(f"Upper bound: {U:.6f} ± {1.96*std_U/np.sqrt(K_outer):.6f}")
        return U, std_U


def main():
    # Create pricer
    pricer = AmericanPutPricer(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=9)
    
    # Train continuation values
    pricer.train_continuation_values(K=50000, batch_size=4096)
    
    # Compute bounds
    L, std_L = pricer.compute_lower_bound(K=100000)
    U, std_U = pricer.compute_upper_bound(K_outer=512, K_inner=512)
    
    # Results
    V = (L + U) / 2
    ci_lower = L - 1.96 * std_L / np.sqrt(100000)
    ci_upper = U + 1.96 * std_U / np.sqrt(512)
    
    print("\n" + "="*50)
    print("AMERICAN PUT PRICING RESULTS")
    print("="*50)
    print(f"Lower bound:     {L:.6f}")
    print(f"Upper bound:     {U:.6f}")
    print(f"Point estimate:  {V:.6f}")
    print(f"95% CI:         [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"Spread:          {U - L:.6f}")
    print("="*50)


if __name__ == "__main__":
    main()
