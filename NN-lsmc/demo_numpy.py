"""
Conceptual demonstration of the deep learning approach for American Put pricing
This version uses NumPy to illustrate the algorithm structure without requiring PyTorch
"""

import numpy as np
from scipy.stats import norm


class SimpleNeuralNet:
    """
    Simplified neural network implementation using NumPy
    This is a minimal implementation for demonstration purposes
    """
    
    def __init__(self, input_dim=2, hidden_dim=20, output_dim=1, seed=42):
        np.random.seed(seed)
        
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim)
        
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b3 = np.zeros(output_dim)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def forward(self, X):
        """Forward pass through the network"""
        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.tanh(self.z1)
        
        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.tanh(self.z2)
        
        # Output layer
        self.z3 = self.a2 @ self.W3 + self.b3
        
        return self.z3.flatten()
    
    def backward(self, X, y, learning_rate=0.001):
        """Backward pass with gradient descent update"""
        m = X.shape[0]
        
        # Forward pass
        output = self.forward(X)
        
        # Compute loss gradient
        dz3 = (output - y).reshape(-1, 1) / m
        
        # Backpropagate through output layer
        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0)
        da2 = dz3 @ self.W3.T
        
        # Backpropagate through layer 2
        dz2 = da2 * self.tanh_derivative(self.z2)
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        
        # Backpropagate through layer 1
        dz1 = da1 * self.tanh_derivative(self.z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)
        
        # Update weights
        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        
        # Return loss
        return np.mean((output - y)**2)
    
    def fit(self, X, y, epochs=1000, learning_rate=0.001, verbose=True):
        """Train the network"""
        for epoch in range(epochs):
            loss = self.backward(X, y, learning_rate)
            
            if verbose and (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)


class AmericanPutPricerNumPy:
    """
    American Put option pricer using neural networks
    NumPy-only implementation for demonstration
    """
    
    def __init__(self, S0=100, K=100, r=0.05, sigma=0.2, T=1.0, N=9):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N
        self.dt = T / N
        
        self.cont_nets = []
        
        print(f"American Put Option Parameters:")
        print(f"  S0={S0}, K={K}, r={r}, σ={sigma}, T={T}, N={N}")
    
    def payoff(self, S):
        """Put option payoff: max(K - S, 0)"""
        return np.maximum(self.K - S, 0)
    
    def disc_payoff(self, n, S):
        """Discounted payoff at time step n"""
        discount = np.exp(-self.r * n * self.dt)
        return discount * self.payoff(S)
    
    def simulate_paths(self, K, seed=None):
        """
        Simulate K paths of the stock price using geometric Brownian motion
        Returns: array of shape (K, N+1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate standard normal increments
        Z = np.random.randn(K, self.N)
        
        # Initialize paths
        S = np.zeros((K, self.N + 1))
        S[:, 0] = self.S0
        
        # Generate paths
        for i in range(self.N):
            S[:, i+1] = S[:, i] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt + 
                self.sigma * np.sqrt(self.dt) * Z[:, i]
            )
        
        return S
    
    def train_continuation_values(self, K=10000, epochs=1000, lr=0.01):
        """
        Train continuation value networks using Longstaff-Schwartz with neural networks
        This is the core algorithm from the paper (Section 2)
        """
        print("\n" + "="*60)
        print("TRAINING CONTINUATION VALUE NETWORKS")
        print("="*60)
        
        # Simulate training paths
        print(f"\nSimulating {K} training paths...")
        S_paths = self.simulate_paths(K, seed=42)
        
        # Initialize stopping times (all paths exercise at maturity initially)
        stop_times = np.full(K, self.N, dtype=int)
        
        # Backward recursion from N-1 to 0
        for n in range(self.N - 1, -1, -1):
            print(f"\n--- Time step {n}/{self.N-1} ---")
            
            # Create neural network for continuation value approximation
            net = SimpleNeuralNet(input_dim=2, hidden_dim=20, seed=42+n)
            
            # Prepare augmented state: (S_n, discounted_payoff_n)
            S_n = S_paths[:, n].reshape(-1, 1)
            payoff_n = self.disc_payoff(n, S_paths[:, n]).reshape(-1, 1)
            X = np.hstack([S_n, payoff_n])
            
            # Target: discounted payoff at current stopping time
            S_stop = S_paths[np.arange(K), stop_times]
            y = self.disc_payoff(stop_times, S_stop)
            
            # Train network to predict continuation value
            print(f"  Training network (input: S and payoff, output: continuation value)...")
            net.fit(X, y, epochs=epochs, learning_rate=lr, verbose=True)
            
            # Update stopping times based on learned continuation values
            cont_values = net.predict(X)
            immediate_payoff = self.payoff(S_paths[:, n])
            
            # Exercise if immediate payoff >= continuation value
            exercise = immediate_payoff >= cont_values
            stop_times = np.where(exercise, n, stop_times)
            
            n_exercise = np.sum(exercise)
            print(f"  Paths exercising at time {n}: {n_exercise}/{K} ({100*n_exercise/K:.1f}%)")
            
            # Store network
            self.cont_nets.insert(0, net)
        
        # Create constant network for time 0
        S_stop = S_paths[np.arange(K), stop_times]
        c0 = np.mean(self.disc_payoff(stop_times, S_stop))
        
        const_net = SimpleNeuralNet(input_dim=2, hidden_dim=20)
        # Set all weights to 0 and output bias to c0
        const_net.W1[:] = 0
        const_net.W2[:] = 0
        const_net.W3[:] = 0
        const_net.b3[0] = c0
        
        self.cont_nets.insert(0, const_net)
        
        print("\n" + "="*60)
        print("CONTINUATION VALUE TRAINING COMPLETE")
        print("="*60)
    
    def compute_lower_bound(self, K=50000):
        """
        Compute lower bound by applying learned stopping strategy
        This implements Section 3.1 of the paper
        """
        print("\n" + "="*60)
        print("COMPUTING LOWER BOUND")
        print("="*60)
        
        # Simulate fresh paths
        print(f"\nSimulating {K} paths for lower bound estimation...")
        S_paths = self.simulate_paths(K, seed=100)
        
        payoffs = []
        
        # Apply learned stopping strategy
        for k in range(K):
            for n in range(self.N + 1):
                if n == self.N:
                    # Must exercise at maturity
                    payoff = self.disc_payoff(n, S_paths[k, n])
                    payoffs.append(payoff)
                    break
                else:
                    # Check if we should exercise
                    S_n = S_paths[k, n].reshape(1, 1)
                    p_n = self.disc_payoff(n, S_paths[k, n]).reshape(1, 1)
                    X = np.hstack([S_n, p_n])
                    
                    cont_value = self.cont_nets[n].predict(X)[0]
                    immediate_payoff = self.payoff(S_paths[k, n])
                    
                    if immediate_payoff >= cont_value:
                        # Exercise now
                        payoff = self.disc_payoff(n, S_paths[k, n])
                        payoffs.append(payoff)
                        break
        
        payoffs = np.array(payoffs)
        L = payoffs.mean()
        std_L = payoffs.std(ddof=1)
        se_L = std_L / np.sqrt(K)
        
        print(f"\nLower Bound Results:")
        print(f"  Estimate: {L:.6f}")
        print(f"  Std Dev:  {std_L:.6f}")
        print(f"  Std Err:  {se_L:.6f}")
        print(f"  95% CI:   [{L - 1.96*se_L:.6f}, {L + 1.96*se_L:.6f}]")
        
        return L, std_L
    
    def compute_upper_bound_simple(self, K=10000):
        """
        Simplified upper bound computation (without full nested simulation)
        Uses a simpler approximation for demonstration purposes
        """
        print("\n" + "="*60)
        print("COMPUTING UPPER BOUND (SIMPLIFIED)")
        print("="*60)
        print("Note: Using simplified method for demo (not full nested simulation)")
        
        S_paths = self.simulate_paths(K, seed=200)
        
        # Compute upper bound using Andersen-Broadie approach
        # This is a simplified version - full implementation requires nested simulation
        max_vals = []
        
        for k in range(K):
            path_max = -np.inf
            
            for n in range(self.N + 1):
                # Payoff at time n
                payoff = self.disc_payoff(n, S_paths[k, n])
                
                # Simple martingale approximation (not the full nested method)
                if n < self.N:
                    S_n = S_paths[k, n].reshape(1, 1)
                    p_n = self.disc_payoff(n, S_paths[k, n]).reshape(1, 1)
                    X = np.hstack([S_n, p_n])
                    cont = self.cont_nets[n].predict(X)[0]
                    
                    # Approximate martingale correction
                    M_approx = cont - payoff
                else:
                    M_approx = 0
                
                val = payoff - M_approx * 0.5  # Damping factor for stability
                path_max = max(path_max, val)
            
            max_vals.append(path_max)
        
        max_vals = np.array(max_vals)
        U = max_vals.mean()
        std_U = max_vals.std(ddof=1)
        se_U = std_U / np.sqrt(K)
        
        print(f"\nUpper Bound Results:")
        print(f"  Estimate: {U:.6f}")
        print(f"  Std Dev:  {std_U:.6f}")
        print(f"  Std Err:  {se_U:.6f}")
        print(f"  95% CI:   [{U - 1.96*se_U:.6f}, {U + 1.96*se_U:.6f}]")
        
        return U, std_U


def main():
    """Demonstration of the deep learning approach for American Put pricing"""
    
    print("\n" + "="*60)
    print("DEEP LEARNING FOR AMERICAN PUT OPTION PRICING")
    print("Based on: Becker, Cheridito, Jentzen (2019)")
    print("="*60)
    
    # Create pricer with standard parameters
    pricer = AmericanPutPricerNumPy(
        S0=100,      # Initial stock price
        K=100,       # Strike price
        r=0.05,      # Risk-free rate (5%)
        sigma=0.2,   # Volatility (20%)
        T=1.0,       # Maturity (1 year)
        N=9          # 9 exercise opportunities
    )
    
    # Step 1: Train continuation value networks
    # This is the core of the algorithm - learns when to exercise optimally
    pricer.train_continuation_values(
        K=10000,     # Number of training paths
        epochs=1000, # Training epochs per network
        lr=0.01      # Learning rate
    )
    
    # Step 2: Compute lower bound
    # Apply learned strategy to fresh simulations
    L, std_L = pricer.compute_lower_bound(K=50000)
    
    # Step 3: Compute upper bound
    # Uses dual approach (simplified version here)
    U, std_U = pricer.compute_upper_bound_simple(K=10000)
    
    # Step 4: Final results
    V = (L + U) / 2
    ci_lower = L - 1.96 * std_L / np.sqrt(50000)
    ci_upper = U + 1.96 * std_U / np.sqrt(10000)
    
    print("\n" + "="*60)
    print("FINAL PRICING RESULTS")
    print("="*60)
    print(f"Lower Bound:      {L:.6f}")
    print(f"Upper Bound:      {U:.6f}")
    print(f"Point Estimate:   {V:.6f}")
    print(f"Spread:           {U - L:.6f}")
    print(f"95% CI:          [{ci_lower:.6f}, {ci_upper:.6f}]")
    print("="*60)
    
    # Reference value (from binomial tree)
    print(f"\nReference American Put value (binomial tree): ~6.05")
    print("\nNote: This NumPy demo uses simplified networks and fewer paths")
    print("      for demonstration. The full PyTorch version produces more")
    print("      accurate results matching the paper.")


if __name__ == "__main__":
    main()
