\subsection*{Caplet Portfolio Specification}

\begin{table}[h]
\centering
\begin{tabular}{ll}
\hline
\textbf{Component} & \textbf{Specification} \\
\hline

Risk factors & Forward rates $F(t)=(F_1(t),\dots,F_d(t))$ \\

Dimension & $d = 80$ (quarterly tenors over a 20-year horizon) \\

Dynamics &
$dF_i(t)=\sigma F_i(t)dW_i^Q(t)$ \\

Volatility & $\sigma = 20\%$ \\

Correlation &
$\rho_{ij} = \exp(-0.2|T_i-T_j|)$ \\

Discount curve & Flat OIS curve $P(0,t)=e^{-rt}$ \\

Reference rate & $r = 3\%$ \\

Portfolio size & $M = 5d = 400$ caplets \\

Caplet payoff &
$N\,\delta_i\,(F_i(T_i)-K)^+$ paid at $T_{i+1}$ \\

Accrual & $\delta_i = 0.25$ \\

Strike selection &
$K \in \{F_i(0)-100\text{bp},\,F_i(0)-50\text{bp},\,F_i(0),\,F_i(0)+50\text{bp},\,F_i(0)+100\text{bp}\}$ \\

Notional & $N = 10^6$ \\

Position sign & $s_m \in \{+1,-1\}$ sampled randomly \\

Counterparty intensity & $\lambda^C = 1.5\%$ \\

Recovery rate & $R_C = 40\%$ \\

Funding spread (borrow) & $r^{f,b} = r + 100$ bp \\

Funding spread (lend) & $r^{f,l} = r$ \\

Collateral threshold & $H = 10^6$ \\

\hline
\end{tabular}
\end{table}
