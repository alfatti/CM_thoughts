\subsection*{Caplet Portfolio Used in the XVA Prototype}

The benchmark portfolio consists of a diversified set of caplets written on a high–dimensional forward-rate curve. The parameters are chosen so that the clean values are available in closed form while still producing realistic exposure dynamics.

\begin{itemize}

\item \textbf{Risk factors}

\begin{itemize}
\item Forward rate vector \(F(t)=(F_1(t),\dots,F_d(t))\)
\item Dimension \(d=80\) (quarterly forwards over a 20-year horizon)
\item Dynamics (under \(Q\)):
\[
dF_i(t)=\sigma F_i(t)dW_i^Q(t)
\]
\item Volatility \(\sigma=20\%\)
\item Correlation
\[
\rho_{ij}=\exp(-0.2|T_i-T_j|)
\]
\end{itemize}

\item \textbf{Discounting}

\begin{itemize}
\item Flat OIS curve
\[
r=3\%
\]
\item Discount factor \(P(0,t)=e^{-rt}\)
\end{itemize}

\item \textbf{Portfolio composition}

\begin{itemize}
\item Caplets written on all tenors \(i=1,\dots,d\)
\item Five caplets per tenor
\item Total portfolio size
\[
M=5d=400
\]
\end{itemize}

\item \textbf{Caplet payoff}

\[
N\,\delta_i\,(F_i(T_i)-K)^+
\]

\begin{itemize}
\item Accrual \(\delta_i=0.25\)
\item Payment at \(T_{i+1}\)
\item Clean price computed via Black's formula
\end{itemize}

\item \textbf{Strike selection}

For each tenor \(i\):

\[
K \in \{F_i(0)-100\text{bp},\;F_i(0)-50\text{bp},\;F_i(0),\;F_i(0)+50\text{bp},\;F_i(0)+100\text{bp}\}
\]

\item \textbf{Notionals}

\[
N = 10^6
\]

Position sign \(s_m\in\{+1,-1\}\) sampled randomly to generate netting effects.

\item \textbf{Credit parameters}

\begin{itemize}
\item Counterparty intensity
\[
\lambda^C=1.5\%
\]
\item Recovery
\[
R_C=40\%
\]
\end{itemize}

\item \textbf{Funding spreads}

\begin{itemize}
\item Borrowing
\[
r^{f,b}=r+100\text{bp}
\]
\item Lending
\[
r^{f,l}=r
\]
\end{itemize}

\item \textbf{Collateral rule}

Threshold CSA with

\[
H=10^6
\]

\[
C_t =
\begin{cases}
\bar V_t-H & \bar V_t>H \\
0 & |\bar V_t|\le H \\
\bar V_t+H & \bar V_t<-H
\end{cases}
\]

\end{itemize}
